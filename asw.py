"""Shared building blocks for the Anchored Sliding Window (ASW) framework.

Everything that decides *which distribution the language model produces at a
given step* lives here, so that embedding (3.Embed_AC.py) and extraction
(4.Extract_AC.py) are guaranteed to see the same numbers.  If those two ever
diverge by a single logit the extracted message is destroyed, so the window is
built in exactly one place.

Reference: "Anchored Sliding Window: Toward Robust and Imperceptible Linguistic
Steganography" (ACL 2026), Section 3 and Algorithms 1-4.
"""

import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------- #
# Bridge contexts
# --------------------------------------------------------------------------- #

# Hard bridge contexts evaluated in Table 1.  ``Hard_0`` is the ablation with no
# bridge at all (prompt directly followed by the latest tokens).
HARD_BRIDGES = {
    'Hard_0': '',
    'Hard_1': '[CONTEXT TRUNCATED]\n',
    'Hard_2': '...\n',
    'Hard_missing': '[Some texts are missing here]\n',
    'Hard_removed': '[previous message removed]\n',
}

SOFT_STRATEGIES = ('Soft_0', 'Soft_forward', 'Soft_reverse')

# ``Rand_<k>`` is the "random k tokens" control of Table 1: a bridge of k tokens
# drawn uniformly from the vocabulary, which isolates "the prompt is anchored"
# from "the bridge says something useful".
RAND_PREFIX = 'Rand_'

ALL_STRATEGIES = tuple(HARD_BRIDGES) + SOFT_STRATEGIES + ('Baseline',)


# ``Text_<anything>`` carries its bridge string alongside it, for sweeping
# candidate hard bridges without touching HARD_BRIDGES.
TEXT_PREFIX = 'Text_'


def is_strategy(name):
    return (name in ALL_STRATEGIES
            or name.startswith(TEXT_PREFIX)
            or (name.startswith(RAND_PREFIX) and name[len(RAND_PREFIX):].isdigit()))


def strategy_arg(name):
    """argparse type: accepts the fixed strategies plus ``Rand_<k>``."""
    import argparse
    if not is_strategy(name):
        raise argparse.ArgumentTypeError(
            f'{name!r} is not a strategy; expected one of {", ".join(ALL_STRATEGIES)} '
            f'or {RAND_PREFIX}<k>')
    return name


DTYPES = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_name, dtype='float32', lora_path=None, trim_vocab=True):
    """Load the LM and tokenizer used for embedding, extraction and evaluation.

    ``dtype`` defaults to float32, because the arithmetic coder needs the encoder
    and the decoder to agree on every logit.  Lower it only for experiments that
    never round-trip a message (e.g. KL measurement).

    ``trim_vocab`` shrinks the output embedding down to the tokenizer's actual
    vocabulary.  Checkpoints pad ``vocab_size`` beyond the tokenizer (Qwen2.5
    pads 151665 -> 151936); trimming keeps the sampler on tokens that tokenize
    back (Appendix H, "unreachable tokens").
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype='auto', device_map='auto'
    ).to(DTYPES[dtype])

    if lora_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    if trim_vocab:
        vocab_size = len(tokenizer.get_vocab())
        if model.get_output_embeddings().weight.shape[0] != vocab_size:
            model.resize_token_embeddings(vocab_size)

    model.eval()
    return model, tokenizer


def chat_prompt(tokenizer, question, system=None):
    """Wrap a raw question in the chat template, as in every experiment script."""
    if system is None:
        system = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_soft_bridge(path, model, soft_length=None):
    """Load a trained soft bridge context.

    Accepts both formats the training scripts emit: a bare tensor from
    1.Prompt_tuning.py and a ``SoftPromptEmbedding`` state dict from
    1.Prompt_tuning_LoRA.py.
    """
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict):
        obj = obj['soft_prompt']
    if soft_length is not None and obj.shape[0] != soft_length:
        raise ValueError(
            f'{path} holds a bridge of length {obj.shape[0]}, but --default_soft_length '
            f'is {soft_length}. Embedding and extraction must agree on the length.'
        )
    return obj.detach().to(device=model.device, dtype=model.dtype)


def bridge_fingerprint(tensor):
    """Content hash of a soft bridge context.

    The run config records the checkpoint path; the hash records what that file
    held at the time.  Extraction compares the two so that it runs against the
    same bridge the stegotexts were produced with.
    """
    import hashlib
    return hashlib.sha256(
        tensor.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The window itself
# --------------------------------------------------------------------------- #

class ASWWindow:
    """Produces the next-token distribution for one steganographic strategy.

    The context window is ``prompt || bridge || latest w tokens`` (Equation 4).
    The prompt and the bridge never change while a text is being generated, so
    their key/value cache is computed once per sample and reused for every step;
    only the ``w`` latest tokens are re-run.  This is numerically identical to a
    plain forward pass, which ``verify_cache_equivalence`` below asserts.

    Strategies:
      ``Baseline``      the plain truncated window of Pang et al. (2025), Fig. 2b
      ``Hard_*``        ASW with a discrete bridge context (Section 4)
      ``Soft_*``        ASW with a tunable soft bridge context (Section 5)
      ``None`` window   robustness-unaware full context (Fig. 2a)
    """

    def __init__(self, model, tokenizer, strategy, context_window,
                 soft_embedding=None, use_cache=True, rand_seed=0, bridge_text=None):
        if not is_strategy(strategy):
            raise ValueError(
                f'unknown strategy {strategy!r}; expected one of {ALL_STRATEGIES}, '
                f'{RAND_PREFIX}<k> or {TEXT_PREFIX}<label>')
        if strategy in SOFT_STRATEGIES and soft_embedding is None:
            raise ValueError(f'strategy {strategy!r} needs a soft bridge context')
        if strategy.startswith(TEXT_PREFIX) and not bridge_text:
            raise ValueError(f'strategy {strategy!r} needs its bridge string passed '
                             f'as bridge_text (--bridge_texts)')

        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.context_window = context_window
        self.use_cache = use_cache and context_window > 0 and strategy != 'Baseline'

        # The hard bridge is tokenized once, up front.  add_special_tokens=False
        # keeps a tokenizer that prepends BOS (Llama does, Qwen does not) from
        # placing one between the prompt and the latest tokens, since the bridge
        # sits in the middle of the sequence.
        self.bridge_ids = None
        # An explicit bridge_text overrides the registry, so sweeping candidate
        # strings does not mean editing HARD_BRIDGES.
        text = bridge_text if bridge_text is not None else HARD_BRIDGES.get(strategy, '')
        if text:
            self.bridge_ids = tokenizer(
                text, return_tensors='pt', add_special_tokens=False
            )['input_ids'].to(model.device)
        elif strategy.startswith(RAND_PREFIX):
            # Drawn once per run: the bridge has to stay fixed for the whole text
            # so that the decoder can rebuild the same window.
            k = int(strategy[len(RAND_PREFIX):])
            generator = torch.Generator().manual_seed(rand_seed)
            self.bridge_ids = torch.randint(
                len(tokenizer.get_vocab()), (1, k), generator=generator).to(model.device)

        self.soft_embedding = None
        if soft_embedding is not None:
            self.soft_embedding = soft_embedding.unsqueeze(0)  # (1, l_bridge, e)

        self._prefix_cache = None
        self._prefix_len = 0
        self._full_cache = None
        self._full_len = 0
        self._cache_enabled = use_cache

    # -- per-sample state ---------------------------------------------------- #

    def reset(self, prompt_ids):
        """Start a new sample.  ``prompt_ids`` is (1, N_p), the anchored prompt."""
        self.prompt_ids = prompt_ids
        self.prompt_len = prompt_ids.shape[1]
        self._prefix_cache = None
        self._prefix_len = 0
        self._full_cache = None
        self._full_len = 0

    # -- internals ----------------------------------------------------------- #

    def _prefix_embeds(self):
        """Embeddings of the invariant ``prompt || bridge`` head of the window."""
        embed = self.model.get_input_embeddings()
        parts = [embed(self.prompt_ids)]
        if self.soft_embedding is not None:
            parts.append(self.soft_embedding.to(parts[0].dtype))
        elif self.bridge_ids is not None:
            parts.append(embed(self.bridge_ids))
        return torch.cat(parts, dim=1)

    def _build_prefix_cache(self):
        from transformers import DynamicCache
        prefix = self._prefix_embeds()
        cache = DynamicCache()
        with torch.no_grad():
            self.model(inputs_embeds=prefix, past_key_values=cache, use_cache=True)
        self._prefix_cache = cache
        self._prefix_len = prefix.shape[1]

    def _logits_cached(self, latest_ids):
        with torch.no_grad():
            out = self.model(input_ids=latest_ids,
                             past_key_values=self._prefix_cache,
                             use_cache=True)
        # Drop the freshly appended keys/values so the cache holds the prefix
        # only and the next step starts from the same state.
        self._prefix_cache.crop(self._prefix_len)
        return out.logits[0, -1, :]

    def _logits_full(self, context):
        """Full-context logits, extending a cache one token at a time.

        Keeps robustness-unaware generation linear in the sequence length.
        """
        if not self._cache_enabled:
            with torch.no_grad():
                return self.model(context).logits[0, -1, :]

        from transformers import DynamicCache
        # Callers extend the context by one token per step; anything else (a new
        # sample, a decoder restarting) means the cache no longer describes it.
        if self._full_cache is None or context.shape[1] <= self._full_len:
            self._full_cache = DynamicCache()
            self._full_len = 0

        with torch.no_grad():
            out = self.model(context[:, self._full_len:],
                             past_key_values=self._full_cache, use_cache=True)
        self._full_len = context.shape[1]
        return out.logits[0, -1, :]

    def _logits_uncached(self, latest_ids):
        embed = self.model.get_input_embeddings()
        inputs = torch.cat([self._prefix_embeds(), embed(latest_ids)], dim=1)
        with torch.no_grad():
            out = self.model(inputs_embeds=inputs)
        return out.logits[0, -1, :]

    # -- public API ---------------------------------------------------------- #

    def logits(self, context):
        """Next-token logits given the full generated context so far.

        ``context`` is (1, N_p + t): the prompt followed by the t tokens emitted
        so far.  The caller never has to know which strategy is active.
        """
        generated = context[:, self.prompt_len:]
        w = self.context_window

        # Robustness-unaware reference (Figure 2a).
        if w <= 0:
            return self._logits_full(context)

        # WinStega-style plain truncation (Figure 2b): the whole window slides,
        # so there is no invariant prefix to cache.
        if self.strategy == 'Baseline':
            with torch.no_grad():
                return self.model(context[:, -w:]).logits[0, -1, :]

        # Fewer than w tokens generated: the ASW window already *is* the full
        # context, so the bridge would only get in the way.
        if generated.shape[1] < w:
            return self._logits_full(context)

        latest_ids = generated[:, -w:]
        if self.use_cache:
            if self._prefix_cache is None:
                self._build_prefix_cache()
            return self._logits_cached(latest_ids)
        return self._logits_uncached(latest_ids)

    def probs(self, context, temperature=1.0):
        return F.softmax(self.logits(context).double() / temperature, dim=0)

    # -- dependency set, used by the robustness analysis --------------------- #

    def dependency(self, position):
        """Indices of *generated* tokens that step ``position`` reads.

        The prompt is excluded on purpose: both parties hold it and an attacker
        who only sees the channel cannot touch it (Section 3).  This is the set
        whose corruption breaks the inference at ``position``.
        """
        w = self.context_window
        if w <= 0:
            return range(0, position)
        return range(max(0, position - w), position)


def verify_cache_equivalence(window, context, atol=1e-4):
    """Assert the cached path reproduces the plain forward pass.

    Called by the smoke test, since the cached and uncached paths must agree for
    extraction to be lossless.
    """
    if not window.use_cache:
        return 0.0
    window.use_cache = False
    reference = window.logits(context).float()
    window.use_cache = True
    window._prefix_cache = None
    cached = window.logits(context).float()
    gap = (reference - cached).abs().max().item()
    if gap > atol:
        raise AssertionError(
            f'prefix KV cache changes the logits by {gap:.3e} (> {atol:.0e}); '
            f'rerun with --no_cache'
        )
    return gap


# --------------------------------------------------------------------------- #
# Bit helpers for the arithmetic coder (Ziegler et al., 2019)
# --------------------------------------------------------------------------- #

def num_same_from_beg(bits1, bits2):
    """Length of the common prefix of two equal-length bit lists."""
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            return i
    return len(bits1)


def bits2int(bits):
    """Little-endian bit sequence -> int, e.g. [0, 1, 1, 1] is 1110b = 14."""
    res = 0
    for i, bit in enumerate(bits):
        res += int(bit) * (2 ** i)
    return res


def int2bits(inp, num_bits):
    """Int -> little-endian bit list of fixed width."""
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(int(inp))
    return [int(strval) for strval in reversed(strlist)]


def build_intervals(probs_temp, cur_interval, topk, precision):
    """Shared front half of one arithmetic-coding step.

    Both the encoder and the decoder have to slice the current interval the same
    way; doing it here keeps them from drifting apart.  ``probs_temp`` must
    already be sorted descending.

    Returns the cumulative interval boundaries, in absolute coordinates.
    """
    cur_int_range = cur_interval[1] - cur_interval[0]
    cur_threshold = 1.0 / cur_int_range

    below = (probs_temp < cur_threshold).nonzero()
    if below.shape[0] == 0:
        k = topk
    else:
        k = min(max(2, below[0].item()), topk)

    probs_temp_int = probs_temp[:k]
    probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
    probs_temp_int = probs_temp_int.round()
    cum_probs = probs_temp_int.cumsum(0)

    # Rounding can push the total over the interval; drop the tail if so.
    overfill_index = (cum_probs > cur_int_range).nonzero()
    if len(overfill_index) > 0:
        cum_probs = cum_probs[:overfill_index[0]]

    # ...and top the last bucket up if rounding left the total short.
    cum_probs += cur_int_range - cum_probs[-1]
    cum_probs += cur_interval[0]
    return cum_probs


def advance_interval(cum_probs, selection, cur_interval, precision):
    """Shared back half of one arithmetic-coding step.

    Returns ``(num_bits_encoded, encoded_bits, new_interval)``.  The encoder uses
    the bit count to walk through the message; the decoder takes ``encoded_bits``
    as the bits it just recovered.  They are the same bits, because the message
    index lies between the two interval bounds and so shares their common prefix.
    """
    new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
    new_int_top = cum_probs[selection]

    bottom_inc = list(reversed(int2bits(new_int_bottom, precision)))
    top_inc = list(reversed(int2bits(new_int_top - 1, precision)))  # top is exclusive

    num_bits_encoded = num_same_from_beg(bottom_inc, top_inc)

    new_bottom = bottom_inc[num_bits_encoded:] + [0] * num_bits_encoded
    new_top = top_inc[num_bits_encoded:] + [1] * num_bits_encoded

    new_interval = [bits2int(reversed(new_bottom)),
                    bits2int(reversed(new_top)) + 1]  # +1: upper bound exclusive
    return num_bits_encoded, top_inc[:num_bits_encoded], new_interval


# Columns holding digit strings that must stay strings: the secret message is
# thousands of bits wide, well past what a numeric dtype can represent.
STRING_COLUMNS = {'message': str, 'Context_token': str, 'Text_token': str}


def read_stega_tsv(path, **kwargs):
    """Read an experiment TSV, keeping the bit strings as strings."""
    import pandas as pd
    return pd.read_csv(path, sep='\t', dtype=STRING_COLUMNS, **kwargs)


def ensure_dir(path):
    """Create the directory a file is about to be written into."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path
