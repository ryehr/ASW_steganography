import torch
import torch.nn.functional as F
import argparse
import numpy as np
import bitarray
import sys
import re
import math
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import random
import time
import os
import csv
from datasets import load_dataset
from itertools import islice
import pandas as pd
import gc
# import json
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from utils import limit_past, kl, entropy, bits2int, int2bits, is_sent_finish, num_same_from_beg

class SoftPromptEmbedding(torch.nn.Module):
    def __init__(self, init_tensor):
        super().__init__()
        self.soft_prompt = torch.nn.Parameter(init_tensor)

    def forward(self, batch_size):
        return self.soft_prompt.unsqueeze(0).repeat(batch_size, 1, 1)
    
def get_soft_prompt_embedding(path): # LoRA model
    # 1. 构造和保存时相同的 init_tensor（只用于初始化结构，之后会被覆盖）
    # 这里 shape 要和你训练时一样，比如 [soft_length, hidden_size]
    init_tensor = torch.zeros(args.default_soft_length, model.config.hidden_size)  # 假设 soft_length=8，hidden_size=4096

    # 2. 实例化模型结构
    soft_embedding = SoftPromptEmbedding(init_tensor)

    # 3. 加载参数
    state_dict = torch.load(path, map_location='cpu')
    soft_embedding.load_state_dict(state_dict)

    # 4. 移动到目标设备
    soft_embedding = soft_embedding.soft_prompt.to(model.device)
    return soft_embedding

def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break

    return i

def limit_past(past):
    past = list(past)
    for i in range(len(past)):
        past[i] = past[i][:, :, :, -1022:]
    return past

def kl(q, logq, logp):
    res = q*(logq-logp)
    res[q==0] = 0
    # res[torch.isnan(res)] = 0
    return res.sum() # in bits

def calculate_entropy(q, logq):
    res = q*logq/torch.log(torch.tensor(2.0))
    res[q==0] = 0
    return -res.sum().item() # in bits

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = torch.tensor(0, dtype=torch.float64, device = model.device)
    for i, bit in enumerate(bits):
        res += torch.tensor(int(bit)*(2**i), dtype=torch.float64, device= model.device)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(int(inp.item()))
    return [int(strval) for strval in reversed(strlist)]

def is_sent_finish(token_idx, enc):
    token = enc.decoder[token_idx]
    return '.' in token or '!' in token or '?' in token


def get_transition_tokens(key):

    if key == 'Hard_1':
        transition = '[CONTEXT TRUNCATED]\n'
    elif key == 'Hard_2':
        transition = '...\n'
    transition_tokens = tokenizer([transition], return_tensors="pt").to(model.device)['input_ids']
    return transition_tokens

def encode_arithmetic(model, message, context, topk):
    StartTime = time.time()
    start_index = context.shape[1]
    # context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    precision = args.precision
    max_val = 2**precision
    cur_interval = torch.tensor([0, max_val], device = model.device, dtype = torch.float64) # bottom inclusive, top exclusive

    # prev = context
    # output = context
    # past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    # total_kl = torch.tensor(0, device = model.device, dtype = torch.float64) # in bits
    total_entropy_ptau = 0
    total_num_sents = 0
    prob_list = []
    with torch.no_grad():
        i = 0
        sent_finish = False
        while total_num_for_stats < args.token_max :
            ###############################################
            # Reference to the context window
            # output_original = model(context)
            # logits_original = output_original.logits.double()[0, -1, :]
            # probs_original = F.softmax(logits_original, dim=0)
            # log_probs_original = torch.log(probs_original)
            #############################################
            if args.context_window <= 0:
                output_window = model(context)
            elif args.strategy == 'Baseline': 
                output_window = model(context[:, -args.context_window:])
            elif total_num_for_stats < args.context_window:
                output_window = model(context)
            elif total_num_for_stats >= args.context_window:

                model_inputs = context.clone()
                temp_inputs = model_inputs[:, :start_index]
                generated_inputs = model_inputs[:, start_index:]
                    
                if args.strategy == 'Hard_0':
                    temp_inputs = torch.cat((temp_inputs, generated_inputs[:, -args.context_window:]), dim=1)
                    output_window = model(temp_inputs)
                    
        
                elif args.strategy in ['Hard_1', 'Hard_2']:
                    transition_tokens = get_transition_tokens(args.strategy)
                    temp_inputs = torch.cat((temp_inputs, transition_tokens, generated_inputs[:, -args.context_window:]),dim = 1)
                    output_window = model(temp_inputs)
                elif 'Soft' in args.strategy:
                    if args.lora == 0:
                        q_embedding = model.model.embed_tokens(temp_inputs)
                        window_embedding = model.model.embed_tokens(generated_inputs[:, -args.context_window:])
                        pass
                    elif args.lora == 1:
                        embedding_layer = model.get_input_embeddings()
                        q_embedding = embedding_layer(temp_inputs)
                        window_embedding = embedding_layer(generated_inputs[:, -args.context_window:])
                        soft_embedding = lora_soft_embedding
                    
                    if args.strategy == 'Soft_0':
                        soft_embedding = torch.randn(args.default_soft_length, model.config.hidden_size).to(model.device)
                    
                    elif args.strategy == 'Soft_forward' and args.lora == 0:
                        if args.default_soft_length == 8:
                            soft_embedding = torch.load('1.soft_prompt/soft_length8_epoch9_reverse0_Qwen2.5-7B-Instruct_0.6792.pt', weights_only = True).to(model.device)

                        elif args.default_soft_length == 32:
                            soft_embedding = torch.load('1.soft_prompt/soft_length32_epoch9_reverse0_Qwen2.5-7B-Instruct_0.6336.pt', weights_only = True).to(model.device)
                    
                    elif args.strategy == 'Soft_reverse' and args.lora == 0:
                        if args.default_soft_length == 8:
                            soft_embedding = torch.load('1.soft_prompt/soft_length8_epoch9_reverse1_Qwen2.5-7B-Instruct_1.1110.pt', weights_only = True).to(model.device)

                        elif args.default_soft_length == 32:
                            soft_embedding = torch.load('1.soft_prompt/soft_length32_epoch8_reverse1_Qwen2.5-7B-Instruct_1.1037.pt', weights_only = True).to(model.device)
                    
                    soft_embedding = soft_embedding.unsqueeze(0).expand(temp_inputs.shape[0], -1, -1)
                    temp_embedding = torch.cat((q_embedding, soft_embedding, window_embedding), dim=1)
                    output_window = model(inputs_embeds = temp_embedding)

                
            # past = limit_past(past)
            logits = output_window.logits
            # logits[0, -1, -1] = -1e20 # endoftext token can't happen
            # logits[0, -1, 628] = -1e20 # 2 newlines token can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / args.temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)
            # print(probs_temp[0:5])
            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                # sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1]-cur_interval[0]
                cur_threshold = 1/cur_int_range
                # print('cur_int_range:', cur_int_range)
                if (probs_temp < cur_threshold).nonzero().shape[0] == 0:
                    k = topk
                else:
                    k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k] # Cutoff all but top k

                # Rescale to correct range
                probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round()
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range-cum_probs[-1] # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Get selected index based on binary fraction from message bits
                message_bits = message[i:i+precision]
                if i+precision > len(message):
                    message_bits = message_bits + '0'*(i+precision-len(message))
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive
                # print('new_int_bottom:', new_int_bottom_bits_inc)
                # print('new_int_top:', new_int_top_bits_inc)
                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1 # +1 here because upper bound is exclusive

                # Gather statistics
                # prob_list.append(probs_temp[selection].item())
                # probs_final = probs_final
                # q = probs_final.double()/probs_final.sum()
                # q = F.pad(q, (0, len(probs_temp)-len(q)))
                
                # q = q/q.sum()
                # logq = q.log()
                # this_kl = kl(q, logq, log_probs[:len(q)])
                # if len(q) < len(log_probs_original):
                    # q = F.pad(q, (0, len(log_probs_original)-len(q)))
                # total_kl += F.kl_div(log_probs_original[indices], q, reduction='sum').item()
                # print(kl(q, logq, log_probs[:len(q)]))
                # print()
                # this_entropy = calculate_entropy(probs_temp, log_probs_temp)
                this_entropy = -torch.sum(probs_temp * torch.log2(probs_temp + 1e-10)).item()
                total_entropy_ptau += this_entropy
                total_num_for_stats += 1
                pass
            
            # Update history with new token
            prev = indices[selection].view(1)

            context = torch.cat((context, prev.unsqueeze(0)), dim = -1)
            total_num += 1
            if prev.item() == tokenizer.eos_token_id:
                break
            #print(enc.decode(prev.tolist()), message_bits[:num_bits_encoded])
            
            # For text->bits->text
            # partial = enc.decode(output[len(context):].tolist())
            # if '<eos>' in partial:
            #     break
            
    avg_NLL = -total_log_probs/total_num_for_stats
    # avg_KL = (total_kl/total_num_for_stats).item()
    avg_Hq = total_entropy_ptau/total_num_for_stats
    bpt = i/total_num_for_stats
    Utilization = bpt/avg_Hq
    Duration = time.time() - StartTime
    # print('avg_KL:', avg_KL)
    # perplexity = np.exp(-np.mean(np.log(prob_list)))
    stego_tokens = context[:, -total_num_for_stats:][0].tolist()
    Stegotext = tokenizer.decode(context[:, -total_num_for_stats:].squeeze(0), skip_special_tokens=True)
    # print(Stegotext)
    # with open(file_name, 'a+', newline = '', encoding = 'utf-8') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerow([idx, avg_KL, total_num_for_stats, bpt, avg_Hq, Utilization, Duration, perplexity, context_text, Stegotext, message[:i]])
    return total_num_for_stats, bpt, avg_Hq, Utilization, Duration, Stegotext, stego_tokens, message[:i]
    # return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default = "Qwen/Qwen2.5-7B-Instruct", type = str, required=False)
    parser.add_argument('--dataset', default = 'instinwild_en', type = str, required=False) # 'instinwild_en' or 'databricks-dolly' or 'supernatural'
    # parser.add_argument('--bit_length', default = 128, type=int, required=False)
    parser.add_argument('--token_max', default = 512, type = int, required = False)
    parser.add_argument('--top_k', default = -1, type = int, required = False)
    parser.add_argument('--temp', default = 1.0, type = float, required = False)
    parser.add_argument('--precision', default = 32, type = int, required = False) #default: 26, max:52
    parser.add_argument('--context_window', default = 10, type = int, required = False) # 10-50
    # parser.add_argument('--generation_size', default = 500, type = int, required = False)
    # parser.add_argument('--index_start', default = 0, type = int, required = False)
    # parser.add_argument('--index_end', default = 200, type = int, required = False)
    parser.add_argument('--strategy', default = 'Soft_forward', type = str, required = False) # 'Baseline', 'Hard_0', 'Hard_1', 'Hard_2', 'Soft_0', 'Soft_forward', 'Soft_reverse'
    parser.add_argument('--default_soft_length', default = 8, type = int, required = False) # for Soft_0
    parser.add_argument('--lora', default = 0, type = int, required = False) # 0: no lora, 1: lora
    args = parser.parse_args()
    print(args)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.language_model
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map='auto').to(torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.lora == 1 and args.context_window > 0:
        if args.default_soft_length == 8:
            if args.strategy == 'Soft_forward':
                lora_model_name = '1.lora_checkpoints/epoch_10_soft_length_8_reverse_0_Qwen2.5-7B-Instruct_0.5970'
                lora_soft_embedding = get_soft_prompt_embedding('1.lora_checkpoints/lora_soft_length8_epoch10_reverse0_Qwen2.5-7B-Instruct_0.5970.pt')
                # print('Loading LoRA model:', lora_model_name)
            elif args.strategy == 'Soft_reverse':
                lora_model_name = '1.lora_checkpoints/epoch_7_soft_length_8_reverse_1_Qwen2.5-7B-Instruct_0.9767'
                lora_soft_embedding = get_soft_prompt_embedding('1.lora_checkpoints/lora_soft_length8_epoch7_reverse1_Qwen2.5-7B-Instruct_0.9767.pt')
        elif args.default_soft_length == 0:
            lora_soft_embedding = torch.randn(args.default_soft_length, model.config.hidden_size).to(model.device)
            if args.strategy == 'Soft_forward':
                lora_model_name = '1.lora_checkpoints/epoch_10_soft_length_0_reverse_0_Qwen2.5-7B-Instruct_0.6012'
            elif args.strategy == 'Soft_reverse':
                lora_model_name = '1.lora_checkpoints/epoch_8_soft_length_0_reverse_1_Qwen2.5-7B-Instruct_0.9975'
        model = PeftModel.from_pretrained(model, lora_model_name).to(model.device)
        

    model.resize_token_embeddings(len(tokenizer.vocab))
    model.eval()

    


    header = ['Idx', 'Token_num', 'BPT', 'Entropy', 'Utilization', 'Time', 'Context', 'Text', 'Context_token', 'Text_token', 'message']
    if args.context_window > 0:
        if 'Soft' not in args.strategy:
            file_name = '3.Stega_data/AC_' + model_name.rsplit('/', 1)[-1] + '_window_{}'.format(args.context_window) + '_strategy_{}_lora_{}_{}.tsv'.format(args.strategy, args.lora, args.dataset)
        else:
            file_name = '3.Stega_data/AC_' + model_name.rsplit('/', 1)[-1] + '_window_{}'.format(args.context_window) + '_strategy_{}_lora_{}_softlength_{}_{}.tsv'.format(args.strategy, args.lora, args.default_soft_length, args.dataset)
    else:
        file_name = '3.Stega_data/AC_' + model_name.rsplit('/', 1)[-1] + '_full_{}.tsv'.format(args.dataset)

    if os.path.exists(file_name) == False:
        with open(file_name, 'w', newline = '', encoding = 'utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(header)

    df = pd.read_csv('{}.tsv'.format(args.dataset), sep='\t', encoding='utf-8')
    df_test = df[df['part'] == 'test']

    for i in range(len(df_test)):
        prompt = list(df_test['question'])[i]
        idx = list(df_test['new_id'])[i]
        print(prompt)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )
        secret_bits = ''.join(str(random.randint(0, 1)) for _ in range(100000))


        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)['input_ids']
        text_tokens = model_inputs[0].tolist()
        Token_num, BPT, Entropy, Utilization, Time, stegotext, stegotext_tokens, secret_bits_real =  encode_arithmetic(model, secret_bits, model_inputs, topk = len(tokenizer.vocab) if args.top_k < 0 else args.top_k)
        
        with open(file_name, 'a+', newline = '', encoding = 'utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([idx, Token_num, BPT, Entropy, Utilization, Time, text, stegotext, text_tokens, stegotext_tokens, secret_bits_real])
        gc.collect() 
        torch.cuda.empty_cache()
        # KL_list.append(KL)
        # bpt_list.append(bpt)
        # entropy_list.append(entropy)
        # utilization_list.append(utilization)
        # time_list.append(duration)
        # print('KL:', sum(KL_list)/len(KL_list))
        # print('bpt:', sum(bpt_list)/len(bpt_list))
        # print('entropy:', sum(entropy_list)/len(entropy_list))
        # print('utilization:', sum(utilization_list)/len(utilization_list))
        # print('time:', sum(time_list)/len(time_list))