import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import kl_div, log_softmax, softmax
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
# import json
from accelerate import Accelerator
import os
import gc
import argparse
import pandas as pd
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def print_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"{prefix} >>> Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length= 256):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]

        return {'question': question, 'answer': answer}
    
def collate_fn(batch):
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]

    q_encodings = tokenizer(questions, return_tensors='pt', truncation=True, padding=True, max_length = int(args.q_token_max))
    q_input_ids = q_encodings['input_ids']
    q_attention_mask = q_encodings['attention_mask']

    a_encodings = tokenizer(answers  , return_tensors='pt', truncation=True, padding=True, max_length = args.a_token_max)
    a_input_ids = a_encodings['input_ids']
    a_attention_mask = a_encodings['attention_mask']

    return {
        'q_input_ids': q_input_ids,
        'q_attention_mask': q_attention_mask,
        'a_input_ids': a_input_ids,
        'a_attention_mask': a_attention_mask
    }

# 初始化

if main := '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default = "Qwen/Qwen2.5-7B-Instruct", type = str, required=False)
    parser.add_argument('--a_token_max', default = 512, type = int, required = False)
    parser.add_argument('--q_token_max', default = 64, type = int, required = False)
    parser.add_argument('--window_size', default = 10, type = int, required = False)
    parser.add_argument('--initialization', default = 0, type = int, required = False)
    parser.add_argument('--soft_prompt_length', default = 8, type = int, required = False)
    parser.add_argument('--epochs', default = 10, type = int, required = False)
    parser.add_argument('--batch_size', default = 1, type = int, required = False)
    parser.add_argument('--KL_reverse', default = 0, type = int, required = False)
    parser.add_argument('--step', default = 50, type = int, required = False) 
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    # 使用 instruct 模型（替换成你自己的）
    model_name = args.language_model  # 可以替换成 llama-7b-instruct
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad token 为 eos token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = "auto").to(torch.float32)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    embedding_size = model.config.hidden_size

    if args.initialization == 1:
        hard_prompt = '[CONTEXT TRUNCATED]\n' * 100
        hard_prompt_ids = tokenizer(hard_prompt, add_special_tokens=False, return_tensors='pt', max_length = args.soft_prompt_length, truncation = True)['input_ids']
        hard_prompt_embeddings = model.model.embed_tokens(hard_prompt_ids).squeeze(0)  # shape: (seq_len, embedding_size)
        soft_prompt = torch.nn.Parameter(hard_prompt_embeddings)
    elif args.initialization == 0:
        soft_prompt = torch.nn.Parameter(torch.randn(args.soft_prompt_length, embedding_size))
    
    optimizer = torch.optim.AdamW([soft_prompt], lr=1e-3)
    soft_length = soft_prompt.shape[0]

    df = pd.read_csv('instinwild_en.tsv', sep='\t', encoding='utf-8')
    df_train = df[df['part'] == 'train']
    df_validation = df[df['part'] == 'validation']
    train_num = len(df_train)
    validation_num = len(df_validation)

    qa_pairs = []

    for subset in [df_train, df_validation]:

        for i in range(len(subset)):
            question = list(subset['question'])[i]
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True
            )
            answer = list(subset['answer'])[i]
            qa_pairs.append((prompt , answer))



    train_dataset = QADataset(qa_pairs[:train_num], tokenizer)
    val_dataset = QADataset(qa_pairs[train_num: train_num + validation_num], tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model, dataloader, soft_prompt, optimizer = accelerator.prepare(model, train_dataloader, soft_prompt, optimizer)
    # print(f"Using device: {accelerator.device}")
    # print(f"Number of processes: {accelerator.num_processes}")
    # epochs = 5
    # context_window = 10  # 你定义的限制窗口

    initial_index = random.randint(0, args.step - 1)  # 随机起始位置
    for epoch in range(args.epochs):
        total_loss = 0
        total_step = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}") if accelerator.is_main_process else dataloader

        for batch in progress:
            # kl_loss = 0.0
            q_ids = batch['q_input_ids'].to(accelerator.device)
            q_attention_mask = batch['q_attention_mask'].to(accelerator.device)
            a_ids = batch['a_input_ids'].to(accelerator.device)
            a_attention_mask = batch['a_attention_mask'].to(accelerator.device)

            batch_size = q_ids.shape[0]

            # =============== Teacher ==================
            full_input_ids = torch.cat([q_ids, a_ids], dim=1) # shape: (batch_size, seq_len)
            full_attention_mask = torch.cat([q_attention_mask, a_attention_mask], dim=1)  # shape: (batch_size, seq_len)
            # labels = full_input_ids.clone()
            # labels[labels == tokenizer.pad_token_id] = -100  # 忽略 pad token

            with torch.no_grad():
                teacher_outputs = model(input_ids=full_input_ids, attention_mask = full_attention_mask)
                teacher_logits = teacher_outputs.logits.clone().detach()  # shape: (batch_size, seq_len, vocab_size)
            # print_memory("After Teacher Forward")

            # =============== Student ==================
            q_embeddings = model.model.embed_tokens(q_ids)
            prompt_embeddings = soft_prompt.unsqueeze(0).repeat(batch_size, 1, 1).to(accelerator.device)

            # ⚙️ 
            for start_index in range(initial_index, a_ids.shape[1] - args.window_size, args.step):
                if a_attention_mask[:, start_index].sum() < 0.8 * args.batch_size:
                    continue
                total_step += 1
                end_index = start_index + args.window_size

                window_embeddings = model.model.embed_tokens(a_ids[:, start_index: end_index])
                soft_promt_attention_mask = torch.ones((batch_size, soft_length), device=accelerator.device)
                student_attention_mask = torch.cat([q_attention_mask, soft_promt_attention_mask, a_attention_mask[:, start_index: end_index]], dim=1)

                student_embeddings = torch.cat([q_embeddings, prompt_embeddings, window_embeddings], dim=1)
                student_outputs = model(inputs_embeds = student_embeddings, attention_mask = student_attention_mask)
                student_logits = student_outputs.logits  # shape: (batch_size, student_seq_len, vocab_size)
                # print_memory("After Student Forward")
                # =============== KL Loss 对齐 ==================

                student_index = q_ids.shape[1] + soft_length + args.window_size
                teacher_index = q_ids.shape[1] + end_index

                student_logits_for_answer = student_logits[:, -1: , :]
                teacher_logits_for_answer = teacher_logits[:, teacher_index -1: teacher_index, :]



                if args.KL_reverse == 0:
                    student_log_probs = log_softmax(student_logits_for_answer, dim=-1)
                    teacher_probs = softmax(teacher_logits_for_answer, dim=-1)
                    kl_loss = kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                else:
                    teacher_log_probs = log_softmax(teacher_logits_for_answer, dim=-1)
                    student_probs = softmax(student_logits_for_answer, dim=-1)
                    kl_loss = kl_div(teacher_log_probs, student_probs, reduction='batchmean')

                optimizer.zero_grad()
                accelerator.backward(kl_loss)
                optimizer.step()
                # print_memory("After Backward + Step")
                total_loss += kl_loss.item()
                initial_index = (initial_index + 1) % args.step  # 更新起始位置

                if accelerator.is_main_process:
                    progress.set_postfix({"KL Loss": total_loss / total_step})

            # del teacher_outputs, student_outputs, teacher_logits, student_logits
            gc.collect()
            torch.cuda.empty_cache()


                # ---------- 验证 ----------
        if accelerator.is_main_process:
            print("Running validation...")

        val_total_loss = 0
        val_total_step = 0

        model.eval()  # important
        with torch.no_grad():
            for batch in val_dataloader:
                q_ids = batch['q_input_ids'].to(accelerator.device)
                q_attention_mask = batch['q_attention_mask'].to(accelerator.device)
                a_ids = batch['a_input_ids'].to(accelerator.device)
                a_attention_mask = batch['a_attention_mask'].to(accelerator.device)

                batch_size = q_ids.shape[0]

                full_input_ids = torch.cat([q_ids, a_ids], dim=1)
                full_attention_mask = torch.cat([q_attention_mask, a_attention_mask], dim=1)

                teacher_outputs = model(input_ids=full_input_ids, attention_mask=full_attention_mask)
                teacher_logits = teacher_outputs.logits

                q_embeddings = model.model.embed_tokens(q_ids)
                prompt_embeddings = soft_prompt.unsqueeze(0).repeat(batch_size, 1, 1).to(accelerator.device)

                for start_index in range(initial_index, a_ids.shape[1] - args.window_size, args.step):
                    if a_attention_mask[:, start_index].sum() < 0.8 * args.batch_size:
                        continue
                    val_total_step += 1
                    end_index = start_index + args.window_size

                    window_embeddings = model.model.embed_tokens(a_ids[:, start_index:end_index])
                    soft_prompt_attention_mask = torch.ones((batch_size, soft_length), device=accelerator.device)
                    student_attention_mask = torch.cat([
                        q_attention_mask,
                        soft_prompt_attention_mask,
                        a_attention_mask[:, start_index:end_index]
                    ], dim=1)

                    student_embeddings = torch.cat([q_embeddings, prompt_embeddings, window_embeddings], dim=1)
                    student_outputs = model(inputs_embeds=student_embeddings, attention_mask=student_attention_mask)
                    student_logits = student_outputs.logits

                    student_logits_for_answer = student_logits[:, -1:, :]
                    teacher_index = q_ids.shape[1] + end_index
                    teacher_logits_for_answer = teacher_logits[:, teacher_index - 1:teacher_index, :]

                    if args.KL_reverse == 0:
                        student_log_probs = log_softmax(student_logits_for_answer, dim=-1)
                        teacher_probs = softmax(teacher_logits_for_answer, dim=-1)
                        kl_loss = kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                    else:
                        teacher_log_probs = log_softmax(teacher_logits_for_answer, dim=-1)
                        student_probs = softmax(student_logits_for_answer, dim=-1)
                        kl_loss = kl_div(teacher_log_probs, student_probs, reduction='batchmean')
                    val_total_loss += kl_loss.item()
                    initial_index = (initial_index + 1) % args.step  # 更新起始位置

        val_avg_loss = val_total_loss / val_total_step
        model.train()  # back to train mode
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} Validation KL Loss: {val_avg_loss:.4f}")
            # 保存 soft prompt
            torch.save(soft_prompt.detach().cpu(), f'1.soft_prompt/soft_length{soft_length}_epoch{epoch+1}_reverse{args.KL_reverse}_{model_name.split('/')[-1]}_{val_avg_loss:.4f}.pt')