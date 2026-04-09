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

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def generate(context):
    StartTime = time.time()

    total_num_for_stats = 0
    total_entropy_ptau = 0

    with torch.no_grad():
        while total_num_for_stats < args.token_max :
            
            output_window = model(context)
            # past = limit_past(past)
            logits = output_window.logits[0, -1, :]
            logits = logits.double()
            logits_temp = logits / args.temp
            probs_temp = F.softmax(logits_temp, dim=0)
            next_token_id = torch.multinomial(probs_temp, num_samples=1)
            
            this_entropy = -torch.sum(probs_temp * torch.log2(probs_temp + 1e-10)).item()
            total_entropy_ptau += this_entropy
            total_num_for_stats += 1

            context = torch.cat((context, next_token_id.unsqueeze(0)), dim=1)
            
            if next_token_id == tokenizer.eos_token_id:
                break

            
    avg_Hq = total_entropy_ptau/total_num_for_stats
    Duration = time.time() - StartTime

    generation_tokens = context[:, -total_num_for_stats:][0].tolist()
    generation_text = tokenizer.decode(context[:, -total_num_for_stats:].squeeze(0), skip_special_tokens=True)

    return total_num_for_stats, avg_Hq, Duration, generation_text, generation_tokens



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'instinwild_en', type = str, required=False) # 'instinwild_en' or 'databricks-dolly' or 'supernatural'
    parser.add_argument('--language_model', default = "Qwen/Qwen2.5-7B-Instruct", type = str, required=False)
    parser.add_argument('--token_max', default = 512, type = int, required = False)
    parser.add_argument('--top_k', default = -1, type = int, required = False)
    parser.add_argument('--temp', default = 1.0, type = float, required = False)
    parser.add_argument('--context_window', default = -1, type = int, required = False) # 10-50
    args = parser.parse_args()
    print(args)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.language_model
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map='auto').to(torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.resize_token_embeddings(len(tokenizer.vocab))
    model.eval()

    


    header = ['Idx', 'Token_num', 'Entropy', 'Time', 'Context', 'Text', 'Context_token', 'Text_token']
    file_name = '3.Stega_data/Normal_' + model_name.rsplit('/', 1)[-1] + '_' + args.dataset + '.tsv'

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


        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)['input_ids']
        text_tokens = model_inputs[0].tolist()
        Token_num, Entropy,  Time, generation_text, generation_tokens =  generate(model_inputs)
        
        with open(file_name, 'a+', newline = '', encoding = 'utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([idx, Token_num, Entropy, Time, text, generation_text, text_tokens, generation_tokens])
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