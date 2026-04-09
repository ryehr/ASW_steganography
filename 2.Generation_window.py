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
import json
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def reconstruct_transition(prompt, latest_words):
#     model.generation_config.temperature = None
#     model.generation_config.top_p = None
#     model.generation_config.top_k = None

#     auxiliary_message = [
#         {"role": "system", "content": "You are given a question and the ending of an answer. Your task is to seamlessly fill in the missing middle content to create a coherent and natural question-answer pair. Ensure that the transition is smooth and the completed sentence makes logical sense."},
#         {"role": "user", "content": 'Question: ' + prompt + '\n' +'The end of an answer: ' + latest_words + '\n' + 'Your task: Fill in the missing middle part to make it coherent. Output only the middle part. '}]
#     form = tokenizer.apply_chat_template(
#         auxiliary_message,
#         tokenize = False,
#         add_generation_prompt = True)
#     inputs = tokenizer([form], return_tensors="pt").to(model.device)
#     input_length = inputs['input_ids'].shape[1]
#     generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)[:, input_length:]
#     # print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
#     return generated_ids

def get_transition_tokens(key):

    if key == 'Hard_1':
        transition = '[CONTEXT TRUNCATED]\n'
    elif key == 'Hard_2':
        transition = '...\n'
    transition_tokens = tokenizer([transition], return_tensors="pt").to(model.device)['input_ids']
    return transition_tokens

def get_prob(model_inputs, start_index, key):
    with torch.no_grad():

        temp_inputs = model_inputs.clone()[:, :start_index]
        if key == 'Baseline':
            temp_inputs = model_inputs[:, -args.context_window:]
            output_window = model_lora(temp_inputs)

        elif key == 'Hard_0':
            temp_inputs = torch.cat((temp_inputs, model_inputs[:, -args.context_window:]), dim=1)
            output_window = model_lora(temp_inputs)
 
        elif key in ['Hard_1', 'Hard_2']:
            transition_tokens = get_transition_tokens(key)
            temp_inputs = torch.cat((temp_inputs, transition_tokens, model_inputs[:, -args.context_window:]),dim = 1)
            output_window = model_lora(temp_inputs)
        
        elif 'Soft' in key:
            if args.lora == 0:
                q_embedding = model_lora.model.embed_tokens(temp_inputs)
                window_embedding = model_lora.model.embed_tokens(model_inputs[:, -args.context_window:])
            elif args.lora == 1:
                embedding_layer = model_lora.get_input_embeddings()
                q_embedding = embedding_layer(temp_inputs)
                window_embedding = embedding_layer(model_inputs[:, -args.context_window:])
            if key == 'Soft_0':
                soft_embedding = torch.randn(6, model_lora.config.hidden_size).to(model_lora.device)
            elif key == 'Soft_1':
                soft_embedding = torch.load('1.soft_prompt/soft_prompt_length6_epoch5_Qwen2.5-7B-Instruct.pt', weights_only = True).to(model_lora.device)
            elif key == 'Soft_2':
                soft_embedding = torch.load('1.soft_prompt/soft_prompt_length6_epoch10_Qwen2.5-7B-Instruct.pt', weights_only = True).to(model_lora.device)
            soft_embedding = soft_embedding.unsqueeze(0).expand(temp_inputs.shape[0], -1, -1)
            temp_embedding = torch.cat((q_embedding, soft_embedding, window_embedding), dim=1)
            output_window = model_lora(inputs_embeds = temp_embedding)
        pass
    logits_window = output_window.logits[0, -1, :]
    probs_window = F.softmax(logits_window, dim=0)
    return probs_window


def generate(model_inputs):
    
    # assert transition_tokens is not None
    strategies = {'Baseline': 0, 'Hard_0': 0, 'Hard_1': 0, 'Hard_2': 0, 'Soft_0': 0, 'Soft_1': 0, 'Soft_2': 0}
    Time = time.time()
    start_index = model_inputs.shape[1]
    for i in range(args.token_max):
        with torch.no_grad():
            output = model(model_inputs)

        
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=0)
        log_probs = torch.log(probs)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # print(f"Time taken for step {i+1}: {time.time() - Time:.2f} seconds")

        for key in strategies.keys():            
            probs_window = get_prob(model_inputs, start_index, key)       
            strategies[key] += F.kl_div(log_probs, probs_window, reduction='sum').item()

            pass
        if (i+1) % 50 == 0:
            print(f"Step {i+1}, KL Divergence: {strategies}")
        model_inputs = torch.cat([model_inputs, next_token_id.unsqueeze(0)], dim=1)
        if next_token_id == tokenizer.eos_token_id:
            break
    for key in strategies.keys():
            strategies[key] /= (i + 1)
    
    return tokenizer.decode(model_inputs[0][-i-1:], skip_special_tokens=True), strategies, i+1, time.time() - Time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default = "Qwen/Qwen2.5-7B-Instruct", type = str, required=False)
    parser.add_argument('--token_max', default = 512, type = int, required = False)
    # parser.add_argument('--optimization', default = 0, type = int, required = False) # 1: [CONTEXT TRUNCATED], 0: [PAD], -1: insert empty, 10: auxiliary model
    parser.add_argument('--lora', default = 0, type = int, required = False)
    # parser.add_argument('--index_start', default = 0, type = int, required = False)
    # parser.add_argument('--index_end', default = 200, type = int, required = False)
    parser.add_argument('--context_window', default = 10, type = int, required = False)

    # parser.add_argument('--context_window', default = 10, type = int, required = False)
    args = parser.parse_args()
    print(args)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    model_name = args.language_model
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map='auto').to(torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer.vocab))
    model.eval()

    if args.lora == 1:
        model_lora = PeftModel.from_pretrained(model, '1.lora_checkpoints/epoch_5')
        model_lora.resize_token_embeddings(len(tokenizer.vocab))
        model_lora.eval()
    elif args.lora == 0:
        model_lora = model



    output_file = '2.data/{}_window_{}_lora_{}.tsv'.format(model_name.rsplit('/', 1)[-1] ,args.context_window, args.lora)

    header = ['Idx', 'Baseline', 'Hard_0', 'Hard_1', 'Hard_2', 'Soft_0', 'Soft_1', 'Soft_2', 'Token_num', 'Time', 'Context', 'stegotext']
    if os.path.exists(output_file) == False:
        with open(output_file, 'w', newline = '', encoding = 'utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(header)

    with open('instinwild_en.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    


    for i in range(args.index_start, args.index_end):
        prompt = data[i]['instruction']
        idx = data[i]['id']
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
        stegotext, kl_avg, token_num, Time = generate(model_inputs)
        with open(output_file, 'a+', newline = '', encoding = 'utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([idx, kl_avg['Baseline'], kl_avg['Hard_0'], kl_avg['Hard_1'], kl_avg['Hard_2'], kl_avg['Soft_0'], kl_avg['Soft_1'], kl_avg['Soft_2'], token_num, Time, text, stegotext])
        
