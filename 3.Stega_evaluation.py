import pandas as pd
import os
import evaluate
import numpy as np
from openai import OpenAI
import logging
import argparse
from transformers import AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_bleu_score(candidate_data, ref_data):
    bleu = evaluate.load("bleu")
    score = bleu.compute(predictions=candidate_data, references=ref_data, max_order = 2)['bleu']
    logging.warning('BLEU: {}'.format(score))
    return 

def calculate_rouge_score(candidate_data, ref_data):

    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=candidate_data, references=ref_data)
    score = result['rougeL']

    # print('ROUGE:',score)
    logging.warning('ROUGE: {}'.format(score))
    return

def calculate_BERTScore(candidate_data, ref_data):

    bertscore = evaluate.load("bertscore")
    result = bertscore.compute(predictions=candidate_data, references=ref_data, model_type="distilbert-base-uncased")
    score = sum(result['f1'])/len(result['f1'])

    # print('BERTScore:',score)
    logging.warning('BERTScore: {}'.format(score))
    return

def calculate_perplexity(candidate_data):
    perplexity = evaluate.load("perplexity", module_type="metric")
    truncated_texts = []
    for text in candidate_data:
        tokens = tokenizer.encode(text, truncation=True, max_length=512)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        truncated_texts.append(truncated_text)
    score = perplexity.compute(predictions=truncated_texts, model_id="gpt2", batch_size = 2)['mean_perplexity']
    logging.warning('PPL: {}'.format(score))
    return score

def calculate_embedding_similarity(candidate_data, ref_data):
    with open('OPENAI_API_KEY.txt', 'r', encoding='utf-8') as f:
        os.environ['OPENAI_API_KEY'] = f.read()
    client = OpenAI()
    score = 0.0
    for i in range(len(candidate_data)):
        result_candi = client.embeddings.create(model = "text-embedding-ada-002",input = candidate_data[i],encoding_format="float").data[0].embedding
        result_ref = client.embeddings.create(model = "text-embedding-ada-002",input = ref_data[i],encoding_format="float").data[0].embedding
        score += cosine_similarity(result_candi, result_ref)
        # print(score/(i+1))
        

    score /= len(candidate_data)
    # print('Embedding Similarity:', score)
    logging.warning('Embedding Similarity: {}'.format(score))
    return score

def evaluate_all(candidate_data, ref_data):
    calculate_bleu_score(candidate_data, ref_data)
    calculate_rouge_score(candidate_data, ref_data)
    calculate_BERTScore(candidate_data, ref_data)
    # calculate_embedding_similarity(candidate_data, ref_data)
    calculate_perplexity(candidate_data)
    # calculate_perplexity(ref_data)

def get_information(candidate_file):
    Avg_bpt = round(pd.read_csv(candidate_file, sep='\t')['BPT'].mean(), 3)
    logging.warning('Average BPT: {}'.format(Avg_bpt))
    Avg_entropy = round(pd.read_csv(candidate_file, sep='\t')['Entropy'].mean(), 3)
    logging.warning('Average Entropy: {}'.format(Avg_entropy))
    Avg_Time = round(pd.read_csv(candidate_file, sep='\t')['Time'].mean(), 3)
    logging.warning('Average Time: {}'.format(Avg_Time))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen2.5-7B-Instruct', required=False)
    parser.add_argument('--window_size', type=int, default=10, required=False)
    args = parser.parse_args()

    ref_file = '3.Stega_data/Normal_{}_supernatural.tsv'.format(args.model)
    
    logging.basicConfig(filename='experiment.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    folder_path = '3.Stega_data'
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for candidate_file in files:
        # if 'Baseline' not in candidate_file:
        #     continue
        # if 'full' not in candidate_file:
        #     continue
        if 'supernatural' not in candidate_file:
            continue
        if 'Normal' in candidate_file:
            continue
        # if 'full' not in candidate_file:
        #     continue
        if args.model not in candidate_file:
            continue
        logging.warning('Evaluating the file: {}'.format(candidate_file))
        candidate_file = folder_path + '/' + candidate_file
        candidate_data = list(pd.read_csv(candidate_file, sep='\t')['Text'])

        ref_data = list(pd.read_csv(ref_file, sep='\t')['Text'])[:len(candidate_data)]  # Ensure the reference data matches the candidate data length

        assert len(candidate_data) == len(ref_data), "The number of candidate data and reference data must be the same."
        logging.warning('Sample size: {}'.format(len(candidate_data)))

        get_information(candidate_file)
        evaluate_all(candidate_data, ref_data)
        pass