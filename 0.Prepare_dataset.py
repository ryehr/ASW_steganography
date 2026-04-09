import json
import csv
from transformers import AutoTokenizer
import random

with open('instinwild_en.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
file_name = 'instinwild_en.tsv'
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

header = ['part', 'new_id','question', 'answer']

train_range = list(range(0, int(len(data) * 0.6)))
random.shuffle(train_range)
validation_range = list(range(int(len(data) * 0.6), int(len(data) * 0.8)))
random.shuffle(validation_range)
test_range = list(range(int(len(data) * 0.8), len(data)))
random.shuffle(test_range)

train_num = 12500
validation_num = 1000
test_num = 500

with open(file_name, 'w', newline = '', encoding = 'utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(header)
    for part in ['train', 'validation', 'test']:
        current_num = 0
        if part == 'train':
            range_data = train_range
        elif part == 'validation':
            range_data = validation_range
        else:
            range_data = test_range
        for i in range_data:
            question = data[i]['instruction']
            answer = data[i]['output']
            new_id = current_num
            answer_length = tokenizer(answer, return_tensors='pt')['input_ids'].shape[1]
            if answer_length > 512 or answer_length < 20 or len(question) < 2:
                continue
            writer.writerow([part, new_id, question, answer])
            current_num += 1
            if part == 'train' and current_num >= train_num:
                break
            elif part == 'validation' and current_num >= validation_num:
                break
            elif part == 'test' and current_num >= test_num:
                break