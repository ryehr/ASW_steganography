from datasets import load_dataset
import csv

dataset = load_dataset("databricks/databricks-dolly-15k")
file_name = 'databricks-dolly.tsv'
header = ['part', 'new_id','question', 'answer']
with open(file_name, 'w', newline = '', encoding = 'utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(header)
    num = 0
    for i in range(len(dataset['train'])):
        if num == 500:
            break
        if dataset['train'][i]['category'] == 'open_qa':
            writer.writerow(['test', num, dataset['train'][i]['instruction'], ''])
            num += 1
