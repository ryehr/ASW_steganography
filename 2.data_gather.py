import pandas as pd

model_name = 'Qwen2.5-7B-Instruct'
window_size = 10
lora = 1  # 0: no lora, 1: lora

filename = '2.data/{}_window_{}_lora_{}.tsv'.format(model_name, window_size, lora)

df = pd.read_csv(filename, sep='\t').iloc[0:100]
print('In-dataset (Size: {}):'.format(len(df)))
for strategy in ['Baseline', 'Hard_0', 'Hard_1', 'Hard_2', 'Soft_0', 'Soft_1', 'Soft_2']:
    KL_avg = round(df[strategy].mean(), 3)
    KL_median = round(df[strategy].median(), 3)
    print(strategy, 'Avg:', KL_avg, 'Median:', KL_median)

print()

df = pd.read_csv(filename, sep='\t').iloc[100:]
print('Out-of-dataset (Size: {}):'.format(len(df)))
for strategy in ['Baseline', 'Hard_0', 'Hard_1', 'Hard_2', 'Soft_0', 'Soft_1', 'Soft_2']:
    KL_avg = round(df[strategy].mean(), 3)
    KL_median = round(df[strategy].median(), 3)
    print(strategy, 'Avg:', KL_avg, 'Median:', KL_median)