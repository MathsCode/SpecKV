import json
data_set = []
with open("/home/xujiaming/xujiaming/Paper/SpecKV/SpecKV/pred/Meta-Llama-3.1-8B-Instruct/qasper-tech2-read.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data_set.append(json.loads(line))
        

import math


prob_set = {str(i):[] for i in [32,64,128,256,512,1024,2048,4096,8192]}


for data in data_set:
    for key, value in data['output'].items():
        max_prob = math.exp(value['prob'][0][0])
        for i in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            if str(i) in value:
                prob_set[str(i)].append(max_prob)


# draw the box plot

labels = list(prob_set.keys())
values = list(prob_set.values())

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
plt.figure(figsize=(10, 6))


sns.boxplot(data=values, palette="Set3")
plt.xticks(range(len(labels)), labels)

plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("./box_plot_distribution.png", dpi=300, bbox_inches='tight')

plt.clf()

for i, value in enumerate(values):
    plt.figure(figsize=(10, 6))
    plt.hist(value, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'./{i}', dpi=300, bbox_inches='tight')
    plt.clf()

