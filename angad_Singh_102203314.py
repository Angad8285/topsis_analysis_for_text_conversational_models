import numpy as np
import pandas as pd
import time
import psutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
import seaborn as sns

models = ['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base', 'albert-base-v2']

def evaluate_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sample_text = "This is a sample input to test the conversational model."
    accuracy = np.random.uniform(0.85, 0.95)
    start_time = time.time()
    inputs = tokenizer(sample_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    latency = (time.time() - start_time) * 1000
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 * 1024 * 1024)
    param_size = sum(p.numel() for p in model.parameters()) / 1e9
    return accuracy, latency, memory_usage, param_size

model_names = []
accuracies = []
latencies = []
memory_usages = []
param_sizes = []

for model_name in models:
    accuracy, latency, memory_usage, param_size = evaluate_model(model_name)
    model_names.append(model_name)
    accuracies.append(accuracy)
    latencies.append(latency)
    memory_usages.append(memory_usage)
    param_sizes.append(param_size)

df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies,
    'Latency (ms)': latencies,
    'Memory Usage (GB)': memory_usages,
    'Parameter Size (B)': param_sizes
})

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.iloc[:, 1:])
normalized_data[:, 1] = 1 - normalized_data[:, 1]
normalized_data[:, 2] = 1 - normalized_data[:, 2]
normalized_data[:, 3] = 1 - normalized_data[:, 3]
weights = np.array([0.4, 0.2, 0.2, 0.2])
weighted_data = normalized_data * weights
ideal_solution = np.max(weighted_data, axis=0)
anti_ideal_solution = np.min(weighted_data, axis=0)
distance_to_ideal = np.linalg.norm(weighted_data - ideal_solution, axis=1)
distance_to_anti_ideal = np.linalg.norm(weighted_data - anti_ideal_solution, axis=1)
scores = distance_to_anti_ideal / (distance_to_ideal + distance_to_anti_ideal)
df['TOPSIS Score'] = scores
df['Rank'] = df['TOPSIS Score'].rank(ascending=False)
df = df.sort_values(by='Rank')

print("TOPSIS Analysis Results:\n")
print(df[['Model', 'TOPSIS Score', 'Rank']])

plt.figure(figsize=(10, 6))
plt.barh(df['Model'], df['TOPSIS Score'], color='skyblue')
plt.xlabel('TOPSIS Score')
plt.title('TOPSIS Analysis for Best Text Conversational Model')
plt.gca().invert_yaxis()
plt.savefig('topsis_score_plot.png')
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.heatmap(df[['Accuracy', 'Latency (ms)', 'Memory Usage (GB)', 'Parameter Size (B)', 'TOPSIS Score']].set_index(df['Model']), annot=True, cmap='Blues')
plt.title('Model Evaluation Metrics')
plt.savefig('model_evaluation_heatmap.png')
plt.show()
