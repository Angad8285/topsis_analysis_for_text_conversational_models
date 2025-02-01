import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = {
    'Model': ['GPT-4', 'Claude 2', 'DialoGPT-medium', 'BlenderBot 400M', 'Falcon 40B'],
    'Accuracy': [0.95, 0.91, 0.87, 0.89, 0.92],
    'Latency (ms)': [150, 120, 200, 250, 180],
    'Memory Usage (GB)': [6, 4, 3, 2, 8],
    'Parameter Size (B)': [175, 70, 0.34, 0.4, 40]
}

df = pd.DataFrame(data)

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

import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.heatmap(df[['Accuracy', 'Latency (ms)', 'Memory Usage (GB)', 'Parameter Size (B)', 'TOPSIS Score']].set_index(df['Model']), annot=True, cmap='Blues')
plt.title('Model Evaluation Metrics')
plt.savefig('model_evaluation_heatmap.png')
plt.show()
