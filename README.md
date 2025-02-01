# Text Conversational Model Evaluation Using TOPSIS

## Overview
This project implements a comprehensive evaluation system for text conversational models using the TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) method. The analysis compares different pretrained models including DistilBERT, BERT, RoBERTa, and ALBERT across multiple performance metrics.

## Implementation Details

### Dependencies
```python
numpy
pandas
time
psutil
transformers
scikit-learn
torch
matplotlib
seaborn
```

### Models Evaluated
- distilbert-base-uncased
- bert-base-uncased
- roberta-base
- albert-base-v2

### Evaluation Metrics
The analysis considers four key metrics:
1. **Accuracy** (40% weight): Model prediction accuracy
2. **Latency** (20% weight): Processing time in milliseconds
3. **Memory Usage** (20% weight): RAM consumption in GB
4. **Parameter Size** (20% weight): Model size in billions of parameters

## Methodology

### Data Collection
- Models are loaded using HuggingFace's transformers library
- Each model processes a sample text input
- Performance metrics are collected using:
  - Random sampling for accuracy simulation
  - Time measurement for latency
  - psutil for memory usage
  - Parameter counting for model size

### TOPSIS Analysis
1. **Data Normalization**: 
   - MinMaxScaler is used to normalize all metrics
   - Latency, memory usage, and parameter size are inverted (1 - normalized_value)

2. **Weight Application**:
   - Metrics are weighted according to their importance
   - Weights: [0.4, 0.2, 0.2, 0.2]

3. **Solution Comparison**:
   - Calculates ideal and anti-ideal solutions
   - Computes Euclidean distances

4. **Score Calculation**:
   - TOPSIS scores determined by relative distances
   - Models ranked based on final scores

## Results

### Performance Visualization
Two plots are generated:

1. **TOPSIS Score Plot** (`topsis_score_plot.png`):
   - Horizontal bar chart showing final TOPSIS scores
   - Higher scores indicate better overall performance
   - DistilBERT-base-uncased achieves the highest score

2. **Evaluation Metrics Heatmap** (`model_evaluation_heatmap.png`):
   - Shows all metrics in a color-coded format
   - Darker blue indicates better performance
   - Provides comprehensive view of model characteristics

### Key Findings
Based on the heatmap and TOPSIS scores:
- DistilBERT-base-uncased shows the best overall performance
- ALBERT-base-v2 ranks second
- Models show different trade-offs between accuracy and resource usage

## Usage
To run the evaluation:
```python
python angad_Singh_102203314.py
```

The script will:
1. Load and evaluate all models
2. Generate performance metrics
3. Perform TOPSIS analysis
4. Create visualization plots
5. Save results to PNG files