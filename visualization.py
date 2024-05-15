import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_top_feature_correlation(df, top_features):
  top_features = list(top_features)
  if 'rating' not in top_features:
     top_features.append('rating')
  selected_data = df[top_features]
  correlation_matrix = selected_data.corr()
  plt.figure(figsize=(12, 8))
  sns.heatmap(correlation_matrix, annot=True)
  plt.show()
  

