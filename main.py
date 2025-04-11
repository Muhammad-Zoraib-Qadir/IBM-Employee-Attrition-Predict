# main.py
import data_analysis as da
import data_preprocessing as dp
import classification as clf
import clustering as cl
import pandas as pd

# Load and analyze data
file_path = "IBM HR Employee Attrition.csv"
df = da.load_data(file_path)
da.analyze_data(df)
da.plot_attrition_distribution(df)
da.check_missing_values(df)

# Preprocess the data
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_encoded = dp.encode_categorical(df)
df_cleaned = dp.remove_outliers(df_encoded, numerical_columns)
df_classification = df_cleaned.copy()
if 'Attrition' not in df_classification.columns:
    df_classification = pd.read_csv("IBM HR Employee Attrition.csv")
df_cleaned = df_classification
# Classification
df_preprocessed = clf.preprocess_target(df_cleaned)
target = 'Attrition'
X = df_preprocessed.drop(columns=[target])
y = df_preprocessed[target]
classification_results = clf.train_classification_models(X, y)
for model, result in classification_results.items():
    print(f"{model} - Accuracy: {result['accuracy']}")
    print(result)

# Clustering
optimal_clusters = 3  # Assuming an optimal number of clusters
cl.perform_clustering(X, n_clusters=optimal_clusters)
