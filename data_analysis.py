# data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def analyze_data(df):
    print(df.info())
    print(df.describe())

def plot_attrition_distribution(df):
    attrition_counts = df['Attrition'].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.title('Attrition Distribution')
    plt.axis('equal')
    plt.show()

def check_missing_values(df):
    print("Missing values per column:")
    print(df.isnull().sum())

def correlation_heatmap(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    corr_matrix = df_encoded.corr()
    threshold = 0.3
    high_corr = corr_matrix[(corr_matrix >= threshold) | (corr_matrix <= -threshold)]
    plt.figure(figsize=(15, 12))
    sns.heatmap(high_corr, annot=False, cmap="coolwarm", mask=high_corr.isnull(), linewidths=0.5)
    plt.show()

