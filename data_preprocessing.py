
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def encode_categorical(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

def plot_distribution(df, numerical_columns):
    for feature in numerical_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, bins=20)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

def remove_outliers(df, columns):
    for feature in columns:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df
