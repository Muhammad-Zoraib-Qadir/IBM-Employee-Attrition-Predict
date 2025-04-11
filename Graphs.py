#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import classification_report, accuracy_score, silhouette_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from scipy.cluster.hierarchy import dendrogram, linkage


sns.set(style="whitegrid")


# In[2]:


df = pd.read_csv("IBM HR Employee Attrition.csv")
df.info()
df.describe()


# In[3]:


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
# Central Tendency: Mean, Median, Mode
central_tendency = pd.DataFrame({
    'Mean': df[numerical_columns].mean(),
    'Median': df[numerical_columns].median(),
    'Mode': df[numerical_columns].mode().iloc[0]
})


# In[4]:


for feature in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True, bins=20)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# In[5]:


# Plot boxplots for numerical columns to detect outliers
for feature in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=feature)
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.show()


# In[6]:


def remove_outliers(df, columns):
    for feature in columns:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df
df_cleaned = remove_outliers(df, numerical_columns)

# Check if any rows were removed
print(f"Original data shape: {df.shape}")
print(f"Cleaned data shape: {df_cleaned.shape}")


# In[8]:


# Plot histograms for cleaned data
for feature in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_cleaned[feature], kde=True, bins=20)
    plt.title(f'Distribution of {feature} (Cleaned)')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Plot boxplots for cleaned data
for feature in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_cleaned, x=feature)
    plt.title(f'Boxplot of {feature} (Cleaned)')
    plt.xlabel(feature)
    plt.show()


# In[7]:


attrition_counts = df['Attrition'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Attrition Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[9]:


print("Missing values per column:")
print(df.isnull().sum())
categorical_columns = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
corr_matrix = df_encoded.corr()
threshold = 0.3
high_corr = corr_matrix[(corr_matrix >= threshold) | (corr_matrix <= -threshold)]
plt.figure(figsize=(15, 12))
sns.heatmap(high_corr, annot=False, cmap="coolwarm", mask=high_corr.isnull(), linewidths=0.5)
plt.show()



# In[10]:


df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
def correlation_analysis(df, threshold=0.1):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if all(upper_triangle[column] < threshold)]
    return df.drop(to_drop, axis=1)
df = correlation_analysis(df)
def select_important_features(df, target, k=10):
    X = df.drop(target, axis=1)
    y = df[target]
    best_features = SelectKBest(score_func=chi2, k=k).fit(X, y)
    return X.columns[best_features.get_support()]
important_features = select_important_features(df, target='Attrition')
X = df[important_features]
y = df['Attrition']


# In[11]:


important_features = ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate',
                      'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                      'YearsWithCurrManager', 'OverTime_Yes']
df_important = df[important_features + ['Attrition']]
continuous_features = ['MonthlyIncome', 'DailyRate', 'MonthlyRate']

non_continuous_features = ['Age', 'DistanceFromHome', 'TotalWorkingYears', 'YearsAtCompany', 
                           'YearsInCurrentRole', 'YearsWithCurrManager', 'OverTime_Yes']
num_continuous = len(continuous_features)
num_non_continuous = len(non_continuous_features)
num_rows_cont = (num_continuous // 2) + (1 if num_continuous % 2 != 0 else 0)
fig_cont, axes_cont = plt.subplots(num_rows_cont, 2, figsize=(12, 5 * num_rows_cont))
axes_cont = axes_cont.flatten()
for idx, feature in enumerate(continuous_features):
    ax = axes_cont[idx]
    sns.boxplot(data=df_important, x='Attrition', y=feature, ax=ax)
    ax.set_title(f'{feature} Distribution by Attrition')
    ax.set_ylabel(feature)
    ax.set_xlabel('Attrition')
for i in range(num_continuous, len(axes_cont)):
    axes_cont[i].axis('off')
num_rows_non_cont = (num_non_continuous // 3) + (1 if num_non_continuous % 3 != 0 else 0)
fig_non_cont, axes_non_cont = plt.subplots(num_rows_non_cont, 3, figsize=(18, 5 * num_rows_non_cont))
axes_non_cont = axes_non_cont.flatten()
for idx, feature in enumerate(non_continuous_features):
    ax = axes_non_cont[idx]
    sns.barplot(data=df_important, x=feature, y='Attrition', ax=ax)
    ax.set_title(f'Attrition Rate by {feature}')
    ax.set_ylabel('Attrition Rate')
    ax.set_xlabel(feature)
for i in range(num_non_continuous, len(axes_non_cont)):
    axes_non_cont[i].axis('off')
plt.tight_layout()
plt.show()


# In[12]:


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}\n")


# In[13]:


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Collect metrics
    model_names.append(name)
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
metrics_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})
metrics_df.set_index('Model').plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon', 'lightgreen', 'gold'])
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc="upper right")
plt.show()


# In[14]:


# Elbow method for optimal cluster count
def find_optimal_clusters(data, max_k=10):
    distortions = []
    for i in range(1, max_k):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(data)
        distortions.append(km.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

find_optimal_clusters(X)


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your DataFrame is 'df', and you're interested in clustering certain columns (drop 'Attrition' or any non-numeric columns)
df_cleaned = df.drop(columns=['Attrition'])  # If 'Attrition' is your target variable, drop it for clustering

# Perform hierarchical/agglomerative clustering using the 'ward' method
Z = linkage(df_cleaned, method='ward')

# Create a dendrogram to visualize the clustering
plt.figure(figsize=(12, 8))
dendrogram(Z, truncate_mode='level', p=5, show_leaf_counts=True, leaf_rotation=90)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Samples')
plt.grid(False)  # Remove grid lines
plt.ylabel('Distance')
plt.show()


# In[16]:


df_cleaned = df.select_dtypes(include=[np.number])  # Select only numerical columns
cluster_range = range(2, 11)  # Try from 2 to 10 clusters
silhouette_scores = []

# Loop through different cluster numbers
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df_cleaned)
    silhouette_avg = silhouette_score(df_cleaned, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg}")

# Plotting the silhouette scores for different cluster numbers
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Score for Different Cluster Numbers')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


# In[17]:


def cluster_data(data, n_clusters):
    data_array = data.values if isinstance(data, pd.DataFrame) else data

    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data_array)
    kmeans_silhouette = silhouette_score(data_array, kmeans_labels)
    kmeans_davies_bouldin = davies_bouldin_score(data_array, kmeans_labels)
    
    print(f"KMeans Silhouette Score: {kmeans_silhouette}, Davies-Bouldin Score: {kmeans_davies_bouldin}")
    print("KMeans Cluster Labels and Counts:", np.unique(kmeans_labels, return_counts=True))

    # Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_labels = agglomerative.fit_predict(data_array)
    agglomerative_silhouette = silhouette_score(data_array, agglomerative_labels)
    agglomerative_davies_bouldin = davies_bouldin_score(data_array, agglomerative_labels)
    
    print(f"Agglomerative Silhouette Score: {agglomerative_silhouette}, Davies-Bouldin Score: {agglomerative_davies_bouldin}")
    print("Agglomerative Cluster Labels and Counts:", np.unique(agglomerative_labels, return_counts=True))
    # Visualizing the clustering results
    plt.figure(figsize=(12, 8))

    # KMeans Plot
    plt.subplot(2, 2, 1)
    plt.scatter(data_array[:, 0], data_array[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('KMeans Clustering')

    # Agglomerative Clustering Plot
    plt.subplot(2, 2, 2)
    plt.scatter(data_array[:, 0], data_array[:, 1], c=agglomerative_labels, cmap='viridis')
    plt.title('Agglomerative Clustering')
    plt.tight_layout()
    plt.show()

# Assuming X is your input data (e.g., a 2D array or DataFrame)
optimal_clusters = 3  # Replace with the number of clusters you determined as optimal
cluster_data(X, n_clusters=optimal_clusters)


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from sklearn.decomposition import PCA  # For dimensionality reduction
from sklearn.metrics import silhouette_samples

def cluster_data_viz(data, n_clusters):
    # Convert data to a NumPy array for easier slicing
    data_array = data.values if isinstance(data, pd.DataFrame) else data

    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data_array)
    
    # 1. Pair Plot with KMeans Labels
    data['KMeans_Label'] = kmeans_labels
    sns.pairplot(data, hue='KMeans_Label', palette='viridis')
    plt.suptitle('Pair Plot of Clusters', y=1.02)
    plt.show()
    
    # 2. 3D Scatter Plot with PCA
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_array)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=kmeans_labels, cmap='viridis')
    plt.colorbar(scatter)
    ax.set_title('3D PCA of KMeans Clusters')
    plt.show()



# Run the visualization function
cluster_data_viz(X, n_clusters=optimal_clusters)


# In[ ]:




