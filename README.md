# IBM Employee Attrition Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.2%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A data-driven analysis of IBM employee attrition patterns using classification, clustering, and EDA. Predicts attrition risk factors and identifies key drivers behind employee turnover.

---

## Dataset: IBM HR Employee Attrition
**Source**: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
**Size**: 1470 employees  
**Features**:
- Demographics (Age, Gender, Marital Status)
- Job-related (Job Role, Department, Monthly Income)
- Behavioral (Job Satisfaction, Work-Life Balance)
- Target Variable: `Attrition` (Yes/No)

---

# Project Structure 📂
- ├── classification.py # 🤖 ML Models (Logistic Regression, Decision Trees, Random Forest)
- ├── clustering.py # 🎯 Clustering Algorithms (K-Means)
- ├── data_preprocessing.py # 🧹 Data Cleaning & Feature Engineering
- ├── Graphs.py # 📊 Visualizations (Heatmaps, Distributions)
- ├── main.py # 🚀 Part 1: Full Analysis Pipeline
- ├── Part2.py # 📈 Part 2: Model Evaluation & Comparison
- ├── Part3.py # 💡 Part 3: Business Recommendations
- ├── Report.pdf # 📄 Detailed Analysis Report
- ├── requirements.txt # 📦 Python Dependencies
- └── README.md # 📋 Project Documentation

---

## 🔑 Key Insights

### Top Attrition Drivers
- **Low job satisfaction** (`JobSatisfaction < 2`)
- **Income below department average**
- **Overtime participation**

### Model Performance
- **Best Model**: Random Forest  
  - Accuracy: **87%**  
  - F1-Score: **0.83**

### Clustering
- **3 distinct employee groups** identified for targeted retention strategies.

---

## 📄 Report Overview (`Report.pdf`)
- **Methodology**: Data preprocessing, model selection, and clustering approach.
- **Performance Metrics**: Accuracy, precision, recall, and F1-score comparisons.
- **Business Recommendations**: 7+ actionable strategies to reduce attrition.
- **Visualizations**:  
  - Feature importance plots  
  - Cluster analysis charts  
  - Heatmaps and distributions

---

## 🚀 Usage
## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/IBM-Employee-Attrition-Analytics.git
Install dependencies:
     
      pip install -r requirements.txt
## 📦 Dependencies 
**See:** `requirements.txt` for full list of Python packages.

---

## 💼 Business Recommendations
1. 🕒 **Implement flexible work arrangements** for employees working overtime  
2. 💰 **Address salary disparities** for underpaid employees  
3. 😊 **Launch job satisfaction programs**  
   - Mentorship initiatives  
   - Employee recognition systems  
4. 🎯 **Develop department-specific retention policies**  
5. 🛠️ **Offer skill development opportunities** for low-satisfaction clusters  
6. ⏱️ **Redesign overtime compensation structures**  
7. 📈 **Create career progression plans** aligned with employee clusters  

---

## 📊 Visualizations  
Key visual outputs include:  
- **Feature importance** (Random Forest-based attrition drivers)  
- **Cluster analysis** (3 employee groups visualized with PCA/K-means)  
- **Heatmaps** (Correlation matrices of attrition factors)  
- **Distribution plots**:  
  - Job Satisfaction vs. Attrition  
  - Monthly Income comparisons across departments


License
This project is licensed under the MIT License.
---

### Why This Works:

1. **Dataset Clarity**: Explicitly describes the IBM dataset source and features.
2. **Reproducibility**: Step-by-step instructions for running all components.
3. **Actionable Insights**: Highlights attrition drivers and model performance upfront.
4. **Structure Alignment**: Matches your file organization and script naming.
5. **Business Focus**: Connects technical analysis to HR strategies.
