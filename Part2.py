import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

data = pd.read_csv("insurance.csv")
data = data.dropna()

features = data.drop(columns=["charges"])
target = data["charges"]

numerical_columns = ["age", "bmi", "children"]
categorical_columns = ["sex", "smoker", "region"]

preprocessing = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), numerical_columns),
        ("encode", OneHotEncoder(drop="first"), categorical_columns)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
majority_class = train_data[train_data["charges"] <= train_data["charges"].median()]
minority_class = train_data[train_data["charges"] > train_data["charges"].median()]

majority_class_downsampled = resample(majority_class, 
                                      replace=False, 
                                      n_samples=len(minority_class), 
                                      random_state=42)

balanced_train_data = pd.concat([majority_class_downsampled, minority_class])

X_train_balanced = balanced_train_data.drop(columns=["charges"])
y_train_balanced = balanced_train_data["charges"]

gb_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessing),
        ("regressor", GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]
)

gb_pipeline.fit(X_train_balanced, y_train_balanced)

predicted = gb_pipeline.predict(X_test)

mae_value = mean_absolute_error(y_test, predicted)
mse_value = mean_squared_error(y_test, predicted)
r2_value = r2_score(y_test, predicted)

print("Performance of the Gradient Boosting Regressor:")
print(f"Mean Absolute Error (MAE): {mae_value:.2f}")
print(f"Mean Squared Error (MSE): {mse_value:.2f}")
print(f"R-squared (R²) Score: {r2_value:.2f}")
