# US AIR QUALITY INTELLIGENCE
# Predictive Analytics Project
# PM2.5 Analysis (2000-2026)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)


# DATA EXPLORATION

df = pd.read_csv("final_pm25_dataset.csv")

print("\nFirst Five Records:")
print(df.head())

print("\nLast Five Records:")
print(df.tail())

print("\nDataset Information:")
df.info()

print("\nDataset Description:")
print(df.describe())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nYear-wise Records:")
print(df["Year"].value_counts().sort_index())

print("\nColumn Names:")
print(df.columns)


# DATA PREPROCESSING

# Handling Missing Numerical Values

numeric_cols = df.select_dtypes(include="number").columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Handling Missing Categorical Values

categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# Outlier Detection Using IQR

q1 = df["Arithmetic Mean"].quantile(0.25)
q3 = df["Arithmetic Mean"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[
    (df["Arithmetic Mean"] < lower) |
    (df["Arithmetic Mean"] > upper)
]

print("\nNumber of Outliers:", len(outliers))

plt.figure(figsize=(6, 4))
plt.boxplot(df["Arithmetic Mean"])
plt.title("PM2.5 Outlier Detection")
plt.ylabel("PM2.5 Mean")
plt.show()

# Normalization

features = [
    "Year",
    "Latitude",
    "Longitude",
    "Observation Count",
    "Observation Percent"
]

X = df[features]
y = df["Arithmetic Mean"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nNormalized Features:")
print(X_scaled[:5])


# EXPLORATORY DATA ANALYSIS

# Year-wise PM2.5 Trend

year_data = df.groupby("Year")["Arithmetic Mean"].mean()

plt.figure(figsize=(10, 5))
plt.plot(year_data.index, year_data.values, marker="o")
plt.title("Year-wise PM2.5 Trend")
plt.xlabel("Year")
plt.ylabel("Average PM2.5")
plt.grid()
plt.show()


# PM2.5 Distribution

plt.figure(figsize=(8, 5))
plt.hist(df["Arithmetic Mean"], bins=30)
plt.title("PM2.5 Distribution")
plt.xlabel("PM2.5 Mean")
plt.ylabel("Frequency")
plt.show()


# Correlation Heatmap

plt.figure(figsize=(8, 5))

sns.heatmap(
    df[features + ["Arithmetic Mean"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()


# REGRESSION MODELLING

X = df[features]
y = df["Arithmetic Mean"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 1. Simple Linear Regression

simple_model = LinearRegression()

simple_model.fit(X_train[["Year"]], y_train)

simple_pred = simple_model.predict(X_test[["Year"]])

print("\nSimple Linear Regression")

print("MAE:", mean_absolute_error(y_test, simple_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, simple_pred)))

print("R2 Score:", r2_score(y_test, simple_pred))

# 2. Multiple Linear Regression

multiple_model = LinearRegression()

multiple_model.fit(X_train_scaled, y_train)

multiple_pred = multiple_model.predict(X_test_scaled)

print("\nMultiple Linear Regression")

print("MAE:", mean_absolute_error(y_test, multiple_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, multiple_pred)))

print("R2 Score:", r2_score(y_test, multiple_pred))

# 3. Polynomial Regression

poly = PolynomialFeatures(degree=2)

year_mean = X_train["Year"].mean()

X_poly_train = poly.fit_transform(X_train[["Year"]] - year_mean)

X_poly_test = poly.transform(X_test[["Year"]] - year_mean)

poly_model = LinearRegression()

poly_model.fit(X_poly_train, y_train)

poly_pred = poly_model.predict(X_poly_test)

print("\nPolynomial Regression")

print("MAE:", mean_absolute_error(y_test, poly_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, poly_pred)))

print("R2 Score:", r2_score(y_test, poly_pred))

# Regression Model Comparison

regression_results = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Polynomial Regression"
    ],

    "MAE": [
        mean_absolute_error(y_test, simple_pred),
        mean_absolute_error(y_test, multiple_pred),
        mean_absolute_error(y_test, poly_pred)
    ],

    "RMSE": [
        np.sqrt(mean_squared_error(y_test, simple_pred)),
        np.sqrt(mean_squared_error(y_test, multiple_pred)),
        np.sqrt(mean_squared_error(y_test, poly_pred))
    ],

    "R2 Score": [
        r2_score(y_test, simple_pred),
        r2_score(y_test, multiple_pred),
        r2_score(y_test, poly_pred)
    ]
})

print("\nRegression Model Comparison:")
print(regression_results)


# CLASSIFICATION

# Creating Binary Classification Target
# 0 = Lower PM2.5
# 1 = Elevated PM2.5

median_value = df["Arithmetic Mean"].median()

df["Pollution_Level"] = (
    df["Arithmetic Mean"] >= median_value
).astype(int)

X = df[features]
y = df["Pollution_Level"]


# Splitting Dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handling Missing Values

imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Logistic Regression

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

print("\nLogistic Regression")

print("Accuracy:", accuracy_score(y_test, log_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, log_pred))

# 2. K-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

knn_pred = knn.predict(X_test_scaled)

print("\nKNN Classification")

print("Accuracy:", accuracy_score(y_test, knn_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))

# 3. Naive Bayes

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

print("\nNaive Bayes Classification")

print("Accuracy:", accuracy_score(y_test, nb_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, nb_pred))

# 4. Decision Tree

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Classification")

print("Accuracy:", accuracy_score(y_test, dt_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

# 5. Support Vector Machine

svm = SVC()

svm.fit(X_train_scaled, y_train)

svm_pred = svm.predict(X_test_scaled)

print("\nSupport Vector Machine")

print("Accuracy:", accuracy_score(y_test, svm_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))

# CLASSIFICATION MODEL COMPARISON

results = pd.DataFrame({

    "Classifier": [
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "Decision Tree",
        "SVM"
    ],

    "Accuracy": [
        accuracy_score(y_test, log_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, svm_pred)
    ]

})

results = results.sort_values(
    by="Accuracy", ascending=False
)

print("\nClassifier Performance Comparison:")
print(results)

# Classification Accuracy Graph

plt.figure(figsize=(8, 5))

plt.bar(
    results["Classifier"],
    results["Accuracy"]
)

plt.title("Classification Model Comparison")
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()