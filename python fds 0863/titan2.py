# ==========================================
# TITANIC DATASET ANALYSIS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# (a) Read the Dataset
# ==========================================

df = pd.read_csv("train.csv")

print("First 5 Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================================
# Exploratory Data Analysis (EDA)
# ==========================================

# Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Gender Count
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

# Passenger Class
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class")
plt.show()

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Fare Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Fare'], bins=20, kde=True)
plt.title("Fare Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Matrix")
plt.show()

# ==========================================
# (b) Identify Attributes
# ==========================================

print("\nNumeric Attributes")
numeric = df.select_dtypes(include=['int64','float64']).columns
print(numeric)

print("\nNominal Attributes")
nominal = ['Sex','Embarked']

print(nominal)

print("\nBinary Attributes")
binary = ['Survived']

print(binary)

# ==========================================
# (c) Dissimilarity for Nominal Attributes
# Example:
# Sex and Embarked
# Simple Matching Dissimilarity
# ==========================================

sex = df['Sex']
embarked = df['Embarked']

count = 0
total = 0

for s,e in zip(sex,embarked):

    if pd.isna(s) or pd.isna(e):
        continue

    total += 1

    if s != e:
        count += 1

dissimilarity = count/total

print("\nNominal Attribute Dissimilarity")
print("Sex vs Embarked =", dissimilarity)

# ==========================================
# (d) Numeric Attribute Dissimilarity
# Example:
# Age and Fare
# Euclidean Distance
# ==========================================

# Remove Missing Values

temp = df[['Age','Fare']].dropna()

age = temp.iloc[0]['Age']
fare = temp.iloc[0]['Fare']

age2 = temp.iloc[1]['Age']
fare2 = temp.iloc[1]['Fare']

distance = np.sqrt((age-age2)**2 + (fare-fare2)**2)

print("\nEuclidean Distance")
print(distance)

# ==========================================
# Manhattan Distance
# ==========================================

manhattan = abs(age-age2)+abs(fare-fare2)

print("\nManhattan Distance")
print(manhattan)

# ==========================================
# Minkowski Distance (p=3)
# ==========================================

minkowski=((abs(age-age2)**3)+(abs(fare-fare2)**3))**(1/3)

print("\nMinkowski Distance")
print(minkowski)

# ==========================================
# Cosine Dissimilarity
# ==========================================

A=np.array([age,fare])
B=np.array([age2,fare2])

cosine_similarity=np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

cosine_dissimilarity=1-cosine_similarity

print("\nCosine Dissimilarity")
print(cosine_dissimilarity)