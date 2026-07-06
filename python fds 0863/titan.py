import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

# Read the dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print(df.head())

# Shape of dataset
print("Shape:", df.shape)

# Information about dataset
print(df.info())

# Summary statistics
print(df.describe())

# Missing values
print("mv:",df.isnull().sum())

# Duplicate rows
print("Duplicate rows:", df.duplicated().sum())
# Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Passenger Class
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()

# Gender Distribution
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

# Age Distribution
plt.hist(df['Age'].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Fare Distribution
plt.hist(df['Fare'], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
nominal = df[['Sex', 'Embarked']].dropna().reset_index(drop=True)

row1 = nominal.iloc[0]
row2 = nominal.iloc[1]

matches = 0

for i in range(len(row1)):
    if row1[i] == row2[i]:
        matches += 1

dissimilarity = 1 - (matches / len(row1))

print("Nominal Dissimilarity:", dissimilarity)

numeric = df[['Age', 'Fare']].dropna().reset_index(drop=True)

record1 = numeric.iloc[0]
record2 = numeric.iloc[1]

distance = np.sqrt(((record1 - record2) ** 2).sum())

print("Euclidean Distance:", distance)