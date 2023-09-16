# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:39:48 2023

@author: HP
"""

# Essential
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as ex
import plotly.express as px
import plotly.graph_objects as go
import missingno as msno 

# Machine learning and statistical
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# Ignore useless warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("D:\\Diabetes\\diabetes.csv") # read data frame.
df.shape
df.head(10)
df.info()
df.describe().T


#Correlation
corr = df.corr()
plt.figure() #plot the heatmap for the correlation
sns.heatmap(corr,fmt=".5f", linewidth=.5, cmap="coolwarm") 
plt.show() #the more darker color the more stronger correlation.

#Data Preprocessing
# Check for duplicates across all columns
duplicated = df.duplicated().sum()

# Print the number of duplicated instances
if duplicated == 0:
    print("Number of duplicated instances:", duplicated)
# Print the duplicated instances
else:
    print("Number of duplicated instances:", duplicated)
    print(df[duplicated])
    
# print the percentage of missing values for instances.
total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
percent = ((df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)[(df.isnull().sum() / df.isnull().count()).sort_values(ascending=False) != 0]) * 100
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing['Percent'] = missing['Percent'].apply(lambda x: "%.2f%%" % x)
print(missing)
# null count analysis
p=msno.bar(df)

#Plot relationship between variables
# Count the occurrences of each outcome (0: No diabetes, 1: Diabetes)
outcome_counts = df['Outcome'].value_counts()
# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Features to plot (excluding 'Outcome' which is the target variable)
features = df.columns[:-1]

# Plot histograms for each feature
for i, feature in enumerate(features):
    ax = axes[i]
    ax.hist(df[df['Outcome'] == 0][feature], alpha=0.5, label='No Diabetes', color='blue', bins=20)
    ax.hist(df[df['Outcome'] == 1][feature], alpha=0.5, label='Diabetes', color='red', bins=20)
    ax.set_title(feature)
    ax.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()

# Create a bar plot
plt.figure(figsize=(6, 4))
plt.bar(outcome_counts.index, outcome_counts.values, tick_label=['No Diabetes', 'Diabetes'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Distribution of Outcome (Diabetes vs. No Diabetes)')

# Show the plot
plt.show()
# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Features to plot (excluding 'Outcome' which is the target variable)
features = df.columns[:-1]

# Plot histograms for each feature
for i, feature in enumerate(features):
    ax = axes[i]
    ax.hist(df[df['Outcome'] == 0][feature], alpha=0.5, label='No Diabetes', color='blue', bins=20)
    ax.hist(df[df['Outcome'] == 1][feature], alpha=0.5, label='Diabetes', color='red', bins=20)
    ax.set_title(feature)
    ax.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()
sns.pairplot(df, hue='Outcome', diag_kind='kde')
plt.show()

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define colors for diabetes status (0 for No Diabetes, 1 for Diabetes)
colors = {0: 'blue', 1: 'red'}

# Scatter plot with Age on the x-axis, BMI on the y-axis, and Glucose on the z-axis
for outcome, color in colors.items():
    subset = df[df['Outcome'] == outcome]
    ax.scatter(subset['Age'], subset['BMI'], subset['Glucose'], c=color, label=f'Diabetes {outcome}')

# Set labels for each axis
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Glucose')

# Add a legend
ax.legend()

# Show the 3D scatter plot
plt.show()
# Create subplots with two violin plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Violin Plot 1: Age by Outcome (Diabetes vs. No Diabetes)
sns.violinplot(x='Outcome', y='Age', data=df, palette='Set1', ax=axes[0])
axes[0].set_title('Violin Plot: Age by Outcome')
axes[0].set_xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
axes[0].set_ylabel('Age')

# Violin Plot 2: Diabetes Pedigree Function by Outcome (Diabetes vs. No Diabetes)
sns.violinplot(x='Outcome', y='DiabetesPedigreeFunction', data=df, palette='Set1', ax=axes[1])
axes[1].set_title('Violin Plot: Diabetes Pedigree Function by Outcome')
axes[1].set_xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
axes[1].set_ylabel('Diabetes Pedigree Function')

# Adjust layout
plt.tight_layout()

# Show the subplots
plt.show()

# Create subplots with two box plots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box Plot 1: Blood Pressure by Outcome (Diabetes vs. No Diabetes)
sns.boxplot(x='Outcome', y='BloodPressure', data=df, palette='Set1', ax=axes[0])
axes[0].set_title('Box Plot: Blood Pressure by Outcome')
axes[0].set_xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
axes[0].set_ylabel('Blood Pressure')

# Box Plot 2: Insulin by Outcome (Diabetes vs. No Diabetes)
sns.boxplot(x='Outcome', y='Insulin', data=df, palette='Set1', ax=axes[1])
axes[1].set_title('Box Plot: Insulin by Outcome')
axes[1].set_xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
axes[1].set_ylabel('Insulin')

# Adjust layout
plt.tight_layout()

# Show the subplots
plt.show()



#Machine Learning
# Extract the features (X) and target variable (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
fig = px.pie(y_train,names='Outcome')
fig.update_layout(title='<b>Outcome Proportion before SMOTE Upsampling</b>')
fig.show()
fig = px.pie(y_train,names='Outcome')
fig.update_layout(title='<b>Outcome Proportion before SMOTE Upsampling</b>')
fig.show()
# Define a list of classifiers to evaluate
classifiers = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42)),
    ('SVM', SVC(kernel='linear', C=1)),
    ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5)),
    ('Naive Bayes', GaussianNB())
]

# Print evaluation results with improved formatting
for name, classifier in classifiers:
    # Fit the model on the training data
    classifier.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = classifier.predict(X_test)
    print(f"{name}")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    print("-" * 50)
    
    # Perform 10-fold cross-validation for each model
for name, classifier in classifiers:
    # Create a KFold cross-validator
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Perform cross-validation
    scores = cross_val_score(classifier, X_train, y_train, cv=kf)
    
    # Print model name and cross-validation scores

    print(name)
    
    # Calculate and print the mean and standard deviation of the cross-validation scores
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Mean Accuracy: {mean_score:.4f}")
    print(f"Standard Deviation: {std_score:.4f}")
    
    # Print a separator for better readability
    print("-" * 50)