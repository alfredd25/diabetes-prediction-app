import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Setup Folder Structure
# This ensures the 'assets/images' folder exists so the App can find the plots
os.makedirs('assets/images', exist_ok=True)

# 2. Load Data
df = pd.read_csv('diabetes.csv')

# 3. Data Cleaning
# Replace 0s with NaN and then fill with Median
cols_with_missing_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_missing_values] = df[cols_with_missing_values].replace(0, np.nan)

for col in cols_with_missing_values:
    df[col] = df[col].fillna(df[col].median())

# Save Cleaned Data (Required for save_all_models.py)
df.to_csv('diabetes_cleaned.csv', index=False)
print("Data cleaned and saved to 'diabetes_cleaned.csv'")

# 4. Visualizations
sns.set(style="whitegrid")

# Plot 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig('assets/images/viz1_correlation.png') # <--- Saved to correct folder
print("Saved viz1_correlation.png")
# plt.show() # Commented out so it runs faster, uncomment if you want to see it pop up

# Plot 2: Pie Chart
plt.figure(figsize=(6, 6))
df['Outcome'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Diabetes', 'Diabetes'], colors=['#ff9999','#66b3ff'])
plt.title("Percentage of Diabetic Patients")
plt.ylabel('')
plt.savefig('assets/images/viz2_outcome_pie.png')
print("Saved viz2_outcome_pie.png")

# Plot 3: Glucose Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Outcome', y='Glucose', data=df, palette="Set2")
plt.title("Glucose Levels: Diabetic vs Non-Diabetic")
plt.savefig('assets/images/viz3_glucose_box.png')
print("Saved viz3_glucose_box.png")

# Plot 4: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Outcome', kde=True, palette="husl")
plt.title("Age Distribution by Diabetes Outcome")
plt.savefig('assets/images/viz4_age_dist.png')
print("Saved viz4_age_dist.png")

# Plot 5: Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, palette="deep", alpha=0.6)
plt.title("Glucose vs BMI Scatter Plot")
plt.savefig('assets/images/viz5_glucose_bmi_scatter.png')
print("Saved viz5_glucose_bmi_scatter.png")

print("\nSuccess! All images saved to 'assets/images/'.")