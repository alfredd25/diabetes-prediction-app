import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('diabetes.csv')


cols_with_missing_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[cols_with_missing_values] = df[cols_with_missing_values].replace(0, np.nan)

for col in cols_with_missing_values:
    df[col] = df[col].fillna(df[col].median())

df.to_csv('diabetes_cleaned.csv', index=False)
print("Data cleaned and saved to 'diabetes_cleaned.csv'")


sns.set(style="whitegrid")

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig('viz1_correlation.png')
plt.show()

plt.figure(figsize=(6, 6))
df['Outcome'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Diabetes', 'Diabetes'], colors=['#ff9999','#66b3ff'])
plt.title("Percentage of Diabetic Patients")
plt.ylabel('')
plt.savefig('viz2_outcome_pie.png')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Outcome', y='Glucose', data=df, palette="Set2")
plt.title("Glucose Levels: Diabetic vs Non-Diabetic")
plt.savefig('viz3_glucose_box.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Outcome', kde=True, palette="husl")
plt.title("Age Distribution by Diabetes Outcome")
plt.savefig('viz4_age_dist.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, palette="deep", alpha=0.6)
plt.title("Glucose vs BMI Scatter Plot")
plt.savefig('viz5_glucose_bmi_scatter.png')
plt.show()
