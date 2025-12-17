import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

try:
    df = pd.read_csv('diabetes_cleaned.csv')
except FileNotFoundError:
    print("Error: Run step 1 first to generate diabetes_cleaned.csv")
    exit()

X = df.drop('Outcome', axis=1)
y = df['Outcome']


# MODEL 1: SUPERVISED (Random Forest)
print("--- Training Supervised Model (Random Forest) ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Supervised Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Saved supervised model to 'diabetes_model.pkl'")

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('model_supervised_confusion_matrix.png')
print("Saved Confusion Matrix plot.")


# MODEL 2: UNSUPERVISED (K-Means Clustering)
print("\n--- Training Unsupervised Model (K-Means) ---")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

print("Comparison of Mathematical Clusters vs Medical Diagnosis:")
print(pd.crosstab(df['Outcome'], df['Cluster']))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters
df_pca['Actual'] = y

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis')
plt.title('Unsupervised Clustering (No Labels)')

plt.subplot(1, 2, 2)
sns.scatterplot(x='PC1', y='PC2', hue='Actual', data=df_pca, palette='coolwarm')
plt.title('Actual Diagnosis (Ground Truth)')

plt.savefig('model_unsupervised_comparison.png')
print("Saved Unsupervised Comparison plot.")