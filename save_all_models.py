import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score

df = pd.read_csv('diabetes_cleaned.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred)
print(f"RF Accuracy: {rf_acc}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
k_score = silhouette_score(X_scaled, clusters)
print(f"K-Means Silhouette Score: {k_score}")

df['Cluster'] = clusters
cluster_glucose = df.groupby('Cluster')['Glucose'].mean()
diabetic_cluster_id = cluster_glucose.idxmax()
print(f"The 'Diabetic-like' cluster is Cluster #{diabetic_cluster_id}")

data_to_save = {
    'rf_model': rf_model,
    'rf_acc': rf_acc,
    'kmeans_model': kmeans,
    'scaler': scaler,
    'k_score': k_score,
    'diabetic_cluster_id': diabetic_cluster_id
}

with open('all_models.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("All models and scores saved to 'all_models.pkl'")