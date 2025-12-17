import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score

# 1. Load Data
df = pd.read_csv('diabetes_cleaned.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

# 4. Train Logistic Regression (NEW)
# Scale data first for LR and SVM (best practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_full_scaled = scaler.transform(X) # For K-Means later

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled))

# 5. Train SVM (NEW)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))

# 6. Train K-Means (Unsupervised)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_full_scaled)
k_score = silhouette_score(X_full_scaled, clusters)

# Identify which cluster is "Diabetic"
df['Cluster'] = clusters
diabetic_cluster_id = df.groupby('Cluster')['Glucose'].mean().idxmax()

# 7. Save EVERYTHING
data_to_save = {
    'rf_model': rf_model,
    'rf_acc': rf_acc,
    'lr_model': lr_model,  # Added
    'lr_acc': lr_acc,      # Added
    'svm_model': svm_model,# Added
    'svm_acc': svm_acc,    # Added
    'kmeans_model': kmeans,
    'scaler': scaler,
    'k_score': k_score,
    'diabetic_cluster_id': diabetic_cluster_id
}

with open('all_models.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("SUCCESS: Updated 'all_models.pkl' with RF, LR, SVM, and K-Means!")