import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

#Import yang digunakan

#Sumber file
file_path = 'Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv'

#Menampilkan Informasi awal isi Data Set
print("--- Pemeriksaan Info Data Awal ---")
df = pd.read_csv(file_path)
df.info()

# --- DATA CLEANING ---
# Menghapus identifier (customerID)
df_clean = df.drop(columns=['customerID']) 

# Memperbaiki tipe data TotalCharges menjadi numerik
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Imputasi Missing Values dengan Median
imputer = SimpleImputer(strategy='median')
df_clean[['TotalCharges']] = imputer.fit_transform(df_clean[['TotalCharges']])

# --- EXPLORATORY DATA ANALYSIS (EDA) ---
# Distribusi Target Churn
print("\n--- Distribusi Churn (%) ---")
print(df_clean['Churn'].value_counts(normalize=True) * 100)

# Visualisasi: Monthly Charges vs Churn
plt.figure(figsize=(8,5))
sns.kdeplot(data=df_clean, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette='crest')
plt.title('Density Plot: Monthly Charges vs Churn')
plt.show()

# Visualisasi: Churn berdasarkan Layanan Internet
plt.figure(figsize=(8,5))
sns.countplot(data=df_clean, x='InternetService', hue='Churn', palette='Set2')
plt.title('Distribusi Churn berdasarkan Layanan Internet')
plt.show()

# Visualisasi: Matriks Korelasi
plt.figure(figsize=(10,6))
# Membuat korelasi hanya untuk kolom angka
corr_matrix = df_clean.select_dtypes(include=['number']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriks Korelasi Fitur Numerik')
plt.show()

# --- TRANSFORMASI FITUR (ENCODING & SCALING) ---
# Ordinal Encoding pada kolom Contract
contract_order = [['Month-to-month', 'One year', 'Two year']]
ordinal_enc = OrdinalEncoder(categories=contract_order)
df_clean['Contract_Encoded'] = ordinal_enc.fit_transform(df_clean[['Contract']])

# Memisahkan Fitur dan Target
X = df_clean.drop(columns=['Churn', 'Contract'])
y = df_clean['Churn'].map({'No': 0, 'Yes': 1}) # Mengubah target ke angka

# One-Hot Encoding untuk sisa kategori
cat_features = X.select_dtypes(include=['object']).columns
X_enc = pd.get_dummies(X, columns=cat_features, drop_first=True)

# SPLIT DATA (80% TRAIN & 20% TEST)
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=y)

print("--- Distribusi Target SEBELUM SMOTE (Data Latih) ---")
print(y_train.value_counts())

#Scaling data setelah split
scaler = MinMaxScaler()
# fit_transform HANYA pada data latih
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# -- PENERAPAN SMOTE (HANYA PADA DATA TRAINING) 
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("\n--- Distribusi Target SESUDAH SMOTE (Data Latih) ---")
print(y_train_smote.value_counts())

# --- EXPORT HASIL KE FOLDER ---
X_train_smote.to_csv('X_train_ready.csv', index=False)
y_train_smote.to_csv('y_train_ready.csv', index=False)
X_test_scaled.to_csv('X_test_ready.csv', index=False)
y_test.to_csv('y_test_ready.csv', index=False)

print("\nData preprocessing, Splitting, dan SMOTE selesai! Data siap masuk ke algoritma.")


# --- IMPLEMENTASI & EKSPERIMEN 3 MODEL ---
# --- MODEL BASELINE PAPER (LOGISTIC REGRESSION) ---
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_smote, y_train_smote)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]

# --- MODEL EKSPERIMEN  1 (DECISION TREE) ---
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_smote, y_train_smote)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

# --- MODEL EKSPERIMEN  2 (RANDOM FOREST + TUNING) ---
print("Sedang melakukan tuning parameter Random Forest... (Tunggu sebentar)")
param_grid = {
    'max_depth': [10, 20, None],
    'n_estimators': [100, 200],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

# --- MEMBUAT TABEL KOMPARASI PERFORMA ---
def get_metrics(y_true, y_pred, y_prob):
    return [
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        roc_auc_score(y_true, y_prob)
    ]

# Mengumpulkan hasil metrik
metrics_logreg = get_metrics(y_test, y_pred_logreg, y_prob_logreg)
metrics_dt = get_metrics(y_test, y_pred_dt, y_prob_dt)
metrics_rf = get_metrics(y_test, y_pred_rf, y_prob_rf)

print("\n===================================================================")
print("TABEL KOMPARASI PERFORMA")
print("===================================================================")

tabel_komparasi = pd.DataFrame({
    'Algoritma': [
        'Baseline (Logistic Regression - Referensi Paper)',
        'Model 1 (Decision Tree)',
        'Model 2 (Random Forest - Optimasi)'
    ],
    'Precision': [metrics_logreg[0], metrics_dt[0], metrics_rf[0]],
    'Recall': [metrics_logreg[1], metrics_dt[1], metrics_rf[1]],
    'F1-Score': [metrics_logreg[2], metrics_dt[2], metrics_rf[2]],
    'AUC-ROC': [metrics_logreg[3], metrics_dt[3], metrics_rf[3]]
})

# Dibulatkan 2 angka di belakang koma
tabel_komparasi = tabel_komparasi.round(2)
print(tabel_komparasi)