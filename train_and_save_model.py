import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold

def get_learning_curve_data(estimator, X, y):
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Calculates learning curve data points based on F1 Score
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv_strategy, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='f1'
    )
    return {
        'train_sizes': train_sizes,
        'train_scores_mean': np.mean(train_scores, axis=1),
        'train_scores_std': np.std(train_scores, axis=1),
        'test_scores_mean': np.mean(test_scores, axis=1),
        'test_scores_std': np.std(test_scores, axis=1)
    }

def main():
    print("Membaca dataset...")
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        print("Error: File 'WA_Fn-UseC_-Telco-Customer-Churn.csv' tidak ditemukan.")
        return

    print("Melakukan Data Preparation...")
    selected_features = [
        'Contract', 'InternetService', 'TotalCharges', 'tenure',
        'PaperlessBilling', 'MultipleLines', 'StreamingMovies', 'Churn'
    ]
    df_clean = df[selected_features].copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

    print("Memisahkan data latih dan data uji...")
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn'].map({'No': 0, 'Yes': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_median = X_train['TotalCharges'].median()
    X_train['TotalCharges'] = X_train['TotalCharges'].fillna(train_median)
    X_test['TotalCharges'] = X_test['TotalCharges'].fillna(train_median)

    numeric_features = ['tenure', 'TotalCharges']
    categorical_features = ['Contract', 'InternetService', 'PaperlessBilling', 'MultipleLines', 'StreamingMovies']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    smote = SMOTE(random_state=42)

    # ========================== XGBOOST ==========================
    print("Membangun dan melatih model XGBoost...")
    xgb_clf = XGBClassifier(
        random_state=42, eval_metric='logloss',
        reg_lambda=1, reg_alpha=0.1, max_depth=3, 
        learning_rate=0.05, gamma=2
    )
    xgb_pipeline = ImbPipeline([('prep', preprocessor), ('smote', smote), ('clf', xgb_clf)])
    xgb_pipeline.fit(X_train, y_train)

    # XGBoost Metrics
    xgb_train_pred = xgb_pipeline.predict(X_train)
    xgb_test_pred = xgb_pipeline.predict(X_test)
    xgb_metrics_train = classification_report(y_train, xgb_train_pred, output_dict=True)
    xgb_metrics_test = classification_report(y_test, xgb_test_pred, output_dict=True)

    print("Menghitung Learning Curve XGBoost...")
    xgb_lc = get_learning_curve_data(xgb_pipeline, X_train, y_train)

    # ========================== RANDOM FOREST ==========================
    print("Membangun dan melatih model Random Forest...")
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_split=50, min_samples_leaf=20,
        random_state=42, class_weight='balanced'
    )
    rf_pipeline = ImbPipeline([('prep', preprocessor), ('smote', smote), ('clf', rf_clf)])
    rf_pipeline.fit(X_train, y_train)

    # Random Forest Metrics
    rf_train_pred = rf_pipeline.predict(X_train)
    rf_test_pred = rf_pipeline.predict(X_test)
    rf_metrics_train = classification_report(y_train, rf_train_pred, output_dict=True)
    rf_metrics_test = classification_report(y_test, rf_test_pred, output_dict=True)

    print("Menghitung Learning Curve Random Forest...")
    rf_lc = get_learning_curve_data(rf_pipeline, X_train, y_train)

    # ========================== EXPORT JOBLIB ==========================
    print("Menyimpan model & evaluasi ke dalam file joblib...")
    model_data = {
        'model': xgb_pipeline, # Keep XGBoost as primary default model for Tab 1
        'rf_model': rf_pipeline,
        'train_median_total_charges': train_median,
        'metrics': {
            'xgboost_train': xgb_metrics_train,
            'xgboost_test': xgb_metrics_test,
            'rf_train': rf_metrics_train,
            'rf_test': rf_metrics_test,
        },
        'learning_curves': {
            'xgboost': xgb_lc,
            'rf': rf_lc
        }
    }
    joblib.dump(model_data, 'churn_model.joblib')
    print("Model berhasil disimpan sebagai 'churn_model.joblib'")

if __name__ == '__main__':
    main()
