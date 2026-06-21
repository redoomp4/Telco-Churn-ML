import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import altair as alt
import os

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Telco Churn Enterprise Analytics",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS Premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important; }
    
    .stApp p, .stApp span, .stApp label, .stApp li {
        color: #334155 !important; font-size: 14px; font-weight: 500;
    }
    
    .stApp label {
        font-weight: 700 !important; color: #475569 !important; font-size: 12px !important;
        text-transform: uppercase !important; letter-spacing: 0.8px !important;
        margin-bottom: 8px !important; display: block;
    }
    
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #0f172a !important; font-weight: 800 !important; letter-spacing: -0.5px;
    }
    

    
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 40px 30px; border-radius: 20px; text-align: center;
        margin-bottom: 30px; box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .header-container h1 { color: #ffffff !important; font-size: 2.6rem !important; margin-bottom: 10px !important; }
    .header-container p { color: #94a3b8 !important; font-size: 1.1rem !important; }

    .custom-card {
        background-color: #ffffff !important; padding: 30px; border-radius: 16px;
        border: 1px solid #e2e8f0 !important; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.02) !important;
        margin-bottom: 25px;
    }
    .card-title {
        font-size: 1.3rem !important; font-weight: 800 !important; color: #0f172a !important;
        margin-bottom: 20px; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px;
    }

    button[data-baseweb="tab"] { color: #64748b !important; font-weight: 700 !important; font-size: 15px !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #4f46e5 !important; border-bottom: 3px solid #4f46e5 !important; }

    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        color: #ffffff !important; border: none !important; padding: 15px 30px !important;
        font-weight: 700 !important; border-radius: 12px !important;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3) !important; width: 100% !important;
    }
    .stButton>button p { color: #ffffff !important; font-weight: 700 !important; }

    .churn-meter-bg {
        background-color: #e2e8f0; border-radius: 10px; height: 20px; width: 100%;
        position: relative; overflow: hidden; margin-top: 12px; margin-bottom: 12px; border: 1px solid #cbd5e1;
    }
    .churn-meter-fill { height: 100%; border-radius: 10px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }

    .result-box { padding: 26px; border-radius: 16px; margin-top: 15px; }
    .result-high { background-color: #fef2f2 !important; border: 1px solid #fee2e2 !important; }
    .result-high h3 { color: #dc2626 !important; }
    .result-medium { background-color: #fffbeb !important; border: 1px solid #fef3c7 !important; }
    .result-medium h3 { color: #d97706 !important; }
    .result-low { background-color: #f0fdf4 !important; border: 1px solid #dcfce7 !important; }
    .result-low h3 { color: #16a34a !important; }
    
    .metric-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px; margin-bottom: 25px; }
    .metric-box {
        background: #ffffff !important; padding: 20px 10px; border-radius: 16px; text-align: center;
        border: 1px solid #e2e8f0 !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02) !important;
    }
    .metric-value { font-size: 1.8rem !important; font-weight: 800 !important; color: #4f46e5 !important; margin-bottom: 4px; }
    .metric-label { font-size: 0.75rem !important; font-weight: 800 !important; color: #64748b !important; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# Helper Functions untuk Render UI
def show_eda_dashboard(df, target_col):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 Exploratory Data Analysis (EDA) - Telco Customer Churn</div>', unsafe_allow_html=True)
    
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    
    fig = plt.figure(figsize=(18, 10), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    palette = {"No": "#3498db", "Yes": "#e74c3c"}
    
    # 1. Target Class Distribution
    ax1 = plt.subplot(2, 3, 1)
    churn_counts = df[target_col].value_counts()
    labels = churn_counts.index
    sizes = churn_counts.values
    colors = [palette.get(l, '#95a5a6') for l in labels]
    explode = [0.1 if l == 'Yes' else 0 for l in labels]
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'color': "white", 'weight': 'bold'})
    ax1.set_title(f'Target Class Distribution ({target_col})', fontweight='bold', pad=15)
    
    # 2. Tenure vs Churn
    ax2 = plt.subplot(2, 3, 2)
    if 'tenure' in df.columns:
        sns.kdeplot(data=df, x='tenure', hue=target_col, fill=True, palette=palette, ax=ax2, alpha=0.3, common_norm=False)
        ax2.set_title(f'Tenure vs {target_col} Distribution', fontweight='bold')
        ax2.set_xlabel('Tenure (Months)')
    
    # 3. Total Charges vs Churn
    ax3 = plt.subplot(2, 3, 3)
    if 'TotalCharges' in df.columns:
        sns.kdeplot(data=df.dropna(subset=['TotalCharges']), x='TotalCharges', hue=target_col, fill=True, palette=palette, ax=ax3, alpha=0.3, common_norm=False)
        ax3.set_title(f'Total Charges vs {target_col} Distribution', fontweight='bold')
        ax3.set_xlabel('Total Charges')
        
    # 4. Contract vs Churn
    ax4 = plt.subplot(2, 3, 4)
    if 'Contract' in df.columns:
        sns.countplot(data=df, x='Contract', hue=target_col, palette=palette, ax=ax4)
        ax4.set_title(f'Contract vs {target_col}', fontweight='bold')
        ax4.set_xlabel('')
        
    # 5. Internet Service vs Churn
    ax5 = plt.subplot(2, 3, 5)
    if 'InternetService' in df.columns:
        sns.countplot(data=df, x='InternetService', hue=target_col, palette=palette, ax=ax5)
        ax5.set_title(f'Internet Service vs {target_col}', fontweight='bold')
        ax5.set_xlabel('')
        
    # 6. Paperless Billing vs Churn
    ax6 = plt.subplot(2, 3, 6)
    if 'PaperlessBilling' in df.columns:
        sns.countplot(data=df, x='PaperlessBilling', hue=target_col, palette=palette, ax=ax6)
        ax6.set_title(f'Paperless Billing vs {target_col}', fontweight='bold')
        ax6.set_xlabel('')
        
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_comparison(metrics, learning_curves):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚖️ Komparasi Metrik & Kurva Pembelajaran AI</div>', unsafe_allow_html=True)
    
    if metrics and learning_curves:
        col_m1, col_m2 = st.columns(2)
        
        def display_classification_metrics(title, train_met, test_met):
            st.markdown(f"### {title}")
            
            # --- TEST MODEL FIRST to highlight Recall ---
            st.markdown("#### 🔸 PERFORMANCE METRICS: TEST MODEL")
            st.info(f"🎯 **Test Recall (Deteksi Churn):** `{test_met['1']['recall']*100:.2f}%`")
            st.markdown(f"**Test Accuracy:** `{test_met['accuracy']*100:.2f}%`")
            df_test = pd.DataFrame({
                'Class': ['No Churn', 'Churn'],
                'Precision': [test_met['0']['precision'], test_met['1']['precision']],
                'Recall': [test_met['0']['recall'], test_met['1']['recall']],
                'F1-Score': [test_met['0']['f1-score'], test_met['1']['f1-score']],
            })
            st.dataframe(df_test.style.format({'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1-Score': '{:.2%}'}), use_container_width=True)

            # --- TRAIN MODEL ---
            st.markdown("#### 🔹 PERFORMANCE METRICS: TRAIN MODEL")
            st.info(f"🎯 **Train Recall (Deteksi Churn):** `{train_met['1']['recall']*100:.2f}%`")
            st.markdown(f"**Train Accuracy:** `{train_met['accuracy']*100:.2f}%`")
            df_train = pd.DataFrame({
                'Class': ['No Churn', 'Churn'],
                'Precision': [train_met['0']['precision'], train_met['1']['precision']],
                'Recall': [train_met['0']['recall'], train_met['1']['recall']],
                'F1-Score': [train_met['0']['f1-score'], train_met['1']['f1-score']],
            })
            st.dataframe(df_train.style.format({'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1-Score': '{:.2%}'}), use_container_width=True)

        with col_m1:
            display_classification_metrics("XGBoost Classifier", metrics['xgboost_train'], metrics['xgboost_test'])
        with col_m2:
            display_classification_metrics("Random Forest Classifier", metrics['rf_train'], metrics['rf_test'])

        st.markdown("<hr style='margin:30px 0;'>", unsafe_allow_html=True)
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Validation Diagnostics: Tuned Learning Curves', fontsize=16, fontweight='bold')
        
        plot_configs = [
            ('XGBoost', learning_curves['xgboost'], axes[0]),
            ('Random Forest', learning_curves['rf'], axes[1])
        ]
        
        for name, lc_data, ax in plot_configs:
            train_sizes = lc_data['train_sizes']
            train_mean = lc_data['train_scores_mean']
            val_mean = lc_data['test_scores_mean']
            
            gap = train_mean[-1] - val_mean[-1]
            status = "EXCELLENT" if gap < 0.03 else "GOOD FIT" if gap < 0.06 else "OVERFITTING"
            
            ax.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
            ax.plot(train_sizes, val_mean, 'o-', color="g", label="Validation Score")
            ax.set_title(f"{name} ({status} | Gap: {gap:.4f})")
            ax.set_xlabel("Training Set Size")
            ax.set_ylabel("F1 Score")
            ax.legend(loc='lower right')
            
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Metrik evaluasi atau data learning curve tidak ditemukan di file joblib.")
    st.markdown('</div>', unsafe_allow_html=True)


# Load Data Evaluasi ML
@st.cache_resource
def load_all_data():
    try:
        return joblib.load('churn_model.joblib')
    except Exception:
        return None

data = load_all_data()

st.markdown("""
<div class="header-container" style="display: flex; align-items: center; justify-content: center; gap: 25px; flex-wrap: wrap;">
    <div style="background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%); padding: 18px; border-radius: 18px; box-shadow: 0 10px 20px -5px rgba(79, 70, 229, 0.4);">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 12h4l2-9 5 18 3-10h4"/>
            <circle cx="12" cy="12" r="10" stroke-opacity="0.3"/>
        </svg>
    </div>
    <div style="text-align: left;">
        <h1 style="margin-top: 0; margin-bottom: 8px; font-size: 2.8rem !important;">Telco Churn Enterprise Portal</h1>
        <p style="margin: 0; max-width: 650px; font-size: 1.15rem !important;">Predict customer churn risk levels instantly, extract insights, and evaluate AI performance.</p>
    </div>
</div>
""", unsafe_allow_html=True)

if data is None:
    st.error("⚠️ File model `churn_model.joblib` tidak ditemukan! Silakan jalankan `train_and_save_model.py` terlebih dahulu di terminal untuk melatih model.")
else:
    model = data.get('model')
    rf_model = data.get('rf_model')
    train_median = data.get('train_median_total_charges')
    metrics = data.get('metrics', {})
    learning_curves = data.get('learning_curves', {})

    def get_feature_importances(model_pipeline):
        try:
            prep = model_pipeline.named_steps['prep']
            clf = model_pipeline.named_steps['clf']
            num_cols = ['tenure', 'TotalCharges']
            cat_cols = list(prep.named_transformers_['cat'].get_feature_names_out([
                'Contract', 'InternetService', 'PaperlessBilling', 'MultipleLines', 'StreamingMovies'
            ]))
            all_cols = num_cols + cat_cols
            importances = clf.feature_importances_
            
            base_features = {
                'tenure': 0.0,
                'TotalCharges': 0.0,
                'Contract': 0.0,
                'InternetService': 0.0,
                'PaperlessBilling': 0.0,
                'MultipleLines': 0.0,
                'StreamingMovies': 0.0
            }
            
            for col, imp in zip(all_cols, importances):
                for base in base_features.keys():
                    if col == base or col.startswith(base + '_'):
                        base_features[base] += imp
                        break
            
            df_imp = pd.DataFrame({
                'Fitur': list(base_features.keys()),
                'Tingkat Pengaruh (%)': np.array(list(base_features.values())) * 100
            }).sort_values(by='Tingkat Pengaruh (%)', ascending=True)
            return df_imp
        except Exception:
            return None

    df_importance = get_feature_importances(model)

    tab1, tab2 = st.tabs(["🔮 Prediksi Satu Pelanggan", "🗂️ Uji Dataset Massal"])

    # ==================== TAB 1: PREDIKSI SATU PELANGGAN ====================
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Parameter Pelanggan</div>', unsafe_allow_html=True)
        col_input1, col_input2, col_input3 = st.columns(3)
        
        with col_input1:
            tenure = st.number_input("Lama Berlangganan (Bulan) / Tenure", min_value=0, max_value=120, value=12)
            total_charges = st.number_input("Total Biaya ($)", min_value=0.0, value=150.0)
        with col_input2:
            contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
            paperless_billing = st.selectbox("Tagihan Digital (Paperless)", ["Yes", "No"])
        with col_input3:
            multiple_lines = st.selectbox("Lebih dari Satu Jalur Telepon", ["No", "Yes", "No phone service"])
            streaming_movies = st.selectbox("Layanan Streaming Film", ["No", "Yes", "No internet service"])
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Jalankan Prediksi Risiko", key="btn_single"):
            input_df = pd.DataFrame({
                'Contract': [contract], 'InternetService': [internet_service],
                'TotalCharges': [total_charges], 'tenure': [tenure],
                'PaperlessBilling': [paperless_billing], 'MultipleLines': [multiple_lines],
                'StreamingMovies': [streaming_movies]
            })
            if total_charges == 0.0: input_df['TotalCharges'] = train_median
            
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            
            col_res1, col_res2 = st.columns([1.1, 0.9])
            with col_res1:
                st.markdown('<div class="custom-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Status Risiko Churn</div>', unsafe_allow_html=True)
                
                if prob >= 0.60:
                    bar_color = "linear-gradient(90deg, #ef4444 0%, #b91c1c 100%)"
                    status_title, status_class = "RISIKO TINGGI 🚨", "result-high"
                elif prob >= 0.30:
                    bar_color = "linear-gradient(90deg, #fbbf24 0%, #d97706 100%)"
                    status_title, status_class = "RISIKO SEDANG ⚠️", "result-medium"
                else:
                    bar_color = "linear-gradient(90deg, #34d399 0%, #059669 100%)"
                    status_title, status_class = "RISIKO RENDAH ✅", "result-low"
                
                st.markdown(f"""
                <div class="result-box {status_class}">
                    <div style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; color:#475569;">Klasifikasi Model</div>
                    <h3 style="margin: 5px 0 10px 0 !important; font-size: 1.5rem !important;">{status_title}</h3>
                    <div style="font-size: 0.95rem; font-weight: 700; color:#0f172a;">Kecenderungan Churn: <b>{prob*100:.1f}%</b></div>
                    <div class="churn-meter-bg"><div class="churn-meter-fill" style="width: {prob*100:.1f}%; background: {bar_color};"></div></div>
                </div>
                """, unsafe_allow_html=True)
                
                reasons_pos, reasons_neg = [], []
                if contract == 'Month-to-month': reasons_pos.append("Tipe Kontrak Bulanan: Tidak ada komitmen jangka panjang.")
                elif contract in ['One year', 'Two year']: reasons_neg.append(f"Kontrak {contract}: Komitmen legal menekan potensi churn.")
                if internet_service == 'Fiber optic': reasons_pos.append("Layanan Fiber Optic: Biaya tinggi memicu pergantian.")
                elif internet_service == 'No': reasons_neg.append("Tanpa Layanan Internet: Sangat stabil secara historis.")
                if tenure <= 12: reasons_pos.append(f"Masa Berlangganan Pendek ({tenure} bulan): Hubungan rentan.")
                elif tenure >= 36: reasons_neg.append(f"Loyalitas Tenure Lama ({tenure} bulan): Loyalitas terbukti stabil.")
                if paperless_billing == 'Yes': reasons_pos.append("Tagihan Digital: Meningkatkan sensitivitas harga.")

                if reasons_pos:
                    st.markdown("<div style='margin-top: 15px;'><b style='color:#dc2626; font-size: 12px; text-transform: uppercase;'>Faktor Pendorong Risiko Churn:</b></div>", unsafe_allow_html=True)
                    for r in reasons_pos: st.markdown(f"<div style='background: #fef2f2; border: 1px solid #fee2e2; padding: 10px; border-radius: 8px; margin-top: 5px; font-size: 0.85rem; color: #991b1b; font-weight: 600;'>📌 {r}</div>", unsafe_allow_html=True)
                if reasons_neg:
                    st.markdown("<div style='margin-top: 15px;'><b style='color:#16a34a; font-size: 12px; text-transform: uppercase;'>Faktor Pendukung Retensi:</b></div>", unsafe_allow_html=True)
                    for r in reasons_neg: st.markdown(f"<div style='background: #f0fdf4; border: 1px solid #dcfce7; padding: 10px; border-radius: 8px; margin-top: 5px; font-size: 0.85rem; color: #166534; font-weight: 600;'>🛡️ {r}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_res2:
                st.markdown('<div class="custom-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Tingkat Pengaruh Fitur Keputusan</div>', unsafe_allow_html=True)
                if df_importance is not None:
                    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
                    fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#ffffff')
                    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
                    bars = ax.barh(df_importance['Fitur'], df_importance['Tingkat Pengaruh (%)'], color='#4f46e5', height=0.6)
                    for bar in bars:
                        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}%', 
                                ha='left', va='center', fontsize=8, color='#0f172a', fontweight='bold')
                    ax.tick_params(axis='both', colors='#334155', labelsize=8)
                    plt.xlabel("Pengaruh (%)", fontsize=9, fontweight='bold')
                    st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)
            st.markdown("## Analitik Ekstra & Performa AI")
            
            if os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv"):
                base_df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
                base_df['TotalCharges'] = pd.to_numeric(base_df['TotalCharges'], errors='coerce')
                st.info("💡 **Konteks Analitik:** Menampilkan EDA berdasarkan keseluruhan dataset historis sebagai perbandingan untuk pelanggan tunggal ini.")
                show_eda_dashboard(base_df, "Churn")
            
            show_model_comparison(metrics, learning_curves)

    # ==================== TAB 2: UJI DATASET MASSAL ====================
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Sumber Data Analisis</div>', unsafe_allow_html=True)
        source_tab1, source_tab2 = st.tabs(["📊 Dataset Bawaan", "📁 Unggah File Baru"])
        test_df = None
        
        with source_tab1:
            if os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv"):
                st.info("ℹ️ Menggunakan dataset bawaan sistem.")
                df_builtin = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
                
                st.write("**Jumlah baris yang dianalisis:**")
                col_s1, col_n1 = st.columns([3, 1])
                with col_s1:
                    size_1_slide = st.slider("Geser Slider", 5, len(df_builtin), min(len(df_builtin), 200), step=5, key="slide_builtin", label_visibility="collapsed")
                with col_n1:
                    size_1 = st.number_input("Tombol +/-", min_value=5, max_value=len(df_builtin), value=size_1_slide, step=5, key="num_builtin", label_visibility="collapsed")
                
                if st.button("Mulai Prediksi Massal", key="btn_builtin"):
                    test_df = df_builtin.head(size_1).copy()
                    
        with source_tab2:
            uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"], key="uploader_csv")
            if uploaded_file is not None:
                df_upload = pd.read_csv(uploaded_file)
                
                st.write("**Jumlah baris yang dianalisis:**")
                col_s2, col_n2 = st.columns([3, 1])
                with col_s2:
                    size_2_slide = st.slider("Geser Slider", 5, len(df_upload), min(len(df_upload), 200), step=5, key="slide_upload", label_visibility="collapsed")
                with col_n2:
                    size_2 = st.number_input("Tombol +/-", min_value=5, max_value=len(df_upload), value=size_2_slide, step=5, key="num_upload", label_visibility="collapsed")
                
                if st.button("Mulai Prediksi Massal", key="btn_upload"):
                    test_df = df_upload.head(size_2).copy()
                    
        if test_df is not None:
            sampled_df = test_df
            req_cols = ['Contract', 'InternetService', 'TotalCharges', 'tenure', 'PaperlessBilling', 'MultipleLines', 'StreamingMovies']
            if not all(c in sampled_df.columns for c in req_cols):
                st.error("Dataset tidak memiliki kolom wajib.")
            else:
                with st.spinner("Memproses..."):
                    prep_df = sampled_df[req_cols].copy()
                    prep_df['TotalCharges'] = pd.to_numeric(prep_df['TotalCharges'], errors='coerce').fillna(train_median)
                    
                    preds, probs = model.predict(prep_df), model.predict_proba(prep_df)[:, 1]
                    preds_rf, probs_rf = rf_model.predict(prep_df), rf_model.predict_proba(prep_df)[:, 1]
                    
                    sampled_df['Prediksi_XGBoost'] = ['Yes' if p == 1 else 'No' for p in preds]
                    sampled_df['Probabilitas_XGBoost (%)'] = np.round(probs * 100, 2)
                    sampled_df['Prediksi_RandomForest'] = ['Yes' if p == 1 else 'No' for p in preds_rf]
                    sampled_df['Probabilitas_RF (%)'] = np.round(probs_rf * 100, 2)
                    
                if 'Churn' in sampled_df.columns:
                    y_actual = sampled_df['Churn'].map({'Yes': 1, 'No': 0})
                    from sklearn.metrics import classification_report
                    
                    def render_html_report(y_true, y_pred, title):
                        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                        html = f'''
                        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 20px;">
                            <h4 style="color: #0f172a; margin-top: 0; padding-bottom: 10px; border-bottom: 1px solid #f1f5f9;">{title}</h4>
                            <table style="width: 100%; border-collapse: collapse; text-align: right; font-size: 0.9rem;">
                                <tr style="color: #64748b; border-bottom: 1px solid #e2e8f0;">
                                    <th style="text-align: left; padding: 8px;">Class</th><th style="padding: 8px;">Precision</th><th style="padding: 8px;">Recall</th><th style="padding: 8px;">F1-Score</th><th style="padding: 8px;">Support</th>
                                </tr>
                        '''
                        def get_row(k, name, bold=False):
                            if k not in rep: return ""
                            p, r, f, s = rep[k].get('precision',''), rep[k].get('recall',''), rep[k].get('f1-score',''), rep[k].get('support','')
                            if isinstance(p, float): p, r, f, s = f"{p:.2f}", f"{r:.2f}", f"{f:.2f}", str(int(s))
                            w = "800" if bold else "500"
                            c = "#0f172a" if bold else "#334155"
                            bg = "#f8fafc" if bold else "transparent"
                            return f'<tr style="background: {bg}; border-bottom: 1px solid #f1f5f9; color: {c}; font-weight: {w};"><td style="text-align: left; padding: 8px;">{name}</td><td style="padding: 8px;">{p}</td><td style="padding: 8px;">{r}</td><td style="padding: 8px;">{f}</td><td style="padding: 8px;">{s}</td></tr>'
                        
                        html += get_row('0', 'No Churn (0)')
                        html += get_row('1', 'Churn (1)')
                        if 'accuracy' in rep:
                            sup = rep['macro avg']['support'] if 'macro avg' in rep else 0
                            html += f'<tr style="border-top: 2px solid #e2e8f0; font-weight: 800; background: #f8fafc; color: #0f172a;"><td style="text-align: left; padding: 8px;">Accuracy</td><td></td><td></td><td style="padding: 8px;">{rep["accuracy"]:.2f}</td><td style="padding: 8px;">{int(sup)}</td></tr>'

                        html += get_row('macro avg', 'macro avg', True)
                        html += get_row('weighted avg', 'weighted avg', True)
                        html += '</table></div>'
                        return html
                        
                    st.markdown("<h3 style='margin-top:20px; font-weight:800;'>📊 Metrik Evaluasi Model (Terhadap Data Uji Ini)</h3>", unsafe_allow_html=True)
                    col_met1, col_met2 = st.columns(2)
                    with col_met1:
                        st.markdown(render_html_report(y_actual, preds, "🚀 XGBoost Classifier"), unsafe_allow_html=True)
                    with col_met2:
                        st.markdown(render_html_report(y_actual, preds_rf, "🌲 Random Forest Classifier"), unsafe_allow_html=True)
                        
                st.dataframe(sampled_df, use_container_width=True)
                
                st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)
                st.markdown("## Exploratory Data Analysis (EDA)")
                
                sampled_df['TotalCharges'] = pd.to_numeric(sampled_df['TotalCharges'], errors='coerce')
                
                if 'Churn' in sampled_df.columns:
                    st.info("💡 **Konteks Analitik:** Grafik Exploratory Data Analysis (EDA) di bawah ini mengacu pada label **Aktual (Ground Truth)** dari dataset yang Anda unggah.")
                    show_eda_dashboard(sampled_df, "Churn")
                else:
                    st.info("💡 **Konteks Analitik:** Karena dataset ini tidak memiliki kolom target 'Churn', Grafik (EDA) di bawah ini otomatis dibuat berdasarkan hasil **Prediksi XGBoost Classifier**.")
                    show_eda_dashboard(sampled_df, "Prediksi_XGBoost")
                
                show_model_comparison(metrics, learning_curves)

        st.markdown('</div>', unsafe_allow_html=True)
