import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from scipy.stats import chi2_contingency, f_oneway

# --- Core Risk Score Calculation Logic (remains the same) ---
def calculate_risk_score_df(df_input, grace_period_days, weights):
    # ... (No changes here, same as previous version) ...
    if df_input.empty: return None
    df = df_input.copy()
    required_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro', 'fechaTRansaccion', 'credito', 'reglaCobranza', 'cuotaCubierta', 'cuotaEsperada', 'saldoCapitalActual', 'totalDesembolso', 'cobranzaTrans', 'categoriaProductoCrediticio']
    missing_cols = [col for col in required_cols if col not in df.columns];
    if missing_cols: st.error(f"Sheet 'HistoricoPagoCuotas' missing columns for risk scoring: {', '.join(missing_cols)}"); return None
    date_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro', 'fechaTRansaccion']
    for col in date_cols:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
    df['credito'] = df['credito'].astype(str)
    grace_period = pd.Timedelta(days=grace_period_days)
    mask_valid_dates_late_payment = df['fechaPagoRecibido'].notna() & df['fechaEsperadaPago'].notna()
    df['late_payment'] = 0
    df.loc[mask_valid_dates_late_payment, 'late_payment'] = (df.loc[mask_valid_dates_late_payment, 'fechaPagoRecibido'] > (df.loc[mask_valid_dates_late_payment, 'fechaEsperadaPago'] + grace_period)).astype(int)
    late_payment_counts = df.groupby('credito')['late_payment'].sum()
    total_payments = df.groupby('credito')['fechaEsperadaPago'].count()
    late_payment_ratio = (late_payment_counts / total_payments.replace(0, np.nan)).fillna(0)
    df['cuotaCubierta'] = df['cuotaCubierta'].fillna(0)
    total_payment_made = df.groupby('credito')['cuotaCubierta'].sum()
    total_payment_expected = df.groupby('credito')['cuotaEsperada'].sum()
    payment_coverage_ratio = (total_payment_made / total_payment_expected.replace(0, np.nan)).fillna(1).replace([np.inf, -np.inf], 1)
    df['saldoCapitalActual'] = pd.to_numeric(df['saldoCapitalActual'], errors='coerce')
    df['totalDesembolso'] = pd.to_numeric(df['totalDesembolso'], errors='coerce')
    last_saldo = df.groupby('credito')['saldoCapitalActual'].last()
    first_desembolso = df.groupby('credito')['totalDesembolso'].first()
    outstanding_balance_ratio = (last_saldo / first_desembolso.replace(0, np.nan)).fillna(0)
    df['cobranzaTrans'] = pd.to_numeric(df['cobranzaTrans'], errors='coerce').fillna(0)
    df['collection_activity'] = (df['cobranzaTrans'] > 0).astype(int)
    collection_activity_count = df.groupby('credito')['collection_activity'].sum()
    creditos_unicos = pd.DataFrame(df['credito'].unique(), columns=['credito'])
    creditos_unicos = creditos_unicos.merge(late_payment_ratio.to_frame(name='late_payment_ratio'), left_on='credito', right_index=True, how='left').fillna({'late_payment_ratio': 0})
    creditos_unicos = creditos_unicos.merge(payment_coverage_ratio.to_frame(name='payment_coverage_ratio'), left_on='credito', right_index=True, how='left').fillna({'payment_coverage_ratio': 1})
    creditos_unicos = creditos_unicos.merge(outstanding_balance_ratio.to_frame(name='outstanding_balance_ratio'), left_on='credito', right_index=True, how='left').fillna({'outstanding_balance_ratio': 0})
    creditos_unicos = creditos_unicos.merge(collection_activity_count.to_frame(name='collection_activity_count'), left_on='credito', right_index=True, how='left').fillna({'collection_activity_count': 0})
    for col_name in ['late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']:
        min_val, max_val = creditos_unicos[col_name].min(), creditos_unicos[col_name].max()
        scaled_col_name = f'{col_name}_scaled'
        creditos_unicos[scaled_col_name] = 0.0 if max_val == min_val else (creditos_unicos[col_name] - min_val) / (max_val - min_val)
        creditos_unicos[scaled_col_name] = creditos_unicos[scaled_col_name].fillna(0)
    creditos_unicos['risk_score'] = (weights['late_payment_ratio'] * creditos_unicos['late_payment_ratio_scaled'] + weights['payment_coverage_ratio'] * (1 - creditos_unicos['payment_coverage_ratio_scaled']) + weights['outstanding_balance_ratio'] * creditos_unicos['outstanding_balance_ratio_scaled'] + weights['collection_activity_count'] * creditos_unicos['collection_activity_count_scaled'])
    return creditos_unicos[['credito', 'risk_score']]


# --- Utility Function for Outlier Detection (remains the same) ---
def get_outliers_iqr(df, column_name):
    # ... (No changes here, same as previous version) ...
    if df is None or df.empty or column_name not in df.columns or df[column_name].isnull().all(): return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
    Q1, Q3 = df[column_name].quantile(0.25), df[column_name].quantile(0.75); IQR = Q3 - Q1
    if IQR == 0: lower_bound, upper_bound = Q1, Q3
    else: lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[df[column_name] < lower_bound], df[df[column_name] > upper_bound], lower_bound, upper_bound

# --- Helper function for Cramer's V (remains the same) ---
def cramers_v(confusion_matrix):
    # ... (No changes here, same as previous version) ...
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if r == 1 or k == 1 or n == 0: return 0 # Handle cases where Cramer's V is undefined or 0
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if rcorr < 1 or kcorr < 1 : return 0 # Denominator would be zero or negative
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# --- Data Preparation for Feature Importance Tab (remains the same logic) ---
@st.cache_data
def prepare_feature_importance_data(risk_df, listado_df, id_col_listado, target_col_risk, selected_features_listado, bin_target, num_bins=3):
    # ... (No changes here, same as previous version) ...
    if risk_df is None or risk_df.empty or listado_df is None or listado_df.empty:
        return None, "Risk scores or customer data not available."
    if id_col_listado not in listado_df.columns:
        return None, f"ID column '{id_col_listado}' not found in customer data."
    if target_col_risk not in risk_df.columns:
        return None, f"Target column '{target_col_risk}' not found in risk scores."
    listado_df_copy = listado_df.copy(); risk_df_copy = risk_df.copy()
    listado_df_copy[id_col_listado] = listado_df_copy[id_col_listado].astype(str)
    risk_df_copy['credito'] = risk_df_copy['credito'].astype(str)
    merged_df = pd.merge(listado_df_copy, risk_df_copy[['credito', target_col_risk]], left_on=id_col_listado, right_on='credito', how='inner')
    if merged_df.empty: return None, "No matching records found after merging."
    features_to_analyze = [f for f in selected_features_listado if f in merged_df.columns and f != target_col_risk and f != id_col_listado and f != 'credito']
    if not features_to_analyze: return None, "No valid features selected/found."
    analysis_df = merged_df[features_to_analyze + [target_col_risk]].copy()
    target_name = target_col_risk
    if bin_target:
        if analysis_df[target_col_risk].nunique() <= 1: return None, f"Target '{target_col_risk}' has <=1 unique value, cannot bin."
        try:
            discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile', subsample=None)
            analysis_df[target_col_risk + '_binned'] = discretizer.fit_transform(analysis_df[[target_col_risk]])
            analysis_df[target_col_risk + '_binned'] = analysis_df[target_col_risk + '_binned'].astype(int).astype(str)
            target_name = target_col_risk + '_binned'
        except ValueError as e: return None, f"Error binning target '{target_col_risk}': {e}."
    processed_features_df = pd.DataFrame(index=analysis_df.index); original_dtypes = {}
    for feature in features_to_analyze:
        original_dtypes[feature] = analysis_df[feature].dtype
        if pd.api.types.is_numeric_dtype(analysis_df[feature]):
            processed_features_df[feature] = analysis_df[feature].fillna(analysis_df[feature].median())
        elif pd.api.types.is_object_dtype(analysis_df[feature]) or pd.api.types.is_categorical_dtype(analysis_df[feature]):
            processed_features_df[feature] = analysis_df[feature].fillna(analysis_df[feature].mode().iloc[0] if not analysis_df[feature].mode().empty else "Unknown")
        else: processed_features_df[feature] = analysis_df[feature].astype(str).fillna("Unknown")
    return processed_features_df, analysis_df[target_name], features_to_analyze, original_dtypes, target_name, None


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🏍️ Motorcycle Loan Risk & Data Insights App")

# --- Sidebar (Inputs) ---
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
default_grace_period = 5
grace_period_input = st.sidebar.number_input("Grace Period (days)", min_value=0, max_value=30, value=default_grace_period, step=1)
st.sidebar.subheader("Indicator Weights")
w_late_payment = st.sidebar.slider("Late Payment Ratio Weight", 0.0, 1.0, 0.40, 0.01)
w_payment_coverage = st.sidebar.slider("Payment Coverage Ratio Weight", 0.0, 1.0, 0.40, 0.01)
w_outstanding_balance = st.sidebar.slider("Outstanding Balance Ratio Weight", 0.0, 1.0, 0.20, 0.01)
w_collection_activity = st.sidebar.slider("Collection Activity Count Weight", 0.0, 1.0, 0.00, 0.01)
user_weights = {'late_payment_ratio': w_late_payment, 'payment_coverage_ratio': w_payment_coverage, 'outstanding_balance_ratio': w_outstanding_balance, 'collection_activity_count': w_collection_activity}

# --- Data Loading and Initial Processing (remains the same logic) ---
risk_scores_df, listado_creditos_df = None, None
processed_data_info, historico_pago_cuotas_loaded, listado_creditos_loaded = "", False, False
numero_credito_col_name = "numeroCredito"

if uploaded_file:
    # ... (file loading logic - condensed for brevity, no changes here) ...
    st.sidebar.info(f"Processing: {uploaded_file.name}")
    try:
        hpc_df = pd.read_excel(uploaded_file, sheet_name="HistoricoPagoCuotas")
        processed_data_info += "✅ 'HistoricoPagoCuotas' loaded.\n"
        historico_pago_cuotas_loaded = True
        if 'categoriaProductoCrediticio' in hpc_df.columns:
            moto_df = hpc_df[hpc_df["categoriaProductoCrediticio"] == "MOTOS"].copy()
            if not moto_df.empty:
                processed_data_info += f"Found {len(moto_df)} 'MOTOS' records.\n"
                with st.spinner("Calculating risk scores..."): risk_scores_df = calculate_risk_score_df(moto_df, grace_period_input, user_weights)
                if risk_scores_df is not None and not risk_scores_df.empty: processed_data_info += f"✅ Risk scores for {len(risk_scores_df)} credits.\n"
                else: processed_data_info += "⚠️ Risk score calculation issues.\n"
            else: processed_data_info += "⚠️ No 'MOTOS' data in 'HistoricoPagoCuotas'.\n"
        else: processed_data_info += "❌ 'categoriaProductoCrediticio' missing.\n"; historico_pago_cuotas_loaded = False
    except Exception as e: processed_data_info += f"❌ Error 'HistoricoPagoCuotas': {e}\n"; historico_pago_cuotas_loaded = False

    try:
        lc_df_temp = pd.read_excel(uploaded_file, sheet_name="ListadoCreditos")
        processed_data_info += f"✅ 'ListadoCreditos' loaded ({lc_df_temp.shape[0]}r, {lc_df_temp.shape[1]}c).\n"
        if numero_credito_col_name in lc_df_temp.columns:
            lc_df_temp[numero_credito_col_name] = lc_df_temp[numero_credito_col_name].astype(str)
            listado_creditos_df = lc_df_temp
            listado_creditos_loaded = True
            processed_data_info += f"'{numero_credito_col_name}' cast to string.\n"
        else:
            processed_data_info += f"⚠️ '{numero_credito_col_name}' missing. Outlier/Feature insights affected.\n"
            listado_creditos_df = lc_df_temp; listado_creditos_loaded = True
    except Exception as e: processed_data_info += f"❌ Error 'ListadoCreditos': {e}\n"; listado_creditos_loaded = False
    st.sidebar.text_area("File Processing Log", processed_data_info, height=200)
else:
    st.info("☝️ Upload an Excel file to begin.")

# --- Tabs ---
tab_titles = ["📊 Risk Scores", "📈 Risk EDA", "🕵️ Outlier Analysis", "📋 Customer Data Quality", "🔍 Pre-Loan Feature Insights"]
tabs = st.tabs(tab_titles)

with tabs[0]: # Risk Scores
    # ... (No changes here, same as previous version) ...
    st.header(tab_titles[0])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Calculated Risk Scores (per Credit)")
        st.dataframe(risk_scores_df.style.format({"risk_score": "{:.4f}"}), height=500, use_container_width=True)
        output = io.BytesIO(); risk_scores_df.to_excel(pd.ExcelWriter(output, engine='xlsxwriter'), index=False, sheet_name='RiskScores'); excel_data = output.getvalue()
        st.download_button(label="📥 Download Risk Scores", data=excel_data, file_name=f"risk_scores_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.ms-excel")
    elif uploaded_file and historico_pago_cuotas_loaded: st.warning("Risk scores not calculated. Check log.")
    elif uploaded_file and not historico_pago_cuotas_loaded: st.error("'HistoricoPagoCuotas' failed to load.")
    else: st.write("Upload file for results.")

with tabs[1]: # Risk EDA
    # ... (No changes here, same as previous version) ...
    st.header(tab_titles[1])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Risk Score Distribution"); st.dataframe(risk_scores_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        col1, col2 = st.columns(2)
        with col1: fig, ax = plt.subplots(); sns.histplot(risk_scores_df['risk_score'], kde=True, ax=ax, bins=20); ax.set_title('Histogram'); st.pyplot(fig); plt.close(fig)
        with col2: fig, ax = plt.subplots(); sns.boxplot(y=risk_scores_df['risk_score'], ax=ax); ax.set_title('Boxplot'); st.pyplot(fig); plt.close(fig)
    elif uploaded_file: st.warning("Risk scores unavailable for EDA.")
    else: st.write("Upload file for EDA.")

with tabs[2]: # Outlier Analysis
    # ... (No changes here, same as previous version) ...
    st.header(tab_titles[2])
    if risk_scores_df is not None and not risk_scores_df.empty:
        low_o, high_o, lb, ub = get_outliers_iqr(risk_scores_df, 'risk_score')
        st.subheader("Risk Score Outlier ID (IQR)");
        if not np.isnan(lb): st.write(f"Bounds: {lb:.4f} - {ub:.4f}")
        else: st.write("Cannot determine outlier bounds.")
        st.markdown("---"); st.subheader("High-Risk Outliers")
        if not high_o.empty: st.write(f"Found: {len(high_o)}"); st.dataframe(high_o['risk_score'].describe().to_frame().T.style.format("{:.4f}"));
        else: st.write("No high-risk outliers.")
        st.markdown("---"); st.subheader("Low-Risk Outliers")
        if not low_o.empty: st.write(f"Found: {len(low_o)}"); st.dataframe(low_o['risk_score'].describe().to_frame().T.style.format("{:.4f}"));
        else: st.write("No low-risk outliers.")
        st.markdown("---"); st.subheader("Download Outlier Details")
        if listado_creditos_loaded and listado_creditos_df is not None and numero_credito_col_name in listado_creditos_df.columns:
            if not high_o.empty or not low_o.empty:
                output_o = io.BytesIO()
                with pd.ExcelWriter(output_o, engine='xlsxwriter') as writer:
                    if not high_o.empty: high_o_details = listado_creditos_df.merge(high_o[['credito', 'risk_score']], left_on=numero_credito_col_name, right_on='credito', how='inner').drop(columns=['credito'], errors='ignore'); high_o_details.to_excel(writer, sheet_name='High Risk', index=False)
                    if not low_o.empty: low_o_details = listado_creditos_df.merge(low_o[['credito', 'risk_score']], left_on=numero_credito_col_name, right_on='credito', how='inner').drop(columns=['credito'], errors='ignore'); low_o_details.to_excel(writer, sheet_name='Low Risk', index=False)
                st.download_button(label="📥 Download Outlier Details", data=output_o.getvalue(), file_name=f"outliers_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.ms-excel")
            else: st.info("No outliers to download.")
        elif uploaded_file: st.warning(f"'ListadoCreditos' or '{numero_credito_col_name}' issue. Check log.")
    elif uploaded_file: st.warning("Risk scores unavailable.")
    else: st.write("Upload file for analysis.")


with tabs[3]: # Customer Data Quality
    st.header(tab_titles[3] + " ('ListadoCreditos')")
    if not uploaded_file: st.write("Upload file for DQA.")
    elif not listado_creditos_loaded or listado_creditos_df is None: st.error("'ListadoCreditos' not loaded. Check log.")
    else:
        df_dqa = listado_creditos_df
        st.subheader("1. Overview"); st.write(f"Rows: {df_dqa.shape[0]}, Columns: {df_dqa.shape[1]}");
        with st.expander("Data Types"): st.dataframe(df_dqa.dtypes.reset_index().rename(columns={'index':'Col',0:'Type'}))
        st.subheader("2. Missing Values"); missing_sum = df_dqa.isnull().sum().reset_index(); missing_sum.columns=['Col','Missing']; missing_sum['%']=(missing_sum['Missing']/len(df_dqa))*100
        missing_sum = missing_sum[missing_sum['Missing']>0].sort_values(by='%',ascending=False)
        if not missing_sum.empty: st.dataframe(missing_sum.style.format({'%':"{:.2f}%"}));
        else: st.success("No missing values! 🎉")
        st.subheader("3. Duplicates"); st.write(f"Full Duplicates: {df_dqa.duplicated().sum()}")
        if numero_credito_col_name in df_dqa.columns: st.write(f"'{numero_credito_col_name}' Duplicates: {df_dqa.duplicated(subset=[numero_credito_col_name]).sum()}")
        
        st.subheader("4. Column Analysis")
        # --- MODIFICATION: Default to only one feature for DQA detailed analysis ---
        default_dqa_col = [df_dqa.columns[0]] if len(df_dqa.columns) > 0 else []
        cols_detail = st.multiselect(
            "Select columns for detail:",
            options=df_dqa.columns.tolist(),
            default=default_dqa_col
        )
        # --- END MODIFICATION ---
        for col in cols_detail:
            with st.expander(f"'{col}' (Type: {df_dqa[col].dtype})"):
                st.write(f"Unique: {df_dqa[col].nunique()}, Missing: {df_dqa[col].isnull().sum()} ({df_dqa[col].isnull().sum()/len(df_dqa)*100:.2f}%)")
                if pd.api.types.is_numeric_dtype(df_dqa[col]): st.dataframe(df_dqa[col].describe().to_frame().T)
                elif pd.api.types.is_object_dtype(df_dqa[col]): st.dataframe(df_dqa[col].value_counts().nlargest(10).reset_index())


with tabs[4]: # Pre-Loan Feature Insights
    st.header(tab_titles[4])

    if not uploaded_file:
        st.write("Upload an Excel file and ensure 'ListadoCreditos' and risk scores are processed.")
    elif listado_creditos_df is None or risk_scores_df is None or risk_scores_df.empty:
        st.warning("Customer data ('ListadoCreditos') or Risk Scores are not available. Please check previous tabs/logs.")
    else:
        st.subheader("Configuration for Feature Insights")
        target_options = ["Raw Risk Score (Continuous)", "Binned Risk Score (Categorical)"]
        chosen_target_type = st.selectbox("How to treat Risk Score for analysis?", target_options, index=0)
        num_bins_fi = 3
        if "Binned" in chosen_target_type:
            num_bins_fi = st.slider("Number of bins for Risk Score:", 2, 10, 3, 1)

        available_features = [col for col in listado_creditos_df.columns if col not in [numero_credito_col_name]]
        if not available_features:
             st.error("No features available from 'ListadoCreditos' for analysis (excluding ID column).")
        else:
            # --- MODIFICATION: Default to only one feature for Feature Insights ---
            default_fi_col = [available_features[0]] if len(available_features) > 0 else []
            selected_cols_fi = st.multiselect(
                "Select customer profile features from 'ListadoCreditos' to analyze:",
                options=available_features,
                default=default_fi_col
            )
            # --- END MODIFICATION ---

            if st.button("🚀 Analyze Feature Importance"):
                if not selected_cols_fi:
                    st.warning("Please select at least one feature to analyze.")
                else:
                    with st.spinner("Preparing data and performing feature analysis..."):
                        prep_result = prepare_feature_importance_data(
                            risk_scores_df, listado_creditos_df, numero_credito_col_name,
                            'risk_score', selected_cols_fi, "Binned" in chosen_target_type, num_bins_fi
                        )
                        if prep_result and len(prep_result) == 6:
                            features_for_analysis_df, target_series, actual_features_analyzed, \
                            original_feature_dtypes, final_target_name, error_message = prep_result
                        else:
                            features_for_analysis_df, target_series, actual_features_analyzed, \
                            original_feature_dtypes, final_target_name, error_message = (None,)*5 + (prep_result if isinstance(prep_result, str) else "Unknown prep error.",)

                        if error_message: st.error(f"Data Prep Error: {error_message}")
                        elif features_for_analysis_df is None or target_series is None: st.error("Failed to prepare data. Check logs.")
                        else:
                            st.success(f"Data prepared. Analyzing {len(actual_features_analyzed)} features against '{final_target_name}'.")
                            # Correlation Analysis
                            if "Raw" in chosen_target_type:
                                st.markdown("---"); st.subheader("A. Correlation with Raw Risk Score")
                                numeric_features_fi = [f for f in actual_features_analyzed if pd.api.types.is_numeric_dtype(original_feature_dtypes.get(f))]
                                if numeric_features_fi:
                                    correlations = {feat: features_for_analysis_df[feat].corr(target_series, method='pearson') if pd.api.types.is_numeric_dtype(target_series) else np.nan for feat in numeric_features_fi}
                                    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Pearson Correlation']).abs().sort_values(by='Pearson Correlation', ascending=False)
                                    corr_df['Pearson Correlation'] = [correlations[idx] for idx in corr_df.index]
                                    st.dataframe(corr_df.style.format("{:.3f}"))
                                    top_n_corr = st.slider("Scatter plots for top N correlated:", 0, min(5, len(corr_df)), 0, key="corr_scatter_slider") # Default 0
                                    for i, feat_name in enumerate(corr_df.index[:top_n_corr]):
                                        fig_corr, ax_corr = plt.subplots(); sns.regplot(x=features_for_analysis_df[feat_name], y=target_series, ax=ax_corr, scatter_kws={'alpha':0.5}); ax_corr.set_title(f"{feat_name} vs. Risk Score"); st.pyplot(fig_corr); plt.close(fig_corr)
                                else: st.info("No numeric features for correlation.")
                            # Group-wise Comparison
                            st.markdown("---"); st.subheader("B. Group-wise Comparisons")
                            categorical_features_fi = [f for f in actual_features_analyzed if not pd.api.types.is_numeric_dtype(original_feature_dtypes.get(f)) or features_for_analysis_df[f].nunique() < 20]
                            if categorical_features_fi:
                                results_groupwise = []
                                for feat in categorical_features_fi:
                                    if features_for_analysis_df[feat].nunique() < 2 or features_for_analysis_df[feat].nunique() > 50 : continue
                                    plot_target_groupwise = pd.to_numeric(target_series, errors='coerce') if "Raw" in chosen_target_type else target_series
                                    
                                    # Attempt to order categories for boxplot
                                    try:
                                        if "Raw" in chosen_target_type and pd.api.types.is_numeric_dtype(plot_target_groupwise):
                                            order = features_for_analysis_df.groupby(feat)[final_target_name].median().sort_values().index.astype(str)
                                        else: # Binned target or non-numeric feature
                                            order = sorted(features_for_analysis_df[feat].dropna().unique().astype(str))
                                    except Exception: # Fallback if ordering fails
                                        order = sorted(features_for_analysis_df[feat].dropna().unique().astype(str))
                                    
                                    if len(order) > 0: # Proceed only if there are categories to plot
                                        fig_box_fi, ax_box_fi = plt.subplots(); 
                                        sns.boxplot(x=features_for_analysis_df[feat].astype(str), y=plot_target_groupwise, ax=ax_box_fi, order=order)
                                        ax_box_fi.set_title(f"Risk Score by {feat}"); ax_box_fi.tick_params(axis='x', rotation=45); plt.tight_layout(); st.pyplot(fig_box_fi); plt.close(fig_box_fi)

                                    if "Binned" in chosen_target_type:
                                        ct = pd.crosstab(features_for_analysis_df[feat], target_series)
                                        if ct.size > 0 and 0 not in ct.shape and ct.sum().sum() > 0:
                                            try: chi2, p, _, _ = chi2_contingency(ct); cv = cramers_v(ct.values); results_groupwise.append({'Feature': feat, 'Test': 'Chi2', 'Stat': chi2, 'p': p, "CramerV": cv})
                                            except: pass # Ignore errors in chi2
                                    else:
                                        groups = [target_series[features_for_analysis_df[feat] == cat] for cat in features_for_analysis_df[feat].unique() if target_series[features_for_analysis_df[feat] == cat].shape[0] > 1]
                                        if len(groups) > 1:
                                            try: f_stat, p_anova = f_oneway(*groups); results_groupwise.append({'Feature': feat, 'Test': 'ANOVA', 'Stat': f_stat, 'p': p_anova, "CramerV": np.nan})
                                            except: pass # Ignore errors in ANOVA
                                if results_groupwise: st.dataframe(pd.DataFrame(results_groupwise).style.format({'Stat': "{:.3f}", 'p': "{:.3g}", "CramerV": "{:.3f}"}))
                                else: st.info("No suitable categories for group tests.")
                            else: st.info("No categorical features for group comparison.")
                            # Mutual Information
                            st.markdown("---"); st.subheader("C. Mutual Information")
                            mi_features_df = pd.DataFrame(index=features_for_analysis_df.index); label_encoders = {}
                            for feat in actual_features_analyzed:
                                if not pd.api.types.is_numeric_dtype(original_feature_dtypes.get(feat)):
                                    le = LabelEncoder(); mi_features_df[feat] = le.fit_transform(features_for_analysis_df[feat].astype(str)); label_encoders[feat] = le
                                else: mi_features_df[feat] = features_for_analysis_df[feat].fillna(features_for_analysis_df[feat].median())
                            if not mi_features_df.empty:
                                mi_target = target_series.astype(int) if "Binned" in chosen_target_type else pd.to_numeric(target_series, errors='coerce').fillna(target_series.median())
                                mi_func = mutual_info_classif if "Binned" in chosen_target_type else mutual_info_regression
                                mi_scores = mi_func(mi_features_df, mi_target, discrete_features='auto', random_state=42)
                                mi_df = pd.DataFrame({'Feature': actual_features_analyzed, 'MI': mi_scores}).sort_values(by='MI', ascending=False)
                                st.dataframe(mi_df.style.format({'MI': "{:.4f}"}))
                                fig_mi, ax_mi = plt.subplots(figsize=(10, max(5, len(mi_df)*0.3))); sns.barplot(x='MI',y='Feature',data=mi_df,ax=ax_mi); ax_mi.set_title("Mutual Information"); plt.tight_layout(); st.pyplot(fig_mi); plt.close(fig_mi)
                            else: st.info("No features for MI.")
# --- Footer ---
st.markdown("---")
st.markdown("App developed by your Expert Data Scientist Antonio Medrano, CepSA.")
