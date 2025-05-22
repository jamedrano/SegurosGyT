import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from scipy.stats import chi2_contingency, f_oneway

# --- Core Risk Score Calculation Logic ---
# MODIFIED to return raw components as well
def calculate_risk_score_df(df_input, grace_period_days, weights):
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
    
    df['cuotaCubierta'] = pd.to_numeric(df['cuotaCubierta'], errors='coerce').fillna(0) # Ensure numeric
    df['cuotaEsperada'] = pd.to_numeric(df['cuotaEsperada'], errors='coerce') # Ensure numeric
    total_payment_made = df.groupby('credito')['cuotaCubierta'].sum()
    total_payment_expected = df.groupby('credito')['cuotaEsperada'].sum()
    payment_coverage_ratio = (total_payment_made / total_payment_expected.replace(0, np.nan))
    payment_coverage_ratio = payment_coverage_ratio.fillna(1).replace([np.inf, -np.inf], 1) # Default to 1 if expected is 0 or NaN

    df['saldoCapitalActual'] = pd.to_numeric(df['saldoCapitalActual'], errors='coerce')
    df['totalDesembolso'] = pd.to_numeric(df['totalDesembolso'], errors='coerce')
    last_saldo = df.groupby('credito')['saldoCapitalActual'].last()
    first_desembolso = df.groupby('credito')['totalDesembolso'].first()
    outstanding_balance_ratio = (last_saldo / first_desembolso.replace(0, np.nan)).fillna(0)
    
    df['cobranzaTrans'] = pd.to_numeric(df['cobranzaTrans'], errors='coerce').fillna(0)
    df['collection_activity'] = (df['cobranzaTrans'] > 0).astype(int)
    collection_activity_count = df.groupby('credito')['collection_activity'].sum()
    
    # DataFrame with unique credits and raw components
    creditos_unicos = pd.DataFrame(df['credito'].unique(), columns=['credito'])
    creditos_unicos = creditos_unicos.merge(late_payment_ratio.to_frame(name='late_payment_ratio'), left_on='credito', right_index=True, how='left').fillna({'late_payment_ratio': 0})
    creditos_unicos = creditos_unicos.merge(payment_coverage_ratio.to_frame(name='payment_coverage_ratio'), left_on='credito', right_index=True, how='left').fillna({'payment_coverage_ratio': 1})
    creditos_unicos = creditos_unicos.merge(outstanding_balance_ratio.to_frame(name='outstanding_balance_ratio'), left_on='credito', right_index=True, how='left').fillna({'outstanding_balance_ratio': 0})
    creditos_unicos = creditos_unicos.merge(collection_activity_count.to_frame(name='collection_activity_count'), left_on='credito', right_index=True, how='left').fillna({'collection_activity_count': 0})

    # Store raw components before scaling
    df_for_return = creditos_unicos.copy()

    # Scaling
    component_cols_to_scale = ['late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']
    for col_name in component_cols_to_scale:
        min_val, max_val = creditos_unicos[col_name].min(), creditos_unicos[col_name].max()
        scaled_col_name = f'{col_name}_scaled'
        if max_val == min_val: # Avoid division by zero if all values are the same
            # If min=max=0, scaled is 0. If min=max!=0, scaled is 0.5 (or 0, depending on logic)
            # For ratios, if min=max, it often implies a constant state, 0 might be appropriate.
            # For payment_coverage_ratio where higher is better, if all are 1 (fully covered), 1-scaled would be 0.
            # Let's stick to 0 for simplicity, meaning no variability from this feature in this case.
            df_for_return[scaled_col_name] = 0.0
        else:
            df_for_return[scaled_col_name] = (creditos_unicos[col_name] - min_val) / (max_val - min_val)
        df_for_return[scaled_col_name] = df_for_return[scaled_col_name].fillna(0) # Handle NaNs from scaling (e.g. if original col was all NaN)

    # Calculate final risk score
    df_for_return['risk_score'] = (
        weights['late_payment_ratio'] * df_for_return['late_payment_ratio_scaled'] +
        weights['payment_coverage_ratio'] * (1 - df_for_return['payment_coverage_ratio_scaled']) + # Invert coverage
        weights['outstanding_balance_ratio'] * df_for_return['outstanding_balance_ratio_scaled'] +
        weights['collection_activity_count'] * df_for_return['collection_activity_count_scaled']
    )
    
    # Select columns to return: ID, final score, and raw components
    cols_to_return = ['credito', 'risk_score', 'late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']
    return df_for_return[cols_to_return]


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
    phi2 = chi2 / n; r, k = confusion_matrix.shape
    if r == 1 or k == 1 or n == 0: return 0
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)); rcorr = r - ((r-1)**2)/(n-1); kcorr = k - ((k-1)**2)/(n-1)
    if rcorr < 1 or kcorr < 1 : return 0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# --- Data Preparation for Feature Importance Tab (remains the same logic) ---
@st.cache_data
def prepare_feature_importance_data(risk_df_with_components, listado_df, id_col_listado, target_col_risk, selected_features_listado, bin_target, num_bins=3):
    # Note: risk_df_with_components now expected to have risk_score
    if risk_df_with_components is None or risk_df_with_components.empty or listado_df is None or listado_df.empty:
        return None, "Risk scores or customer data not available."
    if id_col_listado not in listado_df.columns: return None, f"ID col '{id_col_listado}' not in customer data."
    if target_col_risk not in risk_df_with_components.columns: return None, f"Target '{target_col_risk}' not in risk scores."
    
    listado_df_copy = listado_df.copy(); risk_df_copy = risk_df_with_components.copy()
    listado_df_copy[id_col_listado] = listado_df_copy[id_col_listado].astype(str)
    risk_df_copy['credito'] = risk_df_copy['credito'].astype(str) # 'credito' is the ID in risk_df_copy
    
    merged_df = pd.merge(listado_df_copy, risk_df_copy, left_on=id_col_listado, right_on='credito', how='inner')
    if merged_df.empty: return None, "No matching records found after merging."
    
    features_to_analyze = [f for f in selected_features_listado if f in merged_df.columns and f not in [target_col_risk, id_col_listado, 'credito'] + risk_score_component_names]
    if not features_to_analyze: return None, "No valid features selected/found for analysis."
    
    analysis_df = merged_df[features_to_analyze + [target_col_risk]].copy() # Only need target_col_risk for y
    target_name_for_y = target_col_risk # This is the column to use as 'y'
    
    if bin_target:
        if analysis_df[target_col_risk].nunique() <= 1: return None, f"Target '{target_col_risk}' has <=1 unique value, cannot bin."
        try:
            discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile', subsample=None)
            # We need to apply binning on the original target_col_risk from analysis_df for the y_series
            binned_target_values = discretizer.fit_transform(analysis_df[[target_col_risk]])
            # Create the y-series for return
            y_series = pd.Series(binned_target_values.ravel().astype(int).astype(str), index=analysis_df.index, name=target_col_risk + '_binned')
            target_name_for_y = target_col_risk + '_binned'
        except ValueError as e: return None, f"Error binning target '{target_col_risk}': {e}."
    else:
        y_series = analysis_df[target_col_risk] # Use raw target as y

    # Prepare X (features_for_analysis_df)
    x_features_df = pd.DataFrame(index=analysis_df.index)
    original_dtypes = {}
    for feature in features_to_analyze:
        original_dtypes[feature] = analysis_df[feature].dtype # Store original dtype from analysis_df (before it might be altered)
        if pd.api.types.is_numeric_dtype(analysis_df[feature]):
            x_features_df[feature] = analysis_df[feature].fillna(analysis_df[feature].median())
        elif pd.api.types.is_object_dtype(analysis_df[feature]) or pd.api.types.is_categorical_dtype(analysis_df[feature]):
            x_features_df[feature] = analysis_df[feature].fillna(analysis_df[feature].mode().iloc[0] if not analysis_df[feature].mode().empty else "Unknown")
        else: 
            x_features_df[feature] = analysis_df[feature].astype(str).fillna("Unknown")
            
    return x_features_df, y_series, features_to_analyze, original_dtypes, target_name_for_y, None


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🏍️ Motorcycle Loan Risk & Data Insights App")

# Global names for risk score components
risk_score_component_names = ['late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']

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

# --- Data Loading and Initial Processing ---
# risk_scores_df will now contain raw components as well
risk_scores_df, listado_creditos_df = None, None
processed_data_info, historico_pago_cuotas_loaded, listado_creditos_loaded = "", False, False
numero_credito_col_name = "numeroCredito"

if uploaded_file:
    st.sidebar.info(f"Processing: {uploaded_file.name}")
    try:
        hpc_df = pd.read_excel(uploaded_file, sheet_name="HistoricoPagoCuotas")
        processed_data_info += "✅ 'HistoricoPagoCuotas' loaded.\n"; historico_pago_cuotas_loaded = True
        if 'categoriaProductoCrediticio' in hpc_df.columns:
            moto_df = hpc_df[hpc_df["categoriaProductoCrediticio"] == "MOTOS"].copy()
            if not moto_df.empty:
                processed_data_info += f"Found {len(moto_df)} 'MOTOS' records.\n";
                with st.spinner("Calculating risk scores & components..."): # Spinner text updated
                    risk_scores_df = calculate_risk_score_df(moto_df, grace_period_input, user_weights)
                if risk_scores_df is not None and not risk_scores_df.empty: 
                    processed_data_info += f"✅ Risk scores & components for {len(risk_scores_df)} credits.\n"
                else: processed_data_info += "⚠️ Risk score calculation issues.\n"
            else: processed_data_info += "⚠️ No 'MOTOS' data.\n"
        else: processed_data_info += "❌ 'categoriaProductoCrediticio' missing.\n"; historico_pago_cuotas_loaded = False
    except Exception as e: processed_data_info += f"❌ Error 'HistoricoPagoCuotas': {e}\n"; historico_pago_cuotas_loaded = False
    try:
        lc_df_temp = pd.read_excel(uploaded_file, sheet_name="ListadoCreditos")
        processed_data_info += f"✅ 'ListadoCreditos' loaded ({lc_df_temp.shape[0]}r, {lc_df_temp.shape[1]}c).\n"; listado_creditos_loaded = True
        if numero_credito_col_name in lc_df_temp.columns:
            lc_df_temp[numero_credito_col_name] = lc_df_temp[numero_credito_col_name].astype(str)
            listado_creditos_df = lc_df_temp; processed_data_info += f"'{numero_credito_col_name}' cast.\n"
        else:
            processed_data_info += f"⚠️ '{numero_credito_col_name}' missing.\n"; listado_creditos_df = lc_df_temp
    except Exception as e: processed_data_info += f"❌ Error 'ListadoCreditos': {e}\n"; listado_creditos_loaded = False
    st.sidebar.text_area("File Processing Log", processed_data_info, height=200)
else:
    st.info("☝️ Upload an Excel file to begin.")

# --- Tabs ---
tab_titles = ["📊 Risk Scores", "📈 Risk EDA", "🕵️ Outlier Analysis", "📋 Customer Data Quality", "🔍 Pre-Loan Feature Insights", " сегмент Segment Performance"]
tabs = st.tabs(tab_titles)

with tabs[0]: # Risk Scores
    st.header(tab_titles[0])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Calculated Risk Scores & Components (per Credit)")
        # Display only key columns initially, or allow user to select
        display_cols_scores = ['credito', 'risk_score'] + risk_score_component_names
        st.dataframe(risk_scores_df[display_cols_scores].style.format({
            "risk_score": "{:.4f}",
            "late_payment_ratio": "{:.2%}",
            "payment_coverage_ratio": "{:.2%}",
            "outstanding_balance_ratio": "{:.2%}",
            "collection_activity_count": "{:.0f}"
        }), height=500, use_container_width=True)
        
        output = io.BytesIO(); risk_scores_df.to_excel(pd.ExcelWriter(output, engine='xlsxwriter'), index=False, sheet_name='RiskScoresAndComponents'); excel_data = output.getvalue()
        st.download_button(label="📥 Download Scores & Components", data=excel_data, file_name=f"risk_scores_components_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.ms-excel")
    elif uploaded_file and historico_pago_cuotas_loaded: st.warning("Risk scores not calculated. Check log.")
    elif uploaded_file and not historico_pago_cuotas_loaded: st.error("'HistoricoPagoCuotas' failed to load.")
    else: st.write("Upload file for results.")

with tabs[1]: # Risk EDA
    st.header(tab_titles[1])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Risk Score Distribution"); st.dataframe(risk_scores_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        col1, col2 = st.columns(2)
        with col1: fig, ax = plt.subplots(); sns.histplot(risk_scores_df['risk_score'], kde=True, ax=ax, bins=20); ax.set_title('Histogram'); st.pyplot(fig); plt.close(fig)
        with col2: fig, ax = plt.subplots(); sns.boxplot(y=risk_scores_df['risk_score'], ax=ax); ax.set_title('Boxplot'); st.pyplot(fig); plt.close(fig)
    elif uploaded_file: st.warning("Risk scores unavailable for EDA.")
    else: st.write("Upload file for EDA.")

with tabs[2]: # Outlier Analysis
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
                    # Ensure 'risk_score' and other components are included if needed
                    if not high_o.empty:
                        high_o_details = listado_creditos_df.merge(high_o, left_on=numero_credito_col_name, right_on='credito', how='inner')
                        if 'credito_y' in high_o_details.columns : high_o_details = high_o_details.drop(columns=['credito_y']) # Drop redundant if names clash
                        if 'credito_x' in high_o_details.columns : high_o_details = high_o_details.rename(columns={'credito_x':'credito'})
                        high_o_details.to_excel(writer, sheet_name='High Risk', index=False)
                    if not low_o.empty:
                        low_o_details = listado_creditos_df.merge(low_o, left_on=numero_credito_col_name, right_on='credito', how='inner')
                        if 'credito_y' in low_o_details.columns : low_o_details = low_o_details.drop(columns=['credito_y'])
                        if 'credito_x' in low_o_details.columns : low_o_details = low_o_details.rename(columns={'credito_x':'credito'})
                        low_o_details.to_excel(writer, sheet_name='Low Risk', index=False)
                st.download_button(label="📥 Download Outlier Details", data=output_o.getvalue(), file_name=f"outliers_detailed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.ms-excel")
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
        default_dqa_col = [df_dqa.columns[0]] if len(df_dqa.columns) > 0 else []
        cols_detail = st.multiselect("Select columns for detail:", options=df_dqa.columns.tolist(), default=default_dqa_col)
        for col in cols_detail:
            with st.expander(f"'{col}' (Type: {df_dqa[col].dtype})"):
                st.write(f"Unique: {df_dqa[col].nunique()}, Missing: {df_dqa[col].isnull().sum()} ({df_dqa[col].isnull().sum()/len(df_dqa)*100:.2f}%)")
                if pd.api.types.is_numeric_dtype(df_dqa[col]): st.dataframe(df_dqa[col].describe().to_frame().T)
                elif pd.api.types.is_object_dtype(df_dqa[col]): st.dataframe(df_dqa[col].value_counts().nlargest(10).reset_index())

with tabs[4]: # Pre-Loan Feature Insights
    st.header(tab_titles[4])
    # ... (No changes to this tab's internal logic from previous version with fixed scatter plots) ...
    if not uploaded_file: st.write("Upload an Excel file and ensure 'ListadoCreditos' and risk scores are processed.")
    elif listado_creditos_df is None or risk_scores_df is None or risk_scores_df.empty: st.warning("Customer data ('ListadoCreditos') or Risk Scores are not available. Please check previous tabs/logs.")
    else:
        st.subheader("Configuration for Feature Insights")
        target_options = ["Raw Risk Score (Continuous)", "Binned Risk Score (Categorical)"]
        chosen_target_type = st.selectbox("How to treat Risk Score for analysis?", target_options, index=0)
        num_bins_fi = 3
        if "Binned" in chosen_target_type: num_bins_fi = st.slider("Number of bins for Risk Score:", 2, 10, 3, 1)
        available_features = [col for col in listado_creditos_df.columns if col not in [numero_credito_col_name]]
        if not available_features: st.error("No features available from 'ListadoCreditos' for analysis (excluding ID column).")
        else:
            default_fi_col = [available_features[0]] if len(available_features) > 0 else []
            selected_cols_fi = st.multiselect("Select customer profile features from 'ListadoCreditos' to analyze:", options=available_features, default=default_fi_col)
            if st.button("🚀 Analyze Feature Importance"):
                if not selected_cols_fi: st.warning("Please select at least one feature to analyze.")
                else:
                    with st.spinner("Preparing data and performing feature analysis..."):
                        prep_result = prepare_feature_importance_data(risk_scores_df, listado_creditos_df, numero_credito_col_name, 'risk_score', selected_cols_fi, "Binned" in chosen_target_type, num_bins_fi)
                        if prep_result and len(prep_result) == 6: features_for_analysis_df, target_series, actual_features_analyzed, original_feature_dtypes, final_target_name, error_message = prep_result
                        else: features_for_analysis_df, target_series, actual_features_analyzed, original_feature_dtypes, final_target_name, error_message = (None,)*5 + (prep_result if isinstance(prep_result, str) else "Unknown prep error.",)
                        if error_message: st.error(f"Data Prep Error: {error_message}")
                        elif features_for_analysis_df is None or target_series is None: st.error("Failed to prepare data. Check logs.")
                        else:
                            st.success(f"Data prepared. Analyzing {len(actual_features_analyzed)} features against '{final_target_name}'.")
                            if "Raw" in chosen_target_type:
                                st.markdown("---"); st.subheader("A. Correlation with Raw Risk Score")
                                numeric_features_to_correlate = [f for f in actual_features_analyzed if pd.api.types.is_numeric_dtype(original_feature_dtypes.get(f)) and f in selected_cols_fi]
                                if numeric_features_to_correlate:
                                    correlations = {feat: features_for_analysis_df[feat].corr(target_series, method='pearson') if pd.api.types.is_numeric_dtype(target_series) else np.nan for feat in numeric_features_to_correlate}
                                    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Pearson Correlation']).dropna()
                                    if not corr_df.empty:
                                        corr_df_display = corr_df.copy(); corr_df_display['Abs Correlation'] = corr_df_display['Pearson Correlation'].abs(); corr_df_display = corr_df_display.sort_values(by='Abs Correlation', ascending=False)
                                        st.write("Correlation Table (Sorted by Absolute Value):"); st.dataframe(corr_df_display[['Pearson Correlation']].style.format("{:.3f}"))
                                        st.subheader("Scatter Plots for Selected Numeric Features")
                                        for feat_name in corr_df_display.index: 
                                            if feat_name in features_for_analysis_df.columns and pd.api.types.is_numeric_dtype(target_series):
                                                fig_corr, ax_corr = plt.subplots(); sns.regplot(x=features_for_analysis_df[feat_name], y=target_series, ax=ax_corr, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}); ax_corr.set_title(f"Scatter: {feat_name} vs. Risk Score"); ax_corr.set_xlabel(feat_name); ax_corr.set_ylabel(final_target_name); st.pyplot(fig_corr); plt.close(fig_corr)
                                    else: st.info("No valid correlations for selected numeric features.")
                                else: st.info("No numeric features selected for correlation.")
                            st.markdown("---"); st.subheader("B. Group-wise Comparisons")
                            categorical_features_to_group = [f for f in actual_features_analyzed if (not pd.api.types.is_numeric_dtype(original_feature_dtypes.get(f)) or features_for_analysis_df[f].nunique() < 20) and f in selected_cols_fi]
                            if categorical_features_to_group:
                                results_groupwise = []
                                for feat in categorical_features_to_group:
                                    if features_for_analysis_df[feat].nunique() < 2 or features_for_analysis_df[feat].nunique() > 50 : continue
                                    plot_target_groupwise = pd.to_numeric(target_series, errors='coerce') if "Raw" in chosen_target_type else target_series
                                    try: order = (features_for_analysis_df.groupby(feat)[final_target_name].median().sort_values().index.astype(str) if "Raw" in chosen_target_type and pd.api.types.is_numeric_dtype(plot_target_groupwise) and features_for_analysis_df[feat].nunique() > 1 else sorted(features_for_analysis_df[feat].dropna().unique().astype(str)))
                                    except Exception: order = sorted(features_for_analysis_df[feat].dropna().unique().astype(str))
                                    if len(order) > 0 and not plot_target_groupwise.isnull().all():
                                        fig_box_fi, ax_box_fi = plt.subplots(); sns.boxplot(x=features_for_analysis_df[feat].astype(str), y=plot_target_groupwise, ax=ax_box_fi, order=order); ax_box_fi.set_title(f"Risk Score by {feat}"); ax_box_fi.tick_params(axis='x', rotation=45); plt.tight_layout(); st.pyplot(fig_box_fi); plt.close(fig_box_fi)
                                    if "Binned" in chosen_target_type:
                                        ct = pd.crosstab(features_for_analysis_df[feat], target_series);
                                        if ct.size > 0 and 0 not in ct.shape and ct.sum().sum() > 0:
                                            try: chi2, p, _, _ = chi2_contingency(ct); cv = cramers_v(ct.values); results_groupwise.append({'Feature': feat, 'Test': 'Chi2', 'Stat': chi2, 'p': p, "CramerV": cv})
                                            except ValueError: pass 
                                    else:
                                        groups = [target_series[features_for_analysis_df[feat] == cat] for cat in features_for_analysis_df[feat].unique() if target_series[features_for_analysis_df[feat] == cat].shape[0] > 1]
                                        if len(groups) > 1:
                                            try: f_stat, p_anova = f_oneway(*groups); results_groupwise.append({'Feature': feat, 'Test': 'ANOVA', 'Stat': f_stat, 'p': p_anova, "CramerV": np.nan})
                                            except ValueError: pass 
                                if results_groupwise: st.dataframe(pd.DataFrame(results_groupwise).style.format({'Stat': "{:.3f}", 'p': "{:.3g}", "CramerV": "{:.3f}"}))
                                else: st.info("No group tests for selected categories.")
                            else: st.info("No categorical features selected for group comparison.")
                            st.markdown("---"); st.subheader("C. Mutual Information")
                            mi_features_to_analyze = [f for f in actual_features_analyzed if f in selected_cols_fi]
                            if mi_features_to_analyze:
                                mi_features_df = pd.DataFrame(index=features_for_analysis_df.index); label_encoders = {}
                                for feat in mi_features_to_analyze:
                                    if not pd.api.types.is_numeric_dtype(original_feature_dtypes.get(feat)): le = LabelEncoder(); mi_features_df[feat] = le.fit_transform(features_for_analysis_df[feat].astype(str)); label_encoders[feat] = le
                                    else: mi_features_df[feat] = features_for_analysis_df[feat].fillna(features_for_analysis_df[feat].median())
                                if not mi_features_df.empty:
                                    mi_target = target_series.astype(int) if "Binned" in chosen_target_type else pd.to_numeric(target_series, errors='coerce').fillna(target_series.median())
                                    mi_func = mutual_info_classif if "Binned" in chosen_target_type else mutual_info_regression
                                    mi_scores = mi_func(mi_features_df[mi_features_to_analyze], mi_target, discrete_features='auto', random_state=42)
                                    mi_df = pd.DataFrame({'Feature': mi_features_to_analyze, 'MI': mi_scores}).sort_values(by='MI', ascending=False)
                                    st.dataframe(mi_df.style.format({'MI': "{:.4f}"}))
                                    if not mi_df.empty: fig_mi, ax_mi = plt.subplots(figsize=(10, max(5, len(mi_df)*0.3))); sns.barplot(x='MI',y='Feature',data=mi_df,ax=ax_mi); ax_mi.set_title("Mutual Information"); plt.tight_layout(); st.pyplot(fig_mi); plt.close(fig_mi)
                                else: st.info("No features processed for MI from selection.")
                            else: st.info("No features selected for MI.")


with tabs[5]: # Segment Performance Analyzer
    st.header(tab_titles[5])

    if not uploaded_file:
        st.write("Upload an Excel file and ensure 'ListadoCreditos' and risk scores are processed.")
    elif listado_creditos_df is None or risk_scores_df is None or risk_scores_df.empty:
        st.warning("Customer data ('ListadoCreditos') or Risk Scores (with components) are not available. Please check processing.")
    else:
        st.subheader("Define Customer Segment")

        # Merge listado_creditos with risk_scores_df (which now includes raw components)
        # Ensure 'credito' in risk_scores_df is string, and numero_credito_col_name in listado_creditos_df is string
        
        # Defensive copy for merge
        temp_listado_df = listado_creditos_df.copy()
        temp_risk_scores_df = risk_scores_df.copy()

        temp_listado_df[numero_credito_col_name] = temp_listado_df[numero_credito_col_name].astype(str)
        temp_risk_scores_df['credito'] = temp_risk_scores_df['credito'].astype(str)

        # Check for necessary columns in risk_scores_df
        required_risk_cols_for_segment = ['credito', 'risk_score'] + risk_score_component_names
        if not all(col in temp_risk_scores_df.columns for col in required_risk_cols_for_segment):
            st.error(f"Risk score data is missing one or more required columns for segmentation: {', '.join(required_risk_cols_for_segment)}")
        else:
            # Perform the merge
            segment_data_full = pd.merge(
                temp_listado_df,
                temp_risk_scores_df[required_risk_cols_for_segment], # Select only necessary columns
                left_on=numero_credito_col_name,
                right_on='credito',
                how='inner'
            )

            if segment_data_full.empty:
                st.warning("No matching records found between customer data and risk scores for segmentation.")
            else:
                # Identify categorical columns from listado_creditos part of segment_data_full
                # Exclude ID columns and component/score columns
                demographic_cols = [
                    col for col in listado_creditos_df.columns 
                    if col not in [numero_credito_col_name] + required_risk_cols_for_segment # ensure we don't pick score columns
                ]
                
                categorical_demographics = [
                    col for col in demographic_cols 
                    if segment_data_full[col].dtype == 'object' or segment_data_full[col].nunique() < 20 # Heuristic for categorical
                ]


                if not categorical_demographics:
                    st.info("No suitable categorical demographic variables found for segmentation in 'ListadoCreditos'.")
                else:
                    selected_segment_vars = st.multiselect(
                        "Select demographic variables for segmentation:",
                        options=categorical_demographics,
                        default=categorical_demographics[0] if categorical_demographics else []
                    )

                    filters = {}
                    for var in selected_segment_vars:
                        unique_levels = sorted(segment_data_full[var].dropna().unique().astype(str))
                        if unique_levels: #Proceed only if there are levels to select
                            filters[var] = st.multiselect(
                                f"Select levels for '{var}':",
                                options=unique_levels,
                                default=unique_levels # Default to all levels initially for simplicity for the user
                            )
                        else:
                            st.caption(f"No selectable levels for '{var}' (column might be empty or all NaN).")


                    # Apply filters
                    segmented_df = segment_data_full.copy()
                    if filters:
                        query_parts = []
                        for var, levels in filters.items():
                            if levels: # Only apply filter if levels are selected
                                # Ensure levels are treated as strings for query if original column is object
                                str_levels = [f"'{str(level).replace(\"'\", \"\\'\")}'" for level in levels] # Escape single quotes in levels
                                query_parts.append(f"`{var}` in ({', '.join(str_levels)})")
                        
                        if query_parts:
                            try:
                                segmented_df = segmented_df.query(" and ".join(query_parts))
                            except Exception as e:
                                st.error(f"Error applying filters: {e}. This might be due to special characters in column names or levels.")
                                segmented_df = pd.DataFrame() # Empty df on error

                    if not segmented_df.empty:
                        st.markdown("---")
                        st.subheader(f"Performance for Selected Segment ({len(segmented_df)} loans)")

                        # Calculate averages for the segment
                        avg_risk_score_segment = segmented_df['risk_score'].mean()
                        avg_components_segment = segmented_df[risk_score_component_names].mean()

                        segment_summary_data = {'Metric': ['Risk Score'] + risk_score_component_names,
                                                'Segment Average': [avg_risk_score_segment] + avg_components_segment.tolist()}
                        segment_summary_df = pd.DataFrame(segment_summary_data)

                        # Formatting for display
                        format_dict_segment = {"Segment Average": "{:.4f}"}
                        for comp in risk_score_component_names:
                            if 'ratio' in comp: format_dict_segment[comp] = "{:.2%}" # For ratios
                            elif 'count' in comp: format_dict_segment[comp] = "{:.2f}" # For counts

                        st.dataframe(segment_summary_df.set_index('Metric').style.format(format_dict_segment))

                        # Optional: Comparison with overall portfolio averages
                        with st.expander("Compare with Overall Portfolio Averages"):
                            avg_risk_score_overall = risk_scores_df['risk_score'].mean()
                            avg_components_overall = risk_scores_df[risk_score_component_names].mean()
                            
                            overall_summary_data = {'Metric': ['Risk Score'] + risk_score_component_names,
                                                    'Overall Average': [avg_risk_score_overall] + avg_components_overall.tolist()}
                            overall_summary_df = pd.DataFrame(overall_summary_data)
                            
                            comparison_df = pd.merge(segment_summary_df, overall_summary_df, on="Metric")
                            st.dataframe(comparison_df.set_index('Metric').style.format({"Segment Average":"{:.4f}", "Overall Average":"{:.4f}"}))

                    elif filters and not query_parts: # Filters selected but no levels chosen for any
                        st.info("Please select levels for at least one chosen demographic variable to see segment performance.")
                    elif filters and query_parts: # Filters applied but resulted in empty df
                        st.info("No customers found matching all selected criteria.")
                    else: # No filters selected yet
                        st.info("Select demographic variables and their levels to analyze segment performance.")
# --- Footer ---
st.markdown("---")
st.markdown("App developed by your Expert Data Scientist Antonio Medrano, CepSA.")
