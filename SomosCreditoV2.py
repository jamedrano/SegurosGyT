import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from scipy.stats import chi2_contingency, f_oneway, zscore
import logging

# --- MODIFICATION: Configure logging to write to an in-memory stream ---

# 1. Create an in-memory stream to hold the log messages.
log_stream = io.StringIO()

# 2. Configure the logger to use this stream instead of a file.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 3. Create a handler that writes to our in-memory stream.
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setLevel(logging.INFO)

# 4. Create a formatter to define the log message structure.
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# 5. Add the handler to the logger, but only if it doesn't have handlers already.
# This is crucial for Streamlit's rerun behavior to prevent duplicate logs.
if not logger.handlers:
    logger.addHandler(stream_handler)


# --- Utility Functions ---

def validate_dates(df, date_cols):
    """Converts a list of columns in a DataFrame to datetime objects."""
    df_copy = df.copy()
    for col in date_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    return df_copy

def get_outliers_iqr(df, column_name):
    """Identifies outliers using the Interquartile Range (IQR) method."""
    if df is None or df.empty or column_name not in df.columns or df[column_name].isnull().all():
        return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
    Q1, Q3 = df[column_name].quantile(0.25), df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        lower_bound, upper_bound = Q1, Q3
    else:
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    low_outliers = df[df[column_name] < lower_bound]
    high_outliers = df[df[column_name] > upper_bound]
    return low_outliers, high_outliers, lower_bound, upper_bound

def get_outliers_zscore(df, column_name, threshold=3):
    """Identifies outliers using the Z-score method and separates them into low/high groups."""
    if df is None or df.empty or column_name not in df.columns or df[column_name].isnull().all():
        return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
    col_data = df[column_name].dropna()
    if col_data.empty: return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
    z_scores = np.abs(zscore(col_data))
    outlier_indices = col_data[z_scores > threshold].index
    outliers = df.loc[outlier_indices]
    data_mean = col_data.mean()
    high_outliers = outliers[outliers[column_name] > data_mean]
    low_outliers = outliers[outliers[column_name] < data_mean]
    return low_outliers, high_outliers, np.nan, np.nan

def cramers_v(confusion_matrix):
    """Calculates Cramer's V statistic for categorical-categorical association."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n == 0: return 0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if r == 1 or k == 1: return 0
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if rcorr < 1 or kcorr < 1: return 0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# --- Core Logic and Cached Functions ---

def calculate_risk_score_df(df_input, grace_period_days, weights):
    if df_input.empty:
        logger.error("Input DataFrame is empty.")
        return None
    df = df_input.copy()
    required_base_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro', 'fechaTRansaccion', 'credito', 'reglaCobranza', 'cuotaEsperada', 'totalTrans', 'saldoCapitalActual', 'totalDesembolso', 'cobranzaTrans', 'categoriaProductoCrediticio']
    missing_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {', '.join(missing_cols)}")
        st.error(f"Sheet 'HistoricoPagoCuotas' missing columns: {', '.join(missing_cols)}")
        return None
    date_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro', 'fechaTRansaccion']
    df = validate_dates(df, date_cols)
    df['credito'] = df['credito'].astype(str)
    grace_period = pd.Timedelta(days=grace_period_days)
    mask_valid_dates_late_payment = df['fechaPagoRecibido'].notna() & df['fechaEsperadaPago'].notna()
    df['late_payment'] = 0
    df.loc[mask_valid_dates_late_payment, 'late_payment'] = (df.loc[mask_valid_dates_late_payment, 'fechaPagoRecibido'] > (df.loc[mask_valid_dates_late_payment, 'fechaEsperadaPago'] + grace_period)).astype(int)
    late_payment_counts = df.groupby('credito')['late_payment'].sum()
    total_payments_due_count = df.groupby('credito')['fechaEsperadaPago'].count()
    late_payment_ratio = (late_payment_counts / total_payments_due_count.replace(0, np.nan)).fillna(0)
    df['totalTrans'] = pd.to_numeric(df['totalTrans'], errors='coerce').fillna(0)
    total_payment_made_monetary = df.groupby('credito')['totalTrans'].sum()
    df['cuotaEsperada'] = pd.to_numeric(df['cuotaEsperada'], errors='coerce').fillna(0)
    total_payment_expected_monetary = df.groupby('credito')['cuotaEsperada'].sum()
    payment_coverage_ratio = (total_payment_made_monetary / total_payment_expected_monetary.replace(0, np.nan)).fillna(1).replace([np.inf, -np.inf], 1)
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
    df_for_return = creditos_unicos.copy()
    component_cols_to_scale = ['late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']
    for col_name in component_cols_to_scale:
        min_val, max_val = creditos_unicos[col_name].min(), creditos_unicos[col_name].max()
        scaled_col_name = f'{col_name}_scaled'
        if max_val == min_val:
            df_for_return[scaled_col_name] = 0.0
        else:
            df_for_return[scaled_col_name] = (creditos_unicos[col_name] - min_val) / (max_val - min_val)
        df_for_return[scaled_col_name] = df_for_return[scaled_col_name].fillna(0)
    df_for_return['risk_score'] = (
        weights['late_payment_ratio'] * df_for_return['late_payment_ratio_scaled'] +
        weights['payment_coverage_ratio'] * (1 - df_for_return['payment_coverage_ratio_scaled']) +
        weights['outstanding_balance_ratio'] * df_for_return['outstanding_balance_ratio_scaled'] +
        weights['collection_activity_count'] * df_for_return['collection_activity_count_scaled']
    )
    cols_to_return = ['credito', 'risk_score', 'late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']
    logger.info("Risk score calculation completed successfully.")
    return df_for_return[cols_to_return]

@st.cache_data
def prepare_preloan_insights_data(risk_df_with_components, listado_df, id_col_listado, target_col_name_in_risk_scores, selected_features_from_listado, bin_target_flag, num_bins_for_target=3):
    if risk_df_with_components is None or risk_df_with_components.empty or listado_df is None or listado_df.empty: return None, None, None, None, None, "Risk scores or customer data not available."
    if id_col_listado not in listado_df.columns: return None, None, None, None, None, f"ID column '{id_col_listado}' not found in customer data."
    if target_col_name_in_risk_scores not in risk_df_with_components.columns: return None, None, None, None, None, f"Target column '{target_col_name_in_risk_scores}' not found."
    listado_copy = listado_df.copy(); risk_scores_copy = risk_df_with_components.copy()
    listado_copy[id_col_listado] = listado_copy[id_col_listado].astype(str)
    risk_scores_copy['credito'] = risk_scores_copy['credito'].astype(str)
    merged_df = pd.merge(listado_copy, risk_scores_copy[['credito', target_col_name_in_risk_scores]], left_on=id_col_listado, right_on='credito', how='inner')
    if merged_df.empty: return None, None, None, None, None, "No matching records found after merging."
    features_to_analyze = [f for f in selected_features_from_listado if f in merged_df.columns and f not in [target_col_name_in_risk_scores, id_col_listado, 'credito']]
    if not features_to_analyze: return None, None, None, None, None, "No valid features for analysis."
    analysis_subset_df = merged_df[features_to_analyze + [target_col_name_in_risk_scores]].copy()
    y_series = analysis_subset_df[target_col_name_in_risk_scores].copy()
    final_target_name = target_col_name_in_risk_scores
    if bin_target_flag:
        if y_series.nunique() <= 1: return None, None, None, None, None, f"Target '{target_col_name_in_risk_scores}' has <=1 unique value, cannot bin."
        try:
            discretizer = KBinsDiscretizer(n_bins=num_bins_for_target, encode='ordinal', strategy='quantile', subsample=None)
            binned_target_values = discretizer.fit_transform(y_series.to_frame())
            y_series = pd.Series(binned_target_values.ravel().astype(int).astype(str), index=y_series.index)
            final_target_name = target_col_name_in_risk_scores + '_binned'; y_series.name = final_target_name
        except ValueError as e: return None, None, None, None, None, f"Error binning target '{target_col_name_in_risk_scores}': {e}."
    x_features_df = pd.DataFrame(index=analysis_subset_df.index)
    original_dtypes = {}
    for feature in features_to_analyze:
        original_dtypes[feature] = analysis_subset_df[feature].dtype
        if pd.api.types.is_numeric_dtype(analysis_subset_df[feature]): x_features_df[feature] = analysis_subset_df[feature].fillna(analysis_subset_df[feature].median())
        elif pd.api.types.is_object_dtype(analysis_subset_df[feature]) or pd.api.types.is_categorical_dtype(analysis_subset_df[feature]):
            mode_val = analysis_subset_df[feature].mode()
            impute_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            x_features_df[feature] = analysis_subset_df[feature].fillna(impute_val)
        else: x_features_df[feature] = analysis_subset_df[feature].astype(str).fillna("Unknown")
    return x_features_df, y_series, features_to_analyze, original_dtypes, final_target_name, None

@st.cache_data
def prepare_mi_data(risk_scores_data, listado_data, id_col_listado, target_col_name_in_risk_scores, selected_features_from_listado, bin_target_flag, num_bins_for_target=3):
    if risk_scores_data is None or risk_scores_data.empty or listado_data is None or listado_data.empty: return None, None, None, None, "Risk scores or customer data not available."
    if id_col_listado not in listado_data.columns: return None, None, None, None, f"ID column '{id_col_listado}' not found in customer data."
    if target_col_name_in_risk_scores not in risk_scores_data.columns: return None, None, None, None, f"Target column '{target_col_name_in_risk_scores}' not found."
    listado_copy = listado_data.copy(); risk_scores_copy = risk_scores_data.copy()
    listado_copy[id_col_listado] = listado_copy[id_col_listado].astype(str)
    risk_scores_copy['credito'] = risk_scores_copy['credito'].astype(str)
    merged_df_for_mi = pd.merge(listado_copy, risk_scores_copy[['credito', target_col_name_in_risk_scores]], left_on=id_col_listado, right_on='credito', how='inner')
    if merged_df_for_mi.empty: return None, None, None, None, "No matching records found after merging."
    features_to_process = [f for f in selected_features_from_listado if f in merged_df_for_mi.columns]
    if not features_to_process: return None, None, None, None, "No valid features for MI."
    X_mi = pd.DataFrame(index=merged_df_for_mi.index)
    discrete_mask = []; processed_feature_names_ordered = []
    for feature in features_to_process:
        col_data = merged_df_for_mi[feature].copy()
        if pd.api.types.is_numeric_dtype(col_data):
            X_mi[feature] = col_data.fillna(col_data.median())
            discrete_mask.append(col_data.nunique(dropna=False) < 20)
            processed_feature_names_ordered.append(feature)
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            mode_val = col_data.mode(); impute_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            col_data_filled = col_data.fillna(impute_val).astype(str)
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_mi[feature] = encoder.fit_transform(col_data_filled.to_frame())
            discrete_mask.append(True); processed_feature_names_ordered.append(feature)
        else: logger.warning(f"Skipping feature '{feature}' for MI due to unhandled data type: {col_data.dtype}"); continue
    if X_mi.empty or not processed_feature_names_ordered: return None, None, None, None, "No features were processed for MI."
    y_mi_target_name = target_col_name_in_risk_scores
    y_mi = merged_df_for_mi[target_col_name_in_risk_scores].copy()
    if bin_target_flag:
        if y_mi.nunique() <= 1: return None, None, None, None, f"Target '{y_mi_target_name}' has <=1 unique value, cannot bin."
        try:
            discretizer = KBinsDiscretizer(n_bins=num_bins_for_target, encode='ordinal', strategy='quantile', subsample=None)
            y_mi_binned = discretizer.fit_transform(y_mi.to_frame())
            y_mi = pd.Series(y_mi_binned.ravel().astype(int), index=y_mi.index)
        except ValueError as e: return None, None, None, None, f"Error binning target '{y_mi_target_name}' for MI: {e}."
    else:
        y_mi_numeric = pd.to_numeric(y_mi, errors='coerce')
        y_mi = y_mi_numeric.fillna(y_mi_numeric.median())
    return X_mi[processed_feature_names_ordered], y_mi, processed_feature_names_ordered, discrete_mask, None

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🏍️ Motorcycle Loan Risk & Data Insights App")

# --- Sidebar Configuration ---
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

# --- Data Loading and Processing ---
risk_scores_df, listado_creditos_df = None, None
processed_data_info, historico_pago_cuotas_loaded, listado_creditos_loaded = "", False, False
numero_credito_col_name = "numeroCredito"
logger.info("Application started or refreshed.")

if uploaded_file:
    st.sidebar.info(f"Processing: {uploaded_file.name}")
    logger.info(f"File uploaded: {uploaded_file.name}")
    try:
        hpc_df = pd.read_excel(uploaded_file, sheet_name="HistoricoPagoCuotas")
        processed_data_info += "✅ 'HistoricoPagoCuotas' loaded.\n"; historico_pago_cuotas_loaded = True
        if 'categoriaProductoCrediticio' in hpc_df.columns:
            moto_df = hpc_df[hpc_df["categoriaProductoCrediticio"] == "MOTOS"].copy()
            if not moto_df.empty:
                processed_data_info += f"Found {len(moto_df)} 'MOTOS' records.\n";
                with st.spinner("Calculating risk scores & components..."): risk_scores_df = calculate_risk_score_df(moto_df, grace_period_input, user_weights)
                if risk_scores_df is not None and not risk_scores_df.empty: processed_data_info += f"✅ Risk scores & components for {len(risk_scores_df)} credits.\n"
                else: processed_data_info += "⚠️ Risk score calculation issues.\n"
            else: processed_data_info += "⚠️ No 'MOTOS' data.\n"; logger.warning("No data with 'MOTOS' category found.")
        else: processed_data_info += "❌ 'categoriaProductoCrediticio' missing.\n"; historico_pago_cuotas_loaded = False; logger.error("'categoriaProductoCrediticio' column is missing.")
    except Exception as e: processed_data_info += f"❌ Error 'HistoricoPagoCuotas': {e}\n"; historico_pago_cuotas_loaded = False; logger.error(f"Error loading 'HistoricoPagoCuotas': {e}")
    try:
        lc_df_temp = pd.read_excel(uploaded_file, sheet_name="ListadoCreditos")
        processed_data_info += f"✅ 'ListadoCreditos' loaded ({lc_df_temp.shape[0]}r, {lc_df_temp.shape[1]}c).\n"; listado_creditos_loaded = True
        if numero_credito_col_name in lc_df_temp.columns:
            lc_df_temp[numero_credito_col_name] = lc_df_temp[numero_credito_col_name].astype(str)
            listado_creditos_df = lc_df_temp; processed_data_info += f"'{numero_credito_col_name}' cast to string.\n"
        else: processed_data_info += f"⚠️ '{numero_credito_col_name}' missing.\n"; listado_creditos_df = lc_df_temp; logger.warning(f"ID column '{numero_credito_col_name}' is missing from 'ListadoCreditos'.")
    except Exception as e: processed_data_info += f"❌ Error 'ListadoCreditos': {e}\n"; listado_creditos_loaded = False; logger.error(f"Error loading 'ListadoCreditos': {e}")
else:
    st.info("☝️ Upload an Excel file to begin.")

# --- Sidebar Log Display ---
st.sidebar.text_area("File Processing Log", processed_data_info, height=150)
st.sidebar.subheader("App Activity Log")
with st.sidebar.expander("View Logs", expanded=False):
    log_contents = log_stream.getvalue()
    if log_contents:
        st.code(log_contents)
    else:
        st.text("No log messages yet.")

# --- Main Page Tabs ---
tab_titles = ["📊 Risk Scores", "📈 Risk EDA", "🕵️ Outlier Analysis", "📋 Customer Data Quality", "🔍 Pre-Loan Insights", "📊 Segment Performance", "ℹ️ Feature MI Ranker"]
tabs = st.tabs(tab_titles)

with tabs[0]: # Risk Scores
    st.header(tab_titles[0])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Calculated Risk Scores & Components (per Credit)")
        display_cols_scores = ['credito', 'risk_score'] + risk_score_component_names
        style_format_dict_tab0 = {"risk_score": "{:.4f}", "late_payment_ratio": "{:.4f}", "payment_coverage_ratio": "{:.4f}", "outstanding_balance_ratio": "{:.4f}", "collection_activity_count": "{:.0f}"}
        st.dataframe(risk_scores_df[display_cols_scores].style.format(style_format_dict_tab0), height=500, use_container_width=True)
        output_tab0 = io.BytesIO()
        with pd.ExcelWriter(output_tab0, engine='xlsxwriter') as writer_tab0: risk_scores_df.to_excel(writer_tab0, index=False, sheet_name='RiskScoresAndComponents')
        excel_data_tab0 = output_tab0.getvalue()
        if excel_data_tab0: st.download_button(label="Download Scores & Components", data=excel_data_tab0, file_name=f"risk_scores_components_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif uploaded_file: st.warning("Risk scores not calculated. Check logs for details.")
    else: st.write("Upload a file to view results.")

with tabs[1]: # Risk EDA
    st.header(tab_titles[1])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Risk Score Distribution")
        st.dataframe(risk_scores_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        col1, col2 = st.columns(2)
        with col1: fig, ax = plt.subplots(); sns.histplot(risk_scores_df['risk_score'], kde=True, ax=ax, bins=20); ax.set_title('Histogram of Risk Score'); st.pyplot(fig); plt.close(fig)
        with col2: fig, ax = plt.subplots(); sns.boxplot(y=risk_scores_df['risk_score'], ax=ax); ax.set_title('Boxplot of Risk Score'); st.pyplot(fig); plt.close(fig)
    elif uploaded_file: st.warning("Risk scores are unavailable. Check logs for details.")
    else: st.write("Upload a file to perform EDA.")

with tabs[2]: # Outlier Analysis
    st.header(tab_titles[2])
    if risk_scores_df is not None and not risk_scores_df.empty:
        method = st.radio("Select Outlier Detection Method:", ("IQR Method", "Z-score Method"), horizontal=True, help="IQR is robust to non-normal data. Z-score assumes a more normal distribution.")
        low_o, high_o = pd.DataFrame(), pd.DataFrame()
        if method == "IQR Method":
            st.subheader("Risk Score Outlier ID (IQR Method)")
            low_o, high_o, lb, ub = get_outliers_iqr(risk_scores_df, 'risk_score')
            if not np.isnan(lb): st.write(f"**Lower Bound:** `{lb:.4f}` | **Upper Bound:** `{ub:.4f}`")
            else: st.write("Cannot determine outlier bounds (e.g., no variance in data).")
        elif method == "Z-score Method":
            st.subheader("Risk Score Outlier ID (Z-score Method)")
            z_threshold = st.slider("Z-score Threshold:", min_value=1.5, max_value=4.0, value=3.0, step=0.1, help="A higher threshold is less sensitive and will find fewer, more extreme outliers.")
            low_o, high_o, _, _ = get_outliers_zscore(risk_scores_df, 'risk_score', threshold=z_threshold)
            st.write(f"Outliers are defined as data points with an absolute Z-score greater than **{z_threshold}**.")
        st.markdown("---")
        st.subheader("High-Risk Outliers");
        if not high_o.empty: st.write(f"Found: {len(high_o)} high-risk outliers."); st.dataframe(high_o['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        else: st.write("No high-risk outliers found with the current settings.")
        st.markdown("---")
        st.subheader("Low-Risk Outliers")
        if not low_o.empty: st.write(f"Found: {len(low_o)} low-risk outliers."); st.dataframe(low_o['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        else: st.write("No low-risk outliers found with the current settings.")
        st.markdown("---")
        st.subheader("Download Outlier Details")
        if listado_creditos_loaded and listado_creditos_df is not None and numero_credito_col_name in listado_creditos_df.columns:
            if not high_o.empty or not low_o.empty:
                output_o = io.BytesIO()
                with pd.ExcelWriter(output_o, engine='xlsxwriter') as writer:
                    if not high_o.empty: high_o_details = listado_creditos_df.merge(high_o, left_on=numero_credito_col_name, right_on='credito', how='inner'); high_o_details.to_excel(writer, sheet_name='High Risk', index=False)
                    if not low_o.empty: low_o_details = listado_creditos_df.merge(low_o, left_on=numero_credito_col_name, right_on='credito', how='inner'); low_o_details.to_excel(writer, sheet_name='Low Risk', index=False)
                excel_data_outlier = output_o.getvalue()
                if excel_data_outlier: st.download_button(label="Download Outlier Details", data=excel_data_outlier, file_name=f"outliers_{method.replace(' ','')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else: st.info("No outliers to download.")
        elif uploaded_file: st.warning(f"Outlier details cannot be generated. Check if 'ListadoCreditos' sheet and '{numero_credito_col_name}' column are present and correctly loaded.")
    elif uploaded_file: st.warning("Risk scores are unavailable. Cannot perform outlier analysis.")
    else: st.write("Upload a file to perform analysis.")

with tabs[3]: # Customer Data Quality (Code is unchanged)
    # ... code for this tab ...
    pass # Placeholder for brevity

with tabs[4]: # Pre-Loan Insights (Code is unchanged)
    # ... code for this tab ...
    pass # Placeholder for brevity

with tabs[5]: # Segment Performance Analyzer (Code is unchanged)
    # ... code for this tab ...
    pass # Placeholder for brevity

with tabs[6]: # Feature MI Ranker (Code is unchanged)
    # ... code for this tab ...
    pass # Placeholder for brevity

st.markdown("---")
st.markdown("App developed by your Expert Data Scientist, Antonio Medrano, CepSA")
