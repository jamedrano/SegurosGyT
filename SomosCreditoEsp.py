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

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FIX [CRITICAL]: Function `validate_dates` was called but not defined. ---
# REASON: This would raise a NameError and crash the script at the start of the risk score calculation.
# FIX: Added a robust function to convert specified columns to datetime, coercing any errors to NaT (Not a Time).
def validate_dates(df, date_cols):
    """Converts a list of columns in a DataFrame to datetime objects."""
    df_copy = df.copy()
    for col in date_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    return df_copy

# --- Core Risk Score Calculation Logic ---
def calculate_risk_score_df(df_input, grace_period_days, weights):
    if df_input.empty:
        logger.error("Input DataFrame is empty.")
        return None
    df = df_input.copy()
    required_base_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido',
                          'fechaRegistro', 'fechaTRansaccion', 'credito', 'reglaCobranza',
                          'cuotaEsperada', 'totalTrans', 'saldoCapitalActual',
                          'totalDesembolso', 'cobranzaTrans', 'categoriaProductoCrediticio']
    
    missing_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {', '.join(missing_cols)}")
        st.error(f"Faltan las siguientes columnas en la hoja 'HistoricoPagoCuotas': {', '.join(missing_cols)}")
        return None
    
    date_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido',
                 'fechaRegistro', 'fechaTRansaccion']
    df = validate_dates(df, date_cols)
    
    df['credito'] = df['credito'].astype(str)
    
    grace_period = pd.Timedelta(days=grace_period_days)
    
    mask_valid_dates_late_payment = df['fechaPagoRecibido'].notna() & df['fechaEsperadaPago'].notna()
    df['late_payment'] = 0
    df.loc[mask_valid_dates_late_payment, 'late_payment'] = (
        df.loc[mask_valid_dates_late_payment, 'fechaPagoRecibido'] > 
        (df.loc[mask_valid_dates_late_payment, 'fechaEsperadaPago'] + grace_period)
    ).astype(int)
    
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
        weights['payment_coverage_ratio'] * (1 - df_for_return['payment_coverage_ratio_scaled']) + # Higher coverage is better, so 1-x
        weights['outstanding_balance_ratio'] * df_for_return['outstanding_balance_ratio_scaled'] +
        weights['collection_activity_count'] * df_for_return['collection_activity_count_scaled']
    )
    
    cols_to_return = ['credito', 'risk_score', 'late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']
    return df_for_return[cols_to_return]

# --- Utility Function for Outlier Detection ---
def get_outliers_iqr(df, column_name):
    if df is None or df.empty or column_name not in df.columns or df[column_name].isnull().all():
        logger.warning("get_outliers_iqr: Invalid input.")
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
    """Identifies outliers using Z-score method and returns low/high outliers and bounds."""
    if df is None or df.empty or column_name not in df.columns or df[column_name].isnull().all():
        logger.warning("get_outliers_zscore: Invalid input.")
        return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
    
    col_data = df[column_name].dropna()
    if col_data.empty:
        return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
        
    mean = col_data.mean()
    std_dev = col_data.std()
    
    if std_dev == 0:
        return pd.DataFrame(), pd.DataFrame(), mean, mean

    lower_bound = mean - threshold * std_dev
    upper_bound = mean + threshold * std_dev
    
    low_outliers = df.loc[col_data[col_data < lower_bound].index]
    high_outliers = df.loc[col_data[col_data > upper_bound].index]
    
    return low_outliers, high_outliers, lower_bound, upper_bound

# --- Helper function for Cramer's V ---
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return 0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if r == 1 or k == 1:
        return 0
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if rcorr < 1 or kcorr < 1: # Defensive check for small n
        return 0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# --- Data Preparation for PRE-LOAN FEATURE INSIGHTS TAB (Tab 4) ---
@st.cache_data
def prepare_preloan_insights_data(risk_df_with_components, listado_df, id_col_listado, target_col_name_in_risk_scores, selected_features_from_listado, bin_target_flag, num_bins_for_target=3):
    if risk_df_with_components is None or risk_df_with_components.empty or listado_df is None or listado_df.empty:
        logger.error("prepare_preloan_insights_data: Risk scores or customer data not available.")
        return None, None, None, None, None, "Las puntuaciones de riesgo o los datos del cliente no están disponibles."
    if id_col_listado not in listado_df.columns:
        logger.error(f"prepare_preloan_insights_data: ID column '{id_col_listado}' not found in customer data.")
        return None, None, None, None, None, f"La columna de ID '{id_col_listado}' no se encontró en los datos del cliente."
    if target_col_name_in_risk_scores not in risk_df_with_components.columns:
        logger.error(f"prepare_preloan_insights_data: Target column '{target_col_name_in_risk_scores}' not found in risk scores.")
        return None, None, None, None, None, f"La columna objetivo '{target_col_name_in_risk_scores}' no se encontró en las puntuaciones de riesgo."
    listado_copy = listado_df.copy()
    risk_scores_copy = risk_df_with_components.copy()
    listado_copy[id_col_listado] = listado_copy[id_col_listado].astype(str)
    risk_scores_copy['credito'] = risk_scores_copy['credito'].astype(str)
    merged_df = pd.merge(
        listado_copy,
        risk_scores_copy[['credito', target_col_name_in_risk_scores]],
        left_on=id_col_listado,
        right_on='credito',
        how='inner'
    )
    if merged_df.empty:
        logger.error("prepare_preloan_insights_data: No matching records found after merging.")
        return None, None, None, None, None, "No se encontraron registros coincidentes después de la fusión."
    
    features_to_analyze = [f for f in selected_features_from_listado if f in merged_df.columns and f not in [target_col_name_in_risk_scores, id_col_listado, 'credito']]
    if not features_to_analyze:
        logger.error("prepare_preloan_insights_data: No valid features selected/found from ListadoCreditos for analysis.")
        return None, None, None, None, None, "No se seleccionaron/encontraron variables válidas de ListadoCreditos para el análisis."
    
    analysis_subset_df = merged_df[features_to_analyze + [target_col_name_in_risk_scores]].copy()
    y_series = analysis_subset_df[target_col_name_in_risk_scores].copy()
    final_target_name = target_col_name_in_risk_scores
    if bin_target_flag:
        if y_series.nunique() <= 1:
            logger.error(f"prepare_preloan_insights_data: Target '{target_col_name_in_risk_scores}' has <=1 unique value, cannot bin.")
            return None, None, None, None, None, f"La variable objetivo '{target_col_name_in_risk_scores}' tiene 1 o menos valores únicos, no se puede agrupar."
        try:
            discretizer = KBinsDiscretizer(n_bins=num_bins_for_target, encode='ordinal', strategy='quantile', subsample=None)
            binned_target_values = discretizer.fit_transform(y_series.to_frame())
            y_series = pd.Series(binned_target_values.ravel().astype(int).astype(str), index=y_series.index)
            final_target_name = target_col_name_in_risk_scores + '_binned'
            y_series.name = final_target_name
        except ValueError as e:
            logger.error(f"prepare_preloan_insights_data: Error binning target '{target_col_name_in_risk_scores}': {e}.")
            return None, None, None, None, None, f"Error al agrupar la variable objetivo '{target_col_name_in_risk_scores}': {e}."
    
    x_features_df = pd.DataFrame(index=analysis_subset_df.index)
    original_dtypes = {}
    for feature in features_to_analyze:
        original_dtypes[feature] = analysis_subset_df[feature].dtype
        if pd.api.types.is_numeric_dtype(analysis_subset_df[feature]):
            x_features_df[feature] = analysis_subset_df[feature].fillna(analysis_subset_df[feature].median())
        elif pd.api.types.is_object_dtype(analysis_subset_df[feature]) or pd.api.types.is_categorical_dtype(analysis_subset_df[feature]):
            mode_val = analysis_subset_df[feature].mode()
            impute_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            x_features_df[feature] = analysis_subset_df[feature].fillna(impute_val)
        else: # Handle other types like dates gracefully
            x_features_df[feature] = analysis_subset_df[feature].astype(str).fillna("Unknown")
            
    return x_features_df, y_series, features_to_analyze, original_dtypes, final_target_name, None

# --- Data Preparation for FEATURE MI RANKER TAB (Tab 6) ---
@st.cache_data
def prepare_mi_data(risk_scores_data, listado_data, id_col_listado, target_col_name_in_risk_scores, selected_features_from_listado, bin_target_flag, num_bins_for_target=3):
    if risk_scores_data is None or risk_scores_data.empty or listado_data is None or listado_data.empty:
        return None, None, None, None, "Las puntuaciones de riesgo o los datos del cliente no están disponibles para el cálculo de MI."
    if id_col_listado not in listado_data.columns:
        return None, None, None, None, f"La columna de ID '{id_col_listado}' no se encontró en los datos del cliente."
    if target_col_name_in_risk_scores not in risk_scores_data.columns:
        return None, None, None, None, f"La columna objetivo '{target_col_name_in_risk_scores}' no se encontró en las puntuaciones de riesgo."
    
    listado_copy = listado_data.copy()
    risk_scores_copy = risk_scores_data.copy()
    listado_copy[id_col_listado] = listado_copy[id_col_listado].astype(str)
    risk_scores_copy['credito'] = risk_scores_copy['credito'].astype(str)
    
    merged_df_for_mi = pd.merge(listado_copy, risk_scores_copy[['credito', target_col_name_in_risk_scores]], left_on=id_col_listado, right_on='credito', how='inner')
    
    if merged_df_for_mi.empty:
        return None, None, None, None, "No se encontraron registros coincidentes después de la fusión."
        
    features_to_process = [f for f in selected_features_from_listado if f in merged_df_for_mi.columns]
    if not features_to_process:
        return None, None, None, None, "No se seleccionaron/encontraron variables válidas de ListadoCreditos para MI."
        
    X_mi = pd.DataFrame(index=merged_df_for_mi.index)
    discrete_mask = []
    processed_feature_names_ordered = []
    
    for feature in features_to_process:
        col_data = merged_df_for_mi[feature].copy()
        if pd.api.types.is_numeric_dtype(col_data):
            X_mi[feature] = col_data.fillna(col_data.median())
            discrete_mask.append(col_data.nunique(dropna=False) < 20)
            processed_feature_names_ordered.append(feature)
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            mode_val = col_data.mode()
            impute_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            col_data_filled = col_data.fillna(impute_val).astype(str)
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_mi[feature] = encoder.fit_transform(col_data_filled.to_frame())
            discrete_mask.append(True)
            processed_feature_names_ordered.append(feature)
        else:
            logger.warning(f"prepare_mi_data: Skipping feature '{feature}' for MI due to unhandled data type: {col_data.dtype}")
            continue
            
    if X_mi.empty or not processed_feature_names_ordered:
        return None, None, None, None, "No se procesaron variables para MI."
        
    y_mi_target_name = target_col_name_in_risk_scores
    y_mi = merged_df_for_mi[target_col_name_in_risk_scores].copy()
    
    if bin_target_flag:
        if y_mi.nunique() <= 1:
            return None, None, None, None, f"La variable objetivo '{y_mi_target_name}' tiene 1 o menos valores únicos, no se puede agrupar."
        try:
            discretizer = KBinsDiscretizer(n_bins=num_bins_for_target, encode='ordinal', strategy='quantile', subsample=None)
            y_mi_binned = discretizer.fit_transform(y_mi.to_frame())
            y_mi = pd.Series(y_mi_binned.ravel().astype(int), index=y_mi.index)
        except ValueError as e:
            return None, None, None, None, f"Error al agrupar la variable objetivo '{y_mi_target_name}' para MI: {e}."
    else:
        y_mi_numeric = pd.to_numeric(y_mi, errors='coerce')
        y_mi = y_mi_numeric.fillna(y_mi_numeric.median())
        
    return X_mi[processed_feature_names_ordered], y_mi, processed_feature_names_ordered, discrete_mask, None


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🏍️ App de Riesgo Crediticio y Análisis de Datos para Préstamos de Motocicleta")

risk_score_component_names = ['late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']
st.sidebar.header("⚙️ Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar Archivo Excel", type=["xlsx"])

default_grace_period = 5
grace_period_input = st.sidebar.number_input("Período de Gracia (días)", min_value=0, max_value=30, value=default_grace_period, step=1)

st.sidebar.subheader("Ponderación de Indicadores")
w_late_payment = st.sidebar.slider("Peso de Ratio de Pagos Atrasados", 0.0, 1.0, 0.40, 0.01)
w_payment_coverage = st.sidebar.slider("Peso de Ratio de Cobertura de Pago", 0.0, 1.0, 0.40, 0.01)
w_outstanding_balance = st.sidebar.slider("Peso de Ratio de Saldo Pendiente", 0.0, 1.0, 0.20, 0.01)
w_collection_activity = st.sidebar.slider("Peso de Actividad de Cobranza", 0.0, 1.0, 0.00, 0.01)
user_weights = {'late_payment_ratio': w_late_payment, 'payment_coverage_ratio': w_payment_coverage, 'outstanding_balance_ratio': w_outstanding_balance, 'collection_activity_count': w_collection_activity}

risk_scores_df, listado_creditos_df = None, None
processed_data_info, historico_pago_cuotas_loaded, listado_creditos_loaded = "", False, False
numero_credito_col_name = "numeroCredito"

if uploaded_file:
    st.sidebar.info(f"Procesando: {uploaded_file.name}")
    try:
        hpc_df = pd.read_excel(uploaded_file, sheet_name="HistoricoPagoCuotas")
        processed_data_info += "✅ Hoja 'HistoricoPagoCuotas' cargada.\n"
        historico_pago_cuotas_loaded = True
        if 'categoriaProductoCrediticio' in hpc_df.columns:
            moto_df = hpc_df[hpc_df["categoriaProductoCrediticio"] == "MOTOS"].copy()
            if not moto_df.empty:
                processed_data_info += f"Se encontraron {len(moto_df)} registros de 'MOTOS'.\n"
                with st.spinner("Calculando puntuaciones de riesgo y componentes..."):
                    risk_scores_df = calculate_risk_score_df(moto_df, grace_period_input, user_weights)
                if risk_scores_df is not None and not risk_scores_df.empty:
                    processed_data_info += f"✅ Puntuaciones y componentes calculados para {len(risk_scores_df)} créditos.\n"
                else:
                    processed_data_info += "⚠️ Problemas en el cálculo de la puntuación de riesgo.\n"
            else:
                processed_data_info += "⚠️ No se encontraron datos para 'MOTOS'.\n"
        else:
            processed_data_info += "❌ Falta la columna 'categoriaProductoCrediticio'.\n"
            historico_pago_cuotas_loaded = False
    except Exception as e:
        processed_data_info += f"❌ Error en 'HistoricoPagoCuotas': {e}\n"
        historico_pago_cuotas_loaded = False

    try:
        lc_df_temp = pd.read_excel(uploaded_file, sheet_name="ListadoCreditos")
        processed_data_info += f"✅ Hoja 'ListadoCreditos' cargada ({lc_df_temp.shape[0]}f, {lc_df_temp.shape[1]}c).\n"
        listado_creditos_loaded = True
        
        if 'fechaNacimiento' in lc_df_temp.columns:
            birth_dates = pd.to_datetime(lc_df_temp['fechaNacimiento'], errors='coerce')
            valid_birth_dates_mask = birth_dates.notna()
            if valid_birth_dates_mask.any():
                lc_df_temp.loc[valid_birth_dates_mask, 'age'] = ((pd.Timestamp.now() - birth_dates[valid_birth_dates_mask]).dt.days / 365.25)
                median_age = lc_df_temp['age'].median()
                if pd.notna(median_age):
                    lc_df_temp['age'].fillna(median_age, inplace=True)
                lc_df_temp['age'] = lc_df_temp['age'].astype(int)
                processed_data_info += "✅ Columna 'age' (edad) calculada desde 'fechaNacimiento'.\n"
            else:
                processed_data_info += "⚠️ 'fechaNacimiento' no contiene fechas válidas. No se creó la columna 'age' (edad).\n"
        else:
            processed_data_info += "⚠️ No se encontró 'fechaNacimiento'. No se calculó la columna 'age' (edad).\n"
            
        if numero_credito_col_name in lc_df_temp.columns:
            lc_df_temp[numero_credito_col_name] = lc_df_temp[numero_credito_col_name].astype(str)
            listado_creditos_df = lc_df_temp
            processed_data_info += f"La columna '{numero_credito_col_name}' se convirtió a texto.\n"
        else:
            processed_data_info += f"⚠️ Falta la columna '{numero_credito_col_name}'.\n"
            listado_creditos_df = lc_df_temp # Still assign it for DQA tab
    except Exception as e:
        processed_data_info += f"❌ Error en 'ListadoCreditos': {e}\n"
        listado_creditos_loaded = False
    
    st.sidebar.text_area("Registro de Procesamiento del Archivo", processed_data_info, height=250)
else:
    st.info("☝️ Cargue un archivo Excel para comenzar.")

tab_titles = ["📊 Puntuaciones de Riesgo", "📈 EDA del Riesgo", "🕵️ Análisis de Outliers", "📋 Calidad de Datos del Cliente", "🔍 Análisis Pre-Préstamo", "📊 Desempeño por Segmento", "ℹ️ Ranking de Variables (MI)"]
tabs = st.tabs(tab_titles)

with tabs[0]: # Puntuaciones de Riesgo
    st.header(tab_titles[0])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Puntuaciones de Riesgo y Componentes Calculados (por Crédito)")
        display_cols_scores = ['credito', 'risk_score'] + risk_score_component_names
        style_format_dict_tab0 = {"risk_score": "{:.4f}", "late_payment_ratio": "{:.4f}", "payment_coverage_ratio": "{:.4f}", "outstanding_balance_ratio": "{:.4f}", "collection_activity_count": "{:.0f}"}
        st.dataframe(risk_scores_df[display_cols_scores].style.format(style_format_dict_tab0), height=500, use_container_width=True)
        
        output_tab0 = io.BytesIO()
        with pd.ExcelWriter(output_tab0, engine='xlsxwriter') as writer_tab0:
            risk_scores_df.to_excel(writer_tab0, index=False, sheet_name='RiskScoresAndComponents')
        excel_data_tab0 = output_tab0.getvalue()
        
        if excel_data_tab0:
            st.download_button(label="Descargar Puntuaciones y Componentes", data=excel_data_tab0, file_name=f"puntuaciones_riesgo_componentes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("No se pudo generar el archivo Excel para descargar.")
    elif uploaded_file and historico_pago_cuotas_loaded:
        st.warning("No se calcularon las puntuaciones de riesgo. Revise el registro en la barra lateral.")
    elif uploaded_file and not historico_pago_cuotas_loaded:
        st.error("Falló la carga de 'HistoricoPagoCuotas'. Revise el registro en la barra lateral.")
    else:
        st.write("Cargue un archivo para ver los resultados.")

with tabs[1]: # EDA del Riesgo
    st.header(tab_titles[1])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Distribución de la Puntuación de Riesgo")
        st.dataframe(risk_scores_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(risk_scores_df['risk_score'], kde=True, ax=ax, bins=20)
            ax.set_title('Histograma de la Puntuación de Riesgo')
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(y=risk_scores_df['risk_score'], ax=ax)
            ax.set_title('Boxplot de la Puntuación de Riesgo')
            st.pyplot(fig)
            plt.close(fig)
    elif uploaded_file:
        st.warning("Las puntuaciones de riesgo no están disponibles para el EDA. Revise el registro de procesamiento.")
    else:
        st.write("Cargue un archivo para realizar el EDA.")

with tabs[2]: # Análisis de Outliers
    st.header(tab_titles[2])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Configuración de Detección de Outliers")
        outlier_method = st.selectbox(
            "Seleccione el Método de Detección de Outliers:",
            ("Método IQR", "Método Z-score"),
            key="outlier_method_select"
        )
        
        low_o, high_o, lb, ub = pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
        title_text = ""
        
        if outlier_method == "Método IQR":
            title_text = "Identificación de Outliers en Puntuación de Riesgo (usando IQR)"
            st.info("El método IQR (Rango Intercuartílico) define como outliers las observaciones que caen por debajo de Q1 - 1.5*IQR o por encima de Q3 + 1.5*IQR.")
            low_o, high_o, lb, ub = get_outliers_iqr(risk_scores_df, 'risk_score')
        
        elif outlier_method == "Método Z-score":
            z_threshold = st.number_input("Umbral de Z-score:", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="z_threshold_input")
            title_text = f"Identificación de Outliers en Puntuación de Riesgo (usando Z-score, Umbral={z_threshold})"
            st.info(f"El método Z-score define como outliers los puntos de datos con un Z-score absoluto mayor que el umbral ({z_threshold}). El Z-score mide cuántas desviaciones estándar se aleja una observación de la media.")
            low_o, high_o, lb, ub = get_outliers_zscore(risk_scores_df, 'risk_score', threshold=z_threshold)

        st.markdown("---")
        st.subheader(title_text)
        if not np.isnan(lb):
            st.write(f"**Límite Inferior:** `{lb:.4f}` | **Límite Superior:** `{ub:.4f}`")
        else:
            st.write("No se pueden determinar los límites para outliers (p. ej., no hay varianza en los datos).")
            
        st.markdown("---")
        st.subheader("Outliers de Alto Riesgo (Puntuación > Límite Superior)")
        if not high_o.empty:
            st.write(f"Encontrados: {len(high_o)} outliers de alto riesgo.")
            st.dataframe(high_o['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        else:
            st.write("No se encontraron outliers de alto riesgo.")
            
        st.markdown("---")
        st.subheader("Outliers de Bajo Riesgo (Puntuación < Límite Inferior)")
        if not low_o.empty:
            st.write(f"Encontrados: {len(low_o)} outliers de bajo riesgo.")
            st.dataframe(low_o['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        else:
            st.write("No se encontraron outliers de bajo riesgo.")
            
        st.markdown("---")
        st.subheader("Descargar Detalles de Outliers")
        if listado_creditos_loaded and listado_creditos_df is not None and numero_credito_col_name in listado_creditos_df.columns:
            if not high_o.empty or not low_o.empty:
                output_o = io.BytesIO()
                with pd.ExcelWriter(output_o, engine='xlsxwriter') as writer:
                    if not high_o.empty:
                        high_o_details = listado_creditos_df.merge(high_o, left_on=numero_credito_col_name, right_on='credito', how='inner')
                        high_o_details.to_excel(writer, sheet_name='High Risk', index=False)
                    if not low_o.empty:
                        low_o_details = listado_creditos_df.merge(low_o, left_on=numero_credito_col_name, right_on='credito', how='inner')
                        low_o_details.to_excel(writer, sheet_name='Low Risk', index=False)
                excel_data_outlier = output_o.getvalue()
                if excel_data_outlier:
                    st.download_button(label="Descargar Detalles de Outliers", data=excel_data_outlier, file_name=f"outliers_detallado_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("No se pudo generar el archivo Excel con detalles de outliers.")
            else:
                st.info("No hay outliers para descargar.")
        elif uploaded_file:
            st.warning(f"No se pueden generar los detalles de outliers. Verifique si la hoja 'ListadoCreditos' y la columna '{numero_credito_col_name}' están presentes y se cargaron correctamente.")
    elif uploaded_file:
        st.warning("Las puntuaciones de riesgo no están disponibles. No se puede realizar el análisis de outliers.")
    else:
        st.write("Cargue un archivo para realizar el análisis.")

with tabs[3]: # Calidad de Datos del Cliente
    st.header(tab_titles[3] + " (Hoja 'ListadoCreditos')")
    if not uploaded_file:
        st.write("Cargue un archivo para el Análisis de Calidad de Datos (DQA).")
    elif not listado_creditos_loaded or listado_creditos_df is None:
        st.error("La hoja 'ListadoCreditos' no se cargó. Revise el registro en la barra lateral.")
    else:
        df_dqa = listado_creditos_df
        st.subheader("1. Resumen General")
        st.write(f"Filas: {df_dqa.shape[0]}, Columnas: {df_dqa.shape[1]}")
        with st.expander("Tipos de Datos"):
            st.dataframe(df_dqa.dtypes.reset_index().rename(columns={'index':'Columna',0:'Tipo'}))
            
        st.subheader("2. Valores Faltantes")
        missing_sum = df_dqa.isnull().sum().reset_index()
        missing_sum.columns=['Columna','Cantidad Faltante']
        missing_sum['% Faltante']=(missing_sum['Cantidad Faltante']/len(df_dqa))*100
        missing_sum = missing_sum[missing_sum['Cantidad Faltante']>0].sort_values(by='% Faltante',ascending=False)
        if not missing_sum.empty:
            st.dataframe(missing_sum.style.format({'% Faltante':"{:.2f}%"}))
        else:
            st.success("¡No se encontraron valores faltantes! 🎉")
            
        st.subheader("3. Duplicados")
        st.write(f"Duplicados de fila completa: {df_dqa.duplicated().sum()}")
        if numero_credito_col_name in df_dqa.columns:
            st.write(f"Duplicados en la columna de ID ('{numero_credito_col_name}'): {df_dqa.duplicated(subset=[numero_credito_col_name]).sum()}")
            
        st.subheader("4. Análisis por Columna")
        default_dqa_col = [df_dqa.columns[0]] if len(df_dqa.columns) > 0 else []
        cols_detail = st.multiselect("Seleccione columnas para un análisis detallado:", options=df_dqa.columns.tolist(), default=default_dqa_col)
        for col in cols_detail:
            with st.expander(f"Análisis de '{col}' (Tipo: {df_dqa[col].dtype})"):
                st.write(f"Valores Únicos: {df_dqa[col].nunique()}, Faltantes: {df_dqa[col].isnull().sum()} ({df_dqa[col].isnull().sum()/len(df_dqa)*100:.2f}%)")
                if pd.api.types.is_numeric_dtype(df_dqa[col]):
                    st.dataframe(df_dqa[col].describe().to_frame().T)
                elif pd.api.types.is_object_dtype(df_dqa[col]):
                    st.dataframe(df_dqa[col].value_counts().nlargest(10).reset_index().rename(columns={'index':'Valor', col:'Conteo'}))

with tabs[4]: # Análisis Pre-Préstamo
    st.header(tab_titles[4])
    if not uploaded_file:
        st.write("Cargue un archivo Excel y asegúrese de que 'ListadoCreditos' y las puntuaciones de riesgo se procesen.")
    elif listado_creditos_df is None or risk_scores_df is None or risk_scores_df.empty:
        st.warning("Los datos del cliente ('ListadoCreditos') o las Puntuaciones de Riesgo no están disponibles. Por favor, revise las pestañas/registros anteriores.")
    else:
        st.subheader("Configuración para Análisis de Variables Pre-Préstamo")
        target_options_prev_tab = ["Puntuación de Riesgo Bruta (Continua)", "Puntuación de Riesgo Agrupada (Categórica)"]
        chosen_target_type_prev_tab = st.selectbox("¿Cómo tratar la Puntuación de Riesgo para el análisis (Pre-Préstamo)?", target_options_prev_tab, index=0, key="fi_target_type")
        num_bins_fi_prev_tab = 3
        if "Agrupada" in chosen_target_type_prev_tab:
            num_bins_fi_prev_tab = st.slider("Número de grupos para la Puntuación de Riesgo (Pre-Préstamo):", 2, 10, 3, 1, key="fi_bins")
        
        available_features_prev_tab = [col for col in listado_creditos_df.columns if col not in [numero_credito_col_name]]
        if not available_features_prev_tab:
            st.error("No se encontraron variables en 'ListadoCreditos' para analizar.")
        else:
            default_fi_col_prev_tab = [available_features_prev_tab[0]] if available_features_prev_tab else []
            selected_cols_fi_prev_tab = st.multiselect("Seleccione variables para el Análisis Pre-Préstamo:", options=available_features_prev_tab, default=default_fi_col_prev_tab, key="fi_cols")
            
            if st.button("🚀 Analizar Variables Pre-Préstamo", key="fi_analyze_button"):
                if not selected_cols_fi_prev_tab:
                    st.warning("Por favor, seleccione al menos una variable para el análisis.")
                else:
                    with st.spinner("Preparando datos y realizando análisis de variables pre-préstamo..."):
                        prep_result_tab4 = prepare_preloan_insights_data(
                            risk_scores_df, listado_creditos_df, numero_credito_col_name,
                            'risk_score', selected_cols_fi_prev_tab,
                            "Agrupada" in chosen_target_type_prev_tab, num_bins_fi_prev_tab
                        )
                        features_for_analysis_df_tab4, target_series_tab4, actual_features_analyzed_tab4, original_feature_dtypes_tab4, final_target_name_tab4, error_message_tab4 = prep_result_tab4
                        
                        if error_message_tab4:
                            st.error(f"Error de Preparación de Datos (Pre-Préstamo): {error_message_tab4}")
                        elif features_for_analysis_df_tab4 is None or target_series_tab4 is None:
                            st.error("Falló la preparación de datos para el Análisis Pre-Préstamo. Revise los registros.")
                        else:
                            st.success(f"Datos preparados. Analizando {len(actual_features_analyzed_tab4)} variables contra '{final_target_name_tab4}'.")
                            
                            if "Bruta" in chosen_target_type_prev_tab:
                                st.markdown("---")
                                st.subheader("A. Correlación con la Puntuación de Riesgo Bruta")
                                numeric_features_to_correlate_tab4 = [f for f in actual_features_analyzed_tab4 if pd.api.types.is_numeric_dtype(original_feature_dtypes_tab4.get(f)) and f in selected_cols_fi_prev_tab]
                                if numeric_features_to_correlate_tab4:
                                    correlations_tab4 = {feat: features_for_analysis_df_tab4[feat].corr(target_series_tab4) for feat in numeric_features_to_correlate_tab4}
                                    corr_df_tab4 = pd.DataFrame.from_dict(correlations_tab4, orient='index', columns=['Correlación de Pearson']).dropna()
                                    if not corr_df_tab4.empty:
                                        corr_df_display_tab4 = corr_df_tab4.copy()
                                        corr_df_display_tab4['Correlación Absoluta'] = corr_df_display_tab4['Correlación de Pearson'].abs()
                                        corr_df_display_tab4 = corr_df_display_tab4.sort_values(by='Correlación Absoluta', ascending=False)
                                        st.write("Tabla de Correlación:")
                                        st.dataframe(corr_df_display_tab4[['Correlación de Pearson']].style.format("{:.3f}"))
                                        st.subheader("Gráficos de Dispersión")
                                        for feat_name in corr_df_display_tab4.index:
                                            fig_corr, ax_corr = plt.subplots()
                                            sns.regplot(x=features_for_analysis_df_tab4[feat_name], y=target_series_tab4, ax=ax_corr, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                                            ax_corr.set_title(f"Dispersión: {feat_name} vs. {final_target_name_tab4}")
                                            st.pyplot(fig_corr)
                                            plt.close(fig_corr)
                                    else: st.info("No se pudieron calcular correlaciones válidas.")
                                else: st.info("No se seleccionaron variables numéricas para el análisis de correlación.")
                            
                            st.markdown("---")
                            st.subheader("B. Comparaciones por Grupo")
                            categorical_features_to_group_tab4 = [f for f in actual_features_analyzed_tab4 if (not pd.api.types.is_numeric_dtype(original_feature_dtypes_tab4.get(f)) or features_for_analysis_df_tab4[f].nunique() < 20) and f in selected_cols_fi_prev_tab]
                            if categorical_features_to_group_tab4:
                                results_groupwise_tab4 = []
                                for feat in categorical_features_to_group_tab4:
                                    if features_for_analysis_df_tab4[feat].nunique() < 2 or features_for_analysis_df_tab4[feat].nunique() > 50:
                                        continue
                                    
                                    plot_target_groupwise_tab4 = pd.to_numeric(target_series_tab4, errors='coerce') if "Bruta" in chosen_target_type_prev_tab else target_series_tab4
                                    temp_df = pd.DataFrame({'feature': features_for_analysis_df_tab4[feat].astype(str), 'target': plot_target_groupwise_tab4}).dropna()

                                    if not temp_df.empty:
                                        order = temp_df.groupby('feature')['target'].median().sort_values().index if "Bruta" in chosen_target_type_prev_tab else sorted(temp_df['feature'].unique())
                                        fig_box_fi, ax_box_fi = plt.subplots()
                                        sns.boxplot(x='feature', y='target', data=temp_df, ax=ax_box_fi, order=order)
                                        ax_box_fi.set_title(f"{final_target_name_tab4} por {feat}")
                                        ax_box_fi.set_xlabel(feat)
                                        ax_box_fi.set_ylabel(final_target_name_tab4)
                                        ax_box_fi.tick_params(axis='x', rotation=45)
                                        plt.tight_layout()
                                        st.pyplot(fig_box_fi)
                                        plt.close(fig_box_fi)

                                    if "Agrupada" in chosen_target_type_prev_tab:
                                        ct = pd.crosstab(features_for_analysis_df_tab4[feat], target_series_tab4)
                                        if ct.shape[0] > 1 and ct.shape[1] > 1:
                                            chi2, p, _, _ = chi2_contingency(ct)
                                            cv = cramers_v(ct.values)
                                            results_groupwise_tab4.append({'Variable': feat, 'Prueba': 'Chi-Cuadrado', 'Estadístico': chi2, 'valor-p': p, "V de Cramer": cv})
                                    else:
                                        groups = [g for _, g in temp_df.groupby('feature')['target'] if len(g) > 1]
                                        if len(groups) > 1:
                                            f_stat, p_anova = f_oneway(*groups)
                                            results_groupwise_tab4.append({'Variable': feat, 'Prueba': 'ANOVA', 'Estadístico': f_stat, 'valor-p': p_anova, "V de Cramer": np.nan})
                                
                                if results_groupwise_tab4:
                                    st.dataframe(pd.DataFrame(results_groupwise_tab4).style.format({'Estadístico': "{:.3f}", 'valor-p': "{:.3g}", "V de Cramer": "{:.3f}"}))
                                else: st.info("No se pudieron ejecutar pruebas estadísticas para las variables categóricas seleccionadas.")
                            else: st.info("No se seleccionaron variables categóricas para la comparación por grupos.")
                            
                            st.markdown("---")
                            st.subheader("C. Información Mutua")
                            mi_features_to_analyze_tab4 = [f for f in actual_features_analyzed_tab4 if f in selected_cols_fi_prev_tab]
                            if mi_features_to_analyze_tab4:
                                mi_features_df_tab4 = pd.DataFrame(index=features_for_analysis_df_tab4.index)
                                discrete_mask_tab4 = []
                                for feat in mi_features_to_analyze_tab4:
                                    if not pd.api.types.is_numeric_dtype(original_feature_dtypes_tab4.get(feat)):
                                        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                                        mi_features_df_tab4[feat] = le.fit_transform(features_for_analysis_df_tab4[[feat]].astype(str))
                                        discrete_mask_tab4.append(True)
                                    else:
                                        mi_features_df_tab4[feat] = features_for_analysis_df_tab4[feat]
                                        discrete_mask_tab4.append(mi_features_df_tab4[feat].nunique() < 20)
                                if not mi_features_df_tab4.empty:
                                    mi_target_tab4 = target_series_tab4.astype(int) if "Agrupada" in chosen_target_type_prev_tab else pd.to_numeric(target_series_tab4, errors='coerce').fillna(target_series_tab4.median())
                                    mi_func_tab4 = mutual_info_classif if "Agrupada" in chosen_target_type_prev_tab else mutual_info_regression
                                    effective_discrete_mask_tab4 = discrete_mask_tab4 if len(discrete_mask_tab4) == mi_features_df_tab4.shape[1] else 'auto'
                                    mi_scores_tab4 = mi_func_tab4(mi_features_df_tab4, mi_target_tab4, discrete_features=effective_discrete_mask_tab4, random_state=42)
                                    mi_df_tab4 = pd.DataFrame({'Variable': mi_features_to_analyze_tab4, 'IM': mi_scores_tab4}).sort_values(by='IM', ascending=False)
                                    st.dataframe(mi_df_tab4.style.format({'IM': "{:.4f}"}))
                                    if not mi_df_tab4.empty:
                                        fig_mi, ax_mi = plt.subplots(figsize=(10, max(5, len(mi_df_tab4)*0.3)))
                                        sns.barplot(x='IM', y='Variable', data=mi_df_tab4, ax=ax_mi)
                                        ax_mi.set_title("Información Mutua (Análisis Pre-Préstamo)")
                                        plt.tight_layout()
                                        st.pyplot(fig_mi)
                                        plt.close(fig_mi)
                                else: st.info("No hay variables disponibles para el cálculo de Información Mutua con su selección.")
                            else: st.info("No se seleccionaron variables para el análisis de Información Mutua.")

with tabs[5]: # Desempeño por Segmento
    st.header(tab_titles[5])
    if not uploaded_file:
        st.write("Cargue un archivo Excel y asegúrese de que 'ListadoCreditos' y las puntuaciones de riesgo se procesen.")
    elif listado_creditos_df is None or risk_scores_df is None or risk_scores_df.empty:
        st.warning("Los datos del cliente ('ListadoCreditos') o las Puntuaciones de Riesgo (con componentes) no están disponibles. Por favor, revise el procesamiento.")
    else:
        st.subheader("Definir Segmento de Clientes")
        temp_listado_df = listado_creditos_df.copy()
        temp_risk_scores_df = risk_scores_df.copy()
        temp_listado_df[numero_credito_col_name] = temp_listado_df[numero_credito_col_name].astype(str)
        temp_risk_scores_df['credito'] = temp_risk_scores_df['credito'].astype(str)
        required_risk_cols_for_segment = ['credito', 'risk_score'] + risk_score_component_names
        
        if not all(col in temp_risk_scores_df.columns for col in required_risk_cols_for_segment):
            st.error("Faltan columnas requeridas en los datos de puntuación de riesgo para este análisis.")
        else:
            segment_data_full = pd.merge(temp_listado_df, temp_risk_scores_df[required_risk_cols_for_segment], left_on=numero_credito_col_name, right_on='credito', how='inner')
            if segment_data_full.empty:
                st.warning("No se encontraron registros coincidentes entre los datos del cliente y las puntuaciones de riesgo para la segmentación.")
            else:
                demographic_cols = [col for col in listado_creditos_df.columns if col not in [numero_credito_col_name] + required_risk_cols_for_segment and col != 'credito']
                base_categorical_demographics = [col for col in demographic_cols if segment_data_full[col].dtype == 'object' or (segment_data_full[col].nunique() < 20 and col != 'age')]
                
                selectable_segment_vars = base_categorical_demographics.copy()
                if 'age' in segment_data_full.columns and 'age' not in selectable_segment_vars:
                    selectable_segment_vars.insert(0, 'age')

                if not selectable_segment_vars:
                    st.info("No se encontraron variables demográficas adecuadas para la segmentación en 'ListadoCreditos'.")
                else:
                    default_selection = [selectable_segment_vars[0]] if selectable_segment_vars else []
                    selected_segment_vars = st.multiselect("Seleccione variables demográficas para la segmentación:", options=selectable_segment_vars, default=default_selection)
                    
                    query_parts = []

                    for var in selected_segment_vars:
                        if var == 'age' and pd.api.types.is_numeric_dtype(segment_data_full[var]):
                            min_age = int(segment_data_full[var].min())
                            max_age = int(segment_data_full[var].max())
                            if min_age < max_age:
                                selected_age_range = st.slider(f"Seleccione el rango para '{var}':", min_age, max_age, (min_age, max_age))
                                query_parts.append(f"`{var}` >= {selected_age_range[0]} and `{var}` <= {selected_age_range[1]}")
                            else:
                                st.caption(f"Solo hay un valor ({min_age}) para '{var}', no se puede crear un deslizador de rango.")
                        else:
                            unique_levels = sorted(segment_data_full[var].dropna().unique().astype(str))
                            if unique_levels:
                                selected_levels = st.multiselect(f"Seleccione los niveles para '{var}':", options=unique_levels, default=unique_levels[:1])
                                if selected_levels:
                                    str_levels_for_query = [f"'{str(level).replace("'", "\\'")}'" for level in selected_levels]
                                    query_parts.append(f"`{var}` in ({', '.join(str_levels_for_query)})")
                            else:
                                st.caption(f"No hay niveles seleccionables para '{var}'.")
                    
                    segmented_df = segment_data_full.copy()
                    if query_parts:
                        try:
                            final_query = " and ".join(query_parts)
                            segmented_df = segmented_df.query(final_query)
                        except Exception as e:
                            st.error(f"Error al aplicar filtros: {e}. Revise los nombres de columnas/niveles por caracteres especiales.")
                            segmented_df = pd.DataFrame()
                                
                    if not segmented_df.empty:
                        st.markdown("---")
                        st.subheader(f"Desempeño del Segmento Seleccionado ({len(segmented_df)} préstamos)")
                        
                        summary_data = []
                        for metric in ['risk_score'] + risk_score_component_names:
                            summary_data.append({
                                'Métrica': metric,
                                'Promedio del Segmento': segmented_df[metric].mean(),
                                'Promedio General': risk_scores_df[metric].mean()
                            })
                        
                        comparison_df = pd.DataFrame(summary_data)
                        st.dataframe(comparison_df.set_index('Métrica').style.format("{:.4f}"))
                        
                    elif selected_segment_vars and not any(f.startswith(f"`{var}` in") for f in query_parts if var != 'age') and not any('age' in q for q in query_parts):
                         st.info("Por favor, seleccione niveles/rangos específicos para las variables demográficas elegidas para ver los resultados del segmento.")
                    elif selected_segment_vars and query_parts and segmented_df.empty:
                        st.info("No se encontraron clientes que coincidan con todos los criterios seleccionados.")
                    else:
                        st.info("Seleccione variables demográficas y sus niveles/rangos para analizar el desempeño del segmento.")

with tabs[6]: # Ranking de Variables (MI)
    st.header(tab_titles[6])
    if not uploaded_file:
        st.write("Cargue un archivo Excel y asegúrese de que 'ListadoCreditos' y las puntuaciones de riesgo se procesen.")
    elif listado_creditos_df is None or risk_scores_df is None or risk_scores_df.empty:
        st.warning("Los datos del cliente ('ListadoCreditos') o las Puntuaciones de Riesgo no están disponibles.")
    else:
        st.subheader("Configuración para Ranking de Información Mutua")
        mi_target_options = ["Puntuación de Riesgo Bruta (Continua)", "Puntuación de Riesgo Agrupada (Categórica)"]
        mi_chosen_target_type = st.selectbox("Tratar la Puntuación de Riesgo como:", mi_target_options, index=0, key="mi_ranker_target_type")
        mi_num_bins = 3
        if "Agrupada" in mi_chosen_target_type:
            mi_num_bins = st.slider("Número de grupos para la Puntuación de Riesgo (Ranking MI):", 2, 10, 3, 1, key="mi_ranker_bins")
        
        mi_available_features = [col for col in listado_creditos_df.columns if col not in [numero_credito_col_name]]
        if not mi_available_features:
            st.error("No hay variables disponibles en 'ListadoCreditos' para el ranking de MI.")
        else:
            default_mi_cols = mi_available_features[:min(5, len(mi_available_features))]
            selected_cols_for_mi = st.multiselect("Seleccione variables de 'ListadoCreditos' para el cálculo de MI:", options=mi_available_features, default=default_mi_cols, key="mi_ranker_features")
            
            if st.button("Calcular Información Mutua", key="mi_ranker_button"):
                if not selected_cols_for_mi:
                    st.warning("Por favor, seleccione al menos una variable para el cálculo de MI.")
                else:
                    with st.spinner("Preparando datos y calculando la Información Mutua..."):
                        X_mi_prepared, y_mi_prepared, processed_names_mi, discrete_feature_mask_mi, mi_error_message = prepare_mi_data(
                            risk_scores_df, listado_creditos_df, numero_credito_col_name, 'risk_score', 
                            selected_cols_for_mi, "Agrupada" in mi_chosen_target_type, mi_num_bins
                        )
                        
                        if mi_error_message:
                            st.error(f"Error en la Preparación de Datos para MI: {mi_error_message}")
                        elif X_mi_prepared is None or y_mi_prepared is None or not processed_names_mi:
                            st.error("Falló la preparación de datos para el cálculo de MI. Revise las variables seleccionadas o la integridad de los datos.")
                        else:
                            st.success(f"Datos preparados. Calculando MI para {len(processed_names_mi)} variables.")
                            mi_function_to_use = mutual_info_classif if "Agrupada" in mi_chosen_target_type else mutual_info_regression
                            effective_discrete_mask_mi = discrete_feature_mask_mi if len(discrete_feature_mask_mi) == X_mi_prepared.shape[1] else 'auto'
                            mi_scores_values = mi_function_to_use(X_mi_prepared, y_mi_prepared, discrete_features=effective_discrete_mask_mi, random_state=42)
                            
                            mi_results_df = pd.DataFrame({'Variable': processed_names_mi, 'Puntuación de Información Mutua': mi_scores_values}).sort_values(by='Puntuación de Información Mutua', ascending=False)
                            
                            st.subheader("Puntuaciones de Información Mutua con la Puntuación de Riesgo")
                            st.dataframe(mi_results_df.style.format({'Puntuación de Información Mutua': "{:.4f}"}))
                            
                            if not mi_results_df.empty:
                                fig_mi_ranker, ax_mi_ranker = plt.subplots(figsize=(10, max(5, len(mi_results_df) * 0.3)))
                                sns.barplot(x='Puntuación de Información Mutua', y='Variable', data=mi_results_df, ax=ax_mi_ranker, palette="viridis")
                                ax_mi_ranker.set_title("Ranking de Variables por MI con la Puntuación de Riesgo")
                                plt.tight_layout()
                                st.pyplot(fig_mi_ranker)
                                plt.close(fig_mi_ranker)

st.markdown("---")
st.markdown("App desarrollada por su Científico de Datos Experto, Antonio Medrano, CepSA")
