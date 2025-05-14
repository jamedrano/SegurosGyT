import streamlit as st
import pandas as pd
import numpy as np
import io # Required for Excel download
import matplotlib.pyplot as plt
import seaborn as sns

# --- Core Risk Score Calculation Logic (remains the same) ---
def calculate_risk_score_df(df_input, grace_period_days, weights):
    if df_input.empty:
        # st.warning("Input data is empty after initial filtering. Cannot calculate risk scores.") # Warning handled in main flow
        return None
    df = df_input.copy()
    required_cols = [
        'fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro',
        'fechaTRansaccion', 'credito', 'reglaCobranza', 'cuotaCubierta',
        'cuotaEsperada', 'saldoCapitalActual', 'totalDesembolso', 'cobranzaTrans',
        'categoriaProductoCrediticio'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"The uploaded file (sheet 'HistoricoPagoCuotas') is missing required columns for risk scoring: {', '.join(missing_cols)}")
        return None
    date_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro', 'fechaTRansaccion']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].isnull().any() and 'streamlit_is_running' not in globals():
                 print(f"Warning: Column '{col}' contains values that could not be converted to datetime and were set to NaT.")
    df['credito'] = df['credito'].astype(str)
    grace_period = pd.Timedelta(days=grace_period_days)
    mask_valid_dates_late_payment = df['fechaPagoRecibido'].notna() & df['fechaEsperadaPago'].notna()
    df['late_payment'] = 0
    df.loc[mask_valid_dates_late_payment, 'late_payment'] = \
        (df.loc[mask_valid_dates_late_payment, 'fechaPagoRecibido'] >
         (df.loc[mask_valid_dates_late_payment, 'fechaEsperadaPago'] + grace_period)).astype(int)
    late_payment_counts = df.groupby('credito')['late_payment'].sum()
    total_payments = df.groupby('credito')['fechaEsperadaPago'].count()
    late_payment_ratio = (late_payment_counts / total_payments.replace(0, np.nan)).fillna(0)
    df['cuotaCubierta'] = df['cuotaCubierta'].fillna(0)
    total_payment_made = df.groupby('credito')['cuotaCubierta'].sum()
    total_payment_expected = df.groupby('credito')['cuotaEsperada'].sum()
    payment_coverage_ratio = (total_payment_made / total_payment_expected.replace(0, np.nan))
    payment_coverage_ratio = payment_coverage_ratio.fillna(1)
    payment_coverage_ratio = payment_coverage_ratio.replace([np.inf, -np.inf], 1)
    df['saldoCapitalActual'] = pd.to_numeric(df['saldoCapitalActual'], errors='coerce')
    df['totalDesembolso'] = pd.to_numeric(df['totalDesembolso'], errors='coerce')
    last_saldo = df.groupby('credito')['saldoCapitalActual'].last()
    first_desembolso = df.groupby('credito')['totalDesembolso'].first()
    outstanding_balance_ratio = (last_saldo / first_desembolso.replace(0, np.nan))
    outstanding_balance_ratio = outstanding_balance_ratio.fillna(0)
    df['cobranzaTrans'] = pd.to_numeric(df['cobranzaTrans'], errors='coerce').fillna(0)
    df['collection_activity'] = (df['cobranzaTrans'] > 0).astype(int)
    collection_activity_count = df.groupby('credito')['collection_activity'].sum()
    creditos_unicos = pd.DataFrame(df['credito'].unique(), columns=['credito'])
    creditos_unicos = creditos_unicos.merge(late_payment_ratio.to_frame(name='late_payment_ratio'), left_on='credito', right_index=True, how='left')
    creditos_unicos = creditos_unicos.merge(payment_coverage_ratio.to_frame(name='payment_coverage_ratio'), left_on='credito', right_index=True, how='left')
    creditos_unicos = creditos_unicos.merge(outstanding_balance_ratio.to_frame(name='outstanding_balance_ratio'), left_on='credito', right_index=True, how='left')
    creditos_unicos = creditos_unicos.merge(collection_activity_count.to_frame(name='collection_activity_count'), left_on='credito', right_index=True, how='left')
    creditos_unicos['late_payment_ratio'] = creditos_unicos['late_payment_ratio'].fillna(0)
    creditos_unicos['payment_coverage_ratio'] = creditos_unicos['payment_coverage_ratio'].fillna(1)
    creditos_unicos['outstanding_balance_ratio'] = creditos_unicos['outstanding_balance_ratio'].fillna(0)
    creditos_unicos['collection_activity_count'] = creditos_unicos['collection_activity_count'].fillna(0)
    for col_name in ['late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']:
        min_val = creditos_unicos[col_name].min()
        max_val = creditos_unicos[col_name].max()
        scaled_col_name = f'{col_name}_scaled'
        if max_val == min_val:
            creditos_unicos[scaled_col_name] = 0.0 # if min=max, scaled is 0 (or 0.5 if not 0)
        else:
            creditos_unicos[scaled_col_name] = (creditos_unicos[col_name] - min_val) / (max_val - min_val)
        creditos_unicos[scaled_col_name] = creditos_unicos[scaled_col_name].fillna(0)
    creditos_unicos['risk_score'] = (
        weights['late_payment_ratio'] * creditos_unicos['late_payment_ratio_scaled'] +
        weights['payment_coverage_ratio'] * (1 - creditos_unicos['payment_coverage_ratio_scaled']) +
        weights['outstanding_balance_ratio'] * creditos_unicos['outstanding_balance_ratio_scaled'] +
        weights['collection_activity_count'] * creditos_unicos['collection_activity_count_scaled']
    )
    return creditos_unicos[['credito', 'risk_score']]

# --- Utility Function for Outlier Detection ---
def get_outliers_iqr(df, column_name):
    """Identifies outliers in a DataFrame column using the IQR method."""
    if df is None or df.empty or column_name not in df.columns or df[column_name].isnull().all():
        return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan # Return empty DFs and NaN bounds
    
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    # Handle cases where IQR is 0 (e.g., all values are the same)
    if IQR == 0:
        # No outliers if all values are the same and IQR is 0
        lower_bound = Q1 
        upper_bound = Q3
        low_outliers = df[df[column_name] < lower_bound] # Should be empty
        high_outliers = df[df[column_name] > upper_bound] # Should be empty
    else:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        low_outliers = df[df[column_name] < lower_bound]
        high_outliers = df[df[column_name] > upper_bound]
        
    return low_outliers, high_outliers, lower_bound, upper_bound

# --- Streamlit App UI ---
streamlit_is_running = True
st.set_page_config(layout="wide")
st.title("🏍️ Motorcycle Loan Risk Score Calculator")

# --- Sidebar for User Inputs (remains the same) ---
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Excel File (containing 'HistoricoPagoCuotas' and 'ListadoCreditos' sheets)", type=["xlsx"])
default_grace_period = 5
grace_period_input = st.sidebar.number_input("Grace Period (days)", min_value=0, max_value=30, value=default_grace_period, step=1)
st.sidebar.subheader("Indicator Weights")
w_late_payment = st.sidebar.slider("Late Payment Ratio Weight", 0.0, 1.0, 0.40, 0.01)
w_payment_coverage = st.sidebar.slider("Payment Coverage Ratio Weight", 0.0, 1.0, 0.40, 0.01)
w_outstanding_balance = st.sidebar.slider("Outstanding Balance Ratio Weight", 0.0, 1.0, 0.20, 0.01)
w_collection_activity = st.sidebar.slider("Collection Activity Count Weight", 0.0, 1.0, 0.00, 0.01)
user_weights = {
    'late_payment_ratio': w_late_payment,
    'payment_coverage_ratio': w_payment_coverage,
    'outstanding_balance_ratio': w_outstanding_balance,
    'collection_activity_count': w_collection_activity
}

# --- Initialize DataFrames ---
risk_scores_df = None
listado_creditos_df = None # For "ListadoCreditos" sheet
processed_data_info = ""
historico_pago_cuotas_loaded = False
listado_creditos_loaded = False
numero_credito_col_name = "numeroCredito" # Expected column name in ListadoCreditos

if uploaded_file is not None:
    st.info(f"Processing file: {uploaded_file.name}")
    try:
        # Attempt to read HistoricoPagoCuotas
        credito_df_full = pd.read_excel(uploaded_file, sheet_name="HistoricoPagoCuotas")
        processed_data_info += "✅ Sheet 'HistoricoPagoCuotas' loaded.\n"
        historico_pago_cuotas_loaded = True

        if 'categoriaProductoCrediticio' in credito_df_full.columns:
            creditomotos_df = credito_df_full[credito_df_full["categoriaProductoCrediticio"] == "MOTOS"].copy()
            if creditomotos_df.empty:
                processed_data_info += "⚠️ No data found for 'categoriaProductoCrediticio' == 'MOTOS' in 'HistoricoPagoCuotas'. Risk scores cannot be calculated.\n"
            else:
                processed_data_info += f"Found {len(creditomotos_df)} transaction records for 'MOTOS'. Shape: {creditomotos_df.shape}\n"
                with st.spinner("Calculating risk scores..."):
                    risk_scores_df = calculate_risk_score_df(creditomotos_df, grace_period_input, user_weights)
                if risk_scores_df is not None and not risk_scores_df.empty:
                     processed_data_info += f"✅ Successfully calculated risk scores for {len(risk_scores_df)} unique credits.\n"
                elif risk_scores_df is not None and risk_scores_df.empty:
                    processed_data_info += "⚠️ Risk score calculation resulted in an empty dataset.\n"
                else:
                    processed_data_info += "❌ Risk score calculation failed. Check error messages.\n"
        else:
            processed_data_info += "❌ Column 'categoriaProductoCrediticio' not found in 'HistoricoPagoCuotas'. Cannot filter for 'MOTOS'.\n"
            historico_pago_cuotas_loaded = False # Mark as not fully usable

    except Exception as e:
        if 'HistoricoPagoCuotas' in str(e) or 'No sheet named' in str(e):
            processed_data_info += f"❌ Error: Sheet 'HistoricoPagoCuotas' not found or unreadable: {e}\n"
        else:
            processed_data_info += f"❌ Error processing 'HistoricoPagoCuotas': {e}\n"
        historico_pago_cuotas_loaded = False
        # st.exception(e) # Optionally show full traceback

    try:
        # Attempt to read ListadoCreditos
        listado_creditos_df_temp = pd.read_excel(uploaded_file, sheet_name="ListadoCreditos")
        if numero_credito_col_name in listado_creditos_df_temp.columns:
            listado_creditos_df_temp[numero_credito_col_name] = listado_creditos_df_temp[numero_credito_col_name].astype(str)
            listado_creditos_df = listado_creditos_df_temp # Assign if column exists and casted
            processed_data_info += f"✅ Sheet 'ListadoCreditos' loaded. Found '{numero_credito_col_name}' column.\n"
            listado_creditos_loaded = True
        else:
            processed_data_info += f"⚠️ Sheet 'ListadoCreditos' loaded, but missing '{numero_credito_col_name}' column. Outlier details cannot be fully extracted.\n"
            listado_creditos_df = None # Ensure it's None if key column is missing
            listado_creditos_loaded = False


    except Exception as e:
        if 'ListadoCreditos' in str(e) or 'No sheet named' in str(e):
             processed_data_info += f"⚠️ Sheet 'ListadoCreditos' not found or unreadable. Outlier details will be limited: {e}\n"
        else:
            processed_data_info += f"⚠️ Error processing 'ListadoCreditos': {e}\n"
        listado_creditos_loaded = False
        # st.exception(e) # Optionally show full traceback
else:
    st.info("☝️ Upload an Excel file using the sidebar to begin.")


# Define tabs
tab_titles = ["📊 Risk Score Calculation & Download", "📈 Risk Score EDA", "🕵️ Outlier Analysis"]
tab1, tab2, tab3 = st.tabs(tab_titles)

with tab1:
    st.header(tab_titles[0])
    if processed_data_info:
        st.info(processed_data_info)

    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Calculated Risk Scores (per Credit)")
        st.dataframe(risk_scores_df.style.format({"risk_score": "{:.4f}"}), height=500, use_container_width=True)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            risk_scores_df.to_excel(writer, index=False, sheet_name='RiskScores')
        excel_data = output.getvalue()
        st.download_button(
            label="📥 Download Risk Scores as Excel",
            data=excel_data,
            file_name=f"risk_scores_output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    elif uploaded_file is not None:
        st.warning("Risk scores could not be calculated or resulted in an empty set. Check processing info above or file content.")
    else:
        st.write("Upload a file and configure parameters in the sidebar to see results here.")


with tab2:
    st.header(tab_titles[1])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Risk Score Distribution")
        st.write("Descriptive Statistics for Risk Score:")
        st.dataframe(risk_scores_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        col1, col2 = st.columns(2)
        with col1:
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(risk_scores_df['risk_score'], kde=True, ax=ax_hist, bins=20)
            ax_hist.set_title('Histogram of Risk Scores')
            ax_hist.set_xlabel('Risk Score')
            ax_hist.set_ylabel('Frequency')
            st.pyplot(fig_hist)
        with col2:
            fig_box, ax_box = plt.subplots()
            sns.boxplot(y=risk_scores_df['risk_score'], ax=ax_box)
            ax_box.set_title('Boxplot of Risk Scores')
            ax_box.set_ylabel('Risk Score')
            st.pyplot(fig_box)
    elif uploaded_file is not None:
        st.warning("Risk scores are not available for EDA. Ensure scores are calculated in Tab 1.")
    else:
        st.write("Upload a file and calculate risk scores to see EDA.")

with tab3:
    st.header(tab_titles[2])
    if risk_scores_df is not None and not risk_scores_df.empty:
        low_outliers_df, high_outliers_df, lower_bound, upper_bound = get_outliers_iqr(risk_scores_df, 'risk_score')

        st.subheader("Outlier Identification (based on IQR)")
        st.write(f"Risk Score Lower Bound (Q1 - 1.5 * IQR): {lower_bound:.4f}")
        st.write(f"Risk Score Upper Bound (Q3 + 1.5 * IQR): {upper_bound:.4f}")

        st.markdown("---")
        st.subheader("High-Risk Outliers")
        if not high_outliers_df.empty:
            st.write(f"Number of high-risk outliers found: {len(high_outliers_df)}")
            st.write("Descriptive statistics for high-risk outlier scores:")
            st.dataframe(high_outliers_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
            with st.expander("View High-Risk Outlier Credits & Scores"):
                st.dataframe(high_outliers_df.style.format({"risk_score": "{:.4f}"}))
        else:
            st.write("No high-risk outliers found.")

        st.markdown("---")
        st.subheader("Low-Risk Outliers")
        if not low_outliers_df.empty:
            st.write(f"Number of low-risk outliers found: {len(low_outliers_df)}")
            st.write("Descriptive statistics for low-risk outlier scores:")
            st.dataframe(low_outliers_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
            with st.expander("View Low-Risk Outlier Credits & Scores"):
                st.dataframe(low_outliers_df.style.format({"risk_score": "{:.4f}"}))
        else:
            st.write("No low-risk outliers found.")
        
        st.markdown("---")
        st.subheader("Download Outlier Details")
        if listado_creditos_loaded and listado_creditos_df is not None:
            if not high_outliers_df.empty or not low_outliers_df.empty:
                # Prepare data for Excel export
                output_outliers = io.BytesIO()
                with pd.ExcelWriter(output_outliers, engine='xlsxwriter') as writer:
                    if not high_outliers_df.empty:
                        high_risk_details = listado_creditos_df.merge(
                            high_outliers_df[['credito', 'risk_score']],
                            left_on=numero_credito_col_name,
                            right_on='credito',
                            how='inner'
                        )
                        # Optionally drop the redundant 'credito' column if 'numeroCredito' is primary
                        if 'credito' in high_risk_details.columns and numero_credito_col_name in high_risk_details.columns and 'credito' != numero_credito_col_name:
                             high_risk_details = high_risk_details.drop(columns=['credito'])
                        high_risk_details.to_excel(writer, sheet_name='High Risk Outliers', index=False)

                    if not low_outliers_df.empty:
                        low_risk_details = listado_creditos_df.merge(
                            low_outliers_df[['credito', 'risk_score']],
                            left_on=numero_credito_col_name,
                            right_on='credito',
                            how='inner'
                        )
                        if 'credito' in low_risk_details.columns and numero_credito_col_name in low_risk_details.columns and 'credito' != numero_credito_col_name:
                             low_risk_details = low_risk_details.drop(columns=['credito'])
                        low_risk_details.to_excel(writer, sheet_name='Low Risk Outliers', index=False)
                
                excel_outlier_data = output_outliers.getvalue()
                st.download_button(
                    label="📥 Download Outlier Details as Excel",
                    data=excel_outlier_data,
                    file_name=f"outlier_details_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("No outliers found to download details for.")
        elif not listado_creditos_loaded and uploaded_file is not None:
             st.warning(f"Cannot provide outlier details as 'ListadoCreditos' sheet was not loaded successfully or '{numero_credito_col_name}' column is missing. Please check the uploaded file and processing messages in Tab 1.")
        elif uploaded_file is not None: # listado_creditos_df might be None even if 'loaded' flag was true initially due to missing col
             st.warning(f"Cannot provide outlier details. Issue with 'ListadoCreditos' data. Please check processing messages in Tab 1.")


    elif uploaded_file is not None:
        st.warning("Risk scores are not available for outlier analysis. Ensure scores are calculated in Tab 1.")
    else:
        st.write("Upload a file and calculate risk scores to see outlier analysis.")


st.markdown("---")
st.markdown("App developed by your Expert Data Scientist.")
