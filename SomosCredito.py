import streamlit as st
import pandas as pd
import numpy as np
import io # Required for Excel download
import matplotlib.pyplot as plt
import seaborn as sns

# --- Core Risk Score Calculation Logic (adapted from your script) ---
# This function remains the same as before
def calculate_risk_score_df(df_input, grace_period_days, weights):
    """
    Calculates the risk score based on loan behavior.

    Args:
        df_input (pd.DataFrame): The input DataFrame, expected to be pre-filtered
                                 for "MOTOS" and from "HistoricoPagoCuotas" sheet.
        grace_period_days (int): Grace period in days.
        weights (dict): Dictionary of weights for different indicators.

    Returns:
        pd.DataFrame: DataFrame with 'credito' and 'risk_score'.
                      Returns None if critical errors occur (e.g., missing columns).
    """
    if df_input.empty:
        st.warning("Input data is empty after initial filtering. Cannot calculate risk scores.")
        return None

    df = df_input.copy()

    # --- Essential Column Checks ---
    required_cols = [
        'fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro',
        'fechaTRansaccion', 'credito', 'reglaCobranza', 'cuotaCubierta',
        'cuotaEsperada', 'saldoCapitalActual', 'totalDesembolso', 'cobranzaTrans',
        'categoriaProductoCrediticio'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_cols)}")
        return None

    # st.write("Shape of data for risk scoring:", df.shape) # Moved to main app flow for clarity

    date_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro', 'fechaTRansaccion']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].isnull().any():
                # Only show warning if not run from within the main calculation block in Streamlit
                if 'streamlit_is_running' not in globals() or not streamlit_is_running:
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
            creditos_unicos[scaled_col_name] = 0.0 if max_val == 0 else 0.5 # Or handle as per business logic
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
# --- End of Core Risk Score Calculation Logic ---

# --- Streamlit App UI ---
streamlit_is_running = True # Flag for conditional printing in function
st.set_page_config(layout="wide")
st.title("🏍️ Motorcycle Loan Risk Score Calculator")

# --- Sidebar for User Inputs (remains the same) ---
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload 'DATOS_AL_DDMMYYYY.xlsx'", type=["xlsx"])
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

# --- Main App Area with Tabs ---
# Initialize risk_scores_df to None or an empty DataFrame to ensure it's defined
risk_scores_df = None
processed_data_info = "" # To store info about processed data

if uploaded_file is not None:
    st.info(f"Processing file: {uploaded_file.name}")
    try:
        credito_df_full = pd.read_excel(uploaded_file, sheet_name="HistoricoPagoCuotas")
        processed_data_info += "File loaded successfully. Sheet 'HistoricoPagoCuotas' found.\n"

        if 'categoriaProductoCrediticio' in credito_df_full.columns:
            creditomotos_df = credito_df_full[credito_df_full["categoriaProductoCrediticio"] == "MOTOS"].copy()
            if creditomotos_df.empty:
                st.warning("No data found for 'categoriaProductoCrediticio' == 'MOTOS'. Cannot proceed.")
                # risk_scores_df remains None or empty
            else:
                processed_data_info += f"Found {len(creditomotos_df)} transaction records for 'MOTOS' category. Shape: {creditomotos_df.shape}\n"
                with st.spinner("Calculating risk scores... Please wait."):
                    risk_scores_df = calculate_risk_score_df(creditomotos_df, grace_period_input, user_weights)
                if risk_scores_df is not None and not risk_scores_df.empty:
                     processed_data_info += f"Successfully calculated risk scores for {len(risk_scores_df)} unique credits.\n"
                elif risk_scores_df is not None and risk_scores_df.empty:
                    st.warning("Risk score calculation resulted in an empty dataset. Please check the input data and parameters.")
                else: # risk_scores_df is None due to error in calculation function
                    st.error("Risk score calculation failed. Check for earlier error messages.")
        else:
            st.error("Column 'categoriaProductoCrediticio' not found. Cannot filter for 'MOTOS'.")
            # risk_scores_df remains None or empty

    except KeyError as e:
        if 'HistoricoPagoCuotas' in str(e):
            st.error(f"Error: Sheet 'HistoricoPagoCuotas' not found in the uploaded Excel file.")
        else:
            st.error(f"Error processing the file. A required column might be missing: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during file processing or score calculation: {e}")
        st.exception(e) # Provides full traceback for debugging
else:
    st.info("☝️ Upload an Excel file using the sidebar to begin.")


# Define tabs
tab1, tab2 = st.tabs(["📊 Risk Score Calculation & Download", "📈 Risk Score EDA"])

with tab1:
    st.header("Risk Score Calculation & Download")
    if processed_data_info: # Display info if file processing has started
        st.info(processed_data_info)

    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Calculated Risk Scores (per Credit)")
        st.dataframe(risk_scores_df.style.format({"risk_score": "{:.4f}"}), height=500, use_container_width=True)

        # --- Download Button ---
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
    elif uploaded_file is not None and (risk_scores_df is None or risk_scores_df.empty):
        st.warning("Risk scores could not be calculated or resulted in an empty set. Please check inputs and file content. Error messages might be above or in the sidebar area if file loading failed.")
    elif uploaded_file is None:
        st.write("Upload a file and configure parameters in the sidebar to see results here.")


with tab2:
    st.header("Exploratory Data Analysis (EDA) of Risk Score")
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Risk Score Distribution")

        # Display some descriptive statistics
        st.write("Descriptive Statistics for Risk Score:")
        st.dataframe(risk_scores_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))

        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(risk_scores_df['risk_score'], kde=True, ax=ax_hist, bins=20)
            ax_hist.set_title('Histogram of Risk Scores')
            ax_hist.set_xlabel('Risk Score')
            ax_hist.set_ylabel('Frequency')
            st.pyplot(fig_hist)

        with col2:
            # Boxplot
            fig_box, ax_box = plt.subplots()
            sns.boxplot(y=risk_scores_df['risk_score'], ax=ax_box)
            ax_box.set_title('Boxplot of Risk Scores')
            ax_box.set_ylabel('Risk Score')
            st.pyplot(fig_box)

        # You can add more plots here if needed
        # e.g., st.line_chart(risk_scores_df['risk_score']) # If scores had a natural order

    elif uploaded_file is not None and (risk_scores_df is None or risk_scores_df.empty):
        st.warning("Risk scores are not available for EDA. Please ensure scores are calculated successfully in the 'Risk Score Calculation & Download' tab.")
    else: # No file uploaded yet
        st.write("Upload a file and calculate risk scores to see EDA.")


st.markdown("---")
st.markdown("App developed by your Expert Data Scientist, CepSA.")
