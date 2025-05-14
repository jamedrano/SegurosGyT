import streamlit as st
import pandas as pd
import numpy as np
import io # Required for Excel download

# --- Core Risk Score Calculation Logic (adapted from your script) ---
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
        'categoriaProductoCrediticio' # Though filtered before, good to be aware
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_cols)}")
        return None
    # --- End Essential Column Checks ---

    st.write("Shape of data for risk scoring:", df.shape)

    # 2. Data type conversions
    date_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro', 'fechaTRansaccion']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].isnull().any():
                st.warning(f"Warning: Column '{col}' contains values that could not be converted to datetime and were set to NaT.")


    df['credito'] = df['credito'].astype(str)

    # 3. Define grace period
    grace_period = pd.Timedelta(days=grace_period_days)

    # 4. Calculate late payment indicator
    # Handle cases where fechaPagoRecibido or fechaEsperadaPago might be NaT after conversion
    mask_valid_dates_late_payment = df['fechaPagoRecibido'].notna() & df['fechaEsperadaPago'].notna()
    df['late_payment'] = 0 # Initialize with 0
    df.loc[mask_valid_dates_late_payment, 'late_payment'] = \
        (df.loc[mask_valid_dates_late_payment, 'fechaPagoRecibido'] >
         (df.loc[mask_valid_dates_late_payment, 'fechaEsperadaPago'] + grace_period)).astype(int)


    # 5. Calculate late payment ratio
    late_payment_counts = df.groupby('credito')['late_payment'].sum()
    total_payments = df.groupby('credito')['fechaEsperadaPago'].count() # Count non-NA expected payment dates

    # Avoid division by zero if total_payments is 0
    late_payment_ratio = (late_payment_counts / total_payments.replace(0, np.nan)).fillna(0)


    # 6. Calculate payment coverage ratio
    df['cuotaCubierta'] = df['cuotaCubierta'].fillna(0)
    total_payment_made = df.groupby('credito')['cuotaCubierta'].sum()
    total_payment_expected = df.groupby('credito')['cuotaEsperada'].sum()

    payment_coverage_ratio = (total_payment_made / total_payment_expected.replace(0, np.nan))
    payment_coverage_ratio = payment_coverage_ratio.fillna(1) # If expected is 0 or NaN, assume covered
    payment_coverage_ratio = payment_coverage_ratio.replace([np.inf, -np.inf], 1) # Handle division by zero if expected was 0 and made was not

    # 7. Calculate outstanding balance ratio
    # Ensure 'saldoCapitalActual' and 'totalDesembolso' are numeric, coercing errors
    df['saldoCapitalActual'] = pd.to_numeric(df['saldoCapitalActual'], errors='coerce')
    df['totalDesembolso'] = pd.to_numeric(df['totalDesembolso'], errors='coerce')

    last_saldo = df.groupby('credito')['saldoCapitalActual'].last()
    first_desembolso = df.groupby('credito')['totalDesembolso'].first()

    outstanding_balance_ratio = (last_saldo / first_desembolso.replace(0, np.nan))
    outstanding_balance_ratio = outstanding_balance_ratio.fillna(0) # If first_desembolso is 0 or NaN, ratio is 0

    # 8. Calculate collection activity indicator
    df['cobranzaTrans'] = pd.to_numeric(df['cobranzaTrans'], errors='coerce').fillna(0)
    df['collection_activity'] = (df['cobranzaTrans'] > 0).astype(int)
    collection_activity_count = df.groupby('credito')['collection_activity'].sum()

    # 9. Merge the aggregated indicators back into a new DataFrame for credit-level scores
    # We create a DataFrame of unique creditos to merge onto
    creditos_unicos = pd.DataFrame(df['credito'].unique(), columns=['credito'])

    creditos_unicos = creditos_unicos.merge(late_payment_ratio.to_frame(name='late_payment_ratio'), left_on='credito', right_index=True, how='left')
    creditos_unicos = creditos_unicos.merge(payment_coverage_ratio.to_frame(name='payment_coverage_ratio'), left_on='credito', right_index=True, how='left')
    creditos_unicos = creditos_unicos.merge(outstanding_balance_ratio.to_frame(name='outstanding_balance_ratio'), left_on='credito', right_index=True, how='left')
    creditos_unicos = creditos_unicos.merge(collection_activity_count.to_frame(name='collection_activity_count'), left_on='credito', right_index=True, how='left')

    # Fill NaNs that might arise if a credito didn't have data for a particular aggregation (e.g., no payments)
    creditos_unicos['late_payment_ratio'] = creditos_unicos['late_payment_ratio'].fillna(0)
    creditos_unicos['payment_coverage_ratio'] = creditos_unicos['payment_coverage_ratio'].fillna(1) # Default to fully covered
    creditos_unicos['outstanding_balance_ratio'] = creditos_unicos['outstanding_balance_ratio'].fillna(0) # Default to fully paid
    creditos_unicos['collection_activity_count'] = creditos_unicos['collection_activity_count'].fillna(0)


    # 10. Weights are passed as an argument

    # 11. Scale the indicators to a range of 0-1 using min-max scaling
    # Scaling should be done carefully if there's only one unique value (max=min)
    for col_name in ['late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']:
        min_val = creditos_unicos[col_name].min()
        max_val = creditos_unicos[col_name].max()
        scaled_col_name = f'{col_name}_scaled'
        if max_val == min_val:
            creditos_unicos[scaled_col_name] = 0.0 if max_val == 0 else 0.5 # Or handle as per business logic
        else:
            creditos_unicos[scaled_col_name] = (creditos_unicos[col_name] - min_val) / (max_val - min_val)
        creditos_unicos[scaled_col_name] = creditos_unicos[scaled_col_name].fillna(0) # Fill NaNs from scaling (e.g., if a column was all NaNs before)


    # 13. Calculate the weighted risk score
    creditos_unicos['risk_score'] = (
        weights['late_payment_ratio'] * creditos_unicos['late_payment_ratio_scaled'] +
        weights['payment_coverage_ratio'] * (1 - creditos_unicos['payment_coverage_ratio_scaled']) +  # Invert coverage
        weights['outstanding_balance_ratio'] * creditos_unicos['outstanding_balance_ratio_scaled'] +
        weights['collection_activity_count'] * creditos_unicos['collection_activity_count_scaled']
    )

    return creditos_unicos[['credito', 'risk_score']]

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🏍️ Motorcycle Loan Risk Score Calculator")

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload 'DATOS_AL_DDMMYYYY.xlsx'", type=["xlsx"])

default_grace_period = 5
grace_period_input = st.sidebar.number_input("Grace Period (days)", min_value=0, max_value=30, value=default_grace_period, step=1)

st.sidebar.subheader("Indicator Weights")
# Ensure sum of weights is 1 (optional, but good practice for interpretation)
# For now, allowing independent weights as per original script
w_late_payment = st.sidebar.slider("Late Payment Ratio Weight", 0.0, 1.0, 0.40, 0.01)
w_payment_coverage = st.sidebar.slider("Payment Coverage Ratio Weight", 0.0, 1.0, 0.40, 0.01)
w_outstanding_balance = st.sidebar.slider("Outstanding Balance Ratio Weight", 0.0, 1.0, 0.20, 0.01)
w_collection_activity = st.sidebar.slider("Collection Activity Count Weight", 0.0, 1.0, 0.00, 0.01) # Original was 0

user_weights = {
    'late_payment_ratio': w_late_payment,
    'payment_coverage_ratio': w_payment_coverage,
    'outstanding_balance_ratio': w_outstanding_balance,
    'collection_activity_count': w_collection_activity
}

# --- Main App Area ---
if uploaded_file is not None:
    st.info(f"Processing file: {uploaded_file.name}")
    try:
        # Load the specific sheet and filter for "MOTOS"
        credito_df_full = pd.read_excel(uploaded_file, sheet_name="HistoricoPagoCuotas")
        st.success("File loaded successfully. Sheet 'HistoricoPagoCuotas' found.")

        # Filter for "MOTOS"
        if 'categoriaProductoCrediticio' in credito_df_full.columns:
            creditomotos_df = credito_df_full[credito_df_full["categoriaProductoCrediticio"] == "MOTOS"].copy() # Use .copy()
            if creditomotos_df.empty:
                st.warning("No data found for 'categoriaProductoCrediticio' == 'MOTOS'. Cannot proceed.")
                st.stop()
            st.write(f"Found {len(creditomotos_df)} records for 'MOTOS' category.")
        else:
            st.error("Column 'categoriaProductoCrediticio' not found in the uploaded file. Cannot filter for 'MOTOS'.")
            st.stop()


        # Calculate risk scores
        with st.spinner("Calculating risk scores... Please wait."):
            risk_scores_df = calculate_risk_score_df(creditomotos_df, grace_period_input, user_weights)

        if risk_scores_df is not None and not risk_scores_df.empty:
            st.subheader("Calculated Risk Scores")
            st.dataframe(risk_scores_df.style.format({"risk_score": "{:.4f}"}), height=500) # Display with formatting

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
        elif risk_scores_df is not None and risk_scores_df.empty:
             st.warning("Risk score calculation resulted in an empty dataset. Please check the input data and parameters.")
        else:
            st.error("Risk score calculation failed. Please check the console for error messages or warnings above.")


    except KeyError as e:
        if 'HistoricoPagoCuotas' in str(e):
            st.error(f"Error: Sheet 'HistoricoPagoCuotas' not found in the uploaded Excel file. Please ensure the sheet name is correct.")
        else:
            st.error(f"Error processing the file. A required column might be missing or misnamed: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e) # Provides full traceback for debugging
else:
    st.info("☝️ Upload an Excel file using the sidebar to begin.")

st.markdown("---")
st.markdown("App developed by your Expert Data Scientist.")
