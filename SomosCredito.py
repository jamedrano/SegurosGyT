import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

# --- Core Risk Score Calculation Logic (remains the same) ---
def calculate_risk_score_df(df_input, grace_period_days, weights):
    # ... (previous code for this function - no changes here) ...
    if df_input.empty:
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
            # Suppress warnings in Streamlit context, handled by info messages
            # if df[col].isnull().any() and 'streamlit_is_running' not in globals():
            #      print(f"Warning: Column '{col}' contains values that could not be converted to datetime and were set to NaT.")
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
            creditos_unicos[scaled_col_name] = 0.0
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


# --- Utility Function for Outlier Detection (remains the same) ---
def get_outliers_iqr(df, column_name):
    # ... (previous code for this function - no changes here) ...
    if df is None or df.empty or column_name not in df.columns or df[column_name].isnull().all():
        return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        lower_bound = Q1
        upper_bound = Q3
        low_outliers = df[df[column_name] < lower_bound]
        high_outliers = df[df[column_name] > upper_bound]
    else:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        low_outliers = df[df[column_name] < lower_bound]
        high_outliers = df[df[column_name] > upper_bound]
    return low_outliers, high_outliers, lower_bound, upper_bound

# --- Streamlit App UI ---
# streamlit_is_running = True # Not strictly needed if print statements are removed from functions
st.set_page_config(layout="wide")
st.title("🏍️ Motorcycle Loan Risk & Data Quality App")

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
listado_creditos_df = None
processed_data_info = ""
historico_pago_cuotas_loaded = False
listado_creditos_loaded = False
numero_credito_col_name = "numeroCredito"

if uploaded_file is not None:
    st.sidebar.info(f"Processing file: {uploaded_file.name}") # Moved info to sidebar
    # Try to load HistoricoPagoCuotas for risk scoring
    try:
        credito_df_full = pd.read_excel(uploaded_file, sheet_name="HistoricoPagoCuotas")
        processed_data_info += "✅ Sheet 'HistoricoPagoCuotas' loaded.\n"
        historico_pago_cuotas_loaded = True

        if 'categoriaProductoCrediticio' in credito_df_full.columns:
            creditomotos_df = credito_df_full[credito_df_full["categoriaProductoCrediticio"] == "MOTOS"].copy()
            if creditomotos_df.empty:
                processed_data_info += "⚠️ No 'MOTOS' data in 'HistoricoPagoCuotas'. Risk scores not calculated.\n"
            else:
                processed_data_info += f"Found {len(creditomotos_df)} 'MOTOS' records. Shape: {creditomotos_df.shape}\n"
                with st.spinner("Calculating risk scores..."):
                    risk_scores_df = calculate_risk_score_df(creditomotos_df, grace_period_input, user_weights)
                if risk_scores_df is not None and not risk_scores_df.empty:
                     processed_data_info += f"✅ Risk scores calculated for {len(risk_scores_df)} credits.\n"
                elif risk_scores_df is not None and risk_scores_df.empty: # Should ideally be handled by calc function returning None
                    processed_data_info += "⚠️ Risk score calculation resulted in an empty dataset.\n"
                else: # risk_scores_df is None
                    processed_data_info += "❌ Risk score calculation failed. Check errors in function or data.\n"
        else:
            processed_data_info += "❌ 'categoriaProductoCrediticio' missing in 'HistoricoPagoCuotas'.\n"
            historico_pago_cuotas_loaded = False
    except Exception as e:
        processed_data_info += f"❌ Error loading 'HistoricoPagoCuotas': {e}\n"
        historico_pago_cuotas_loaded = False

    # Try to load ListadoCreditos for DQA and Outlier Details
    try:
        # Use a temporary variable to avoid overwriting listado_creditos_df if it fails partially
        temp_listado_df = pd.read_excel(uploaded_file, sheet_name="ListadoCreditos")
        processed_data_info += f"✅ Sheet 'ListadoCreditos' loaded ({temp_listado_df.shape[0]} rows, {temp_listado_df.shape[1]} cols).\n"
        if numero_credito_col_name in temp_listado_df.columns:
            temp_listado_df[numero_credito_col_name] = temp_listado_df[numero_credito_col_name].astype(str)
            listado_creditos_df = temp_listado_df # Assign to main df if good
            listado_creditos_loaded = True
            processed_data_info += f"Found and casted '{numero_credito_col_name}' to string.\n"
        else:
            processed_data_info += f"⚠️ '{numero_credito_col_name}' column missing in 'ListadoCreditos'. Outlier details and some DQA might be affected.\n"
            listado_creditos_df = temp_listado_df # Still load it for general DQA
            listado_creditos_loaded = True # Mark as loaded, but with a key column missing issue
    except Exception as e:
        processed_data_info += f"❌ Error loading 'ListadoCreditos': {e}\n"
        listado_creditos_loaded = False # Explicitly set to false on error

    st.sidebar.text_area("File Processing Log", processed_data_info, height=200)

else:
    st.info("☝️ Upload an Excel file using the sidebar to begin.")


# Define tabs
tab_titles = ["📊 Risk Scores", "📈 Risk EDA", "🕵️ Outlier Analysis", "📋 Customer Data Quality"]
tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

with tab1:
    st.header(tab_titles[0])
    # if processed_data_info: st.info(processed_data_info) # Info moved to sidebar

    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Calculated Risk Scores (per Credit)")
        st.dataframe(risk_scores_df.style.format({"risk_score": "{:.4f}"}), height=500, use_container_width=True)
        # ... (download button code remains the same) ...
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
    elif uploaded_file is not None and historico_pago_cuotas_loaded:
        st.warning("Risk scores could not be calculated. Check processing log in the sidebar.")
    elif uploaded_file is not None and not historico_pago_cuotas_loaded:
        st.error("Cannot calculate risk scores because 'HistoricoPagoCuotas' sheet failed to load. Check processing log.")
    else: # No file uploaded
        st.write("Upload a file and configure parameters to see results here.")


with tab2: # Risk Score EDA
    st.header(tab_titles[1])
    # ... (previous code for this tab - no changes here) ...
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


with tab3: # Outlier Analysis
    st.header(tab_titles[2])
    # ... (previous code for this tab - no changes here) ...
    if risk_scores_df is not None and not risk_scores_df.empty:
        low_outliers_df, high_outliers_df, lower_bound, upper_bound = get_outliers_iqr(risk_scores_df, 'risk_score')
        st.subheader("Outlier Identification (based on IQR of Risk Score)")
        if not np.isnan(lower_bound) and not np.isnan(upper_bound) :
            st.write(f"Risk Score Lower Bound (Q1 - 1.5 * IQR): {lower_bound:.4f}")
            st.write(f"Risk Score Upper Bound (Q3 + 1.5 * IQR): {upper_bound:.4f}")
        else:
            st.write("Could not determine outlier bounds (e.g. all risk scores are identical or insufficient data).")

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
        # ... (similar for low-risk outliers)
        if not low_outliers_df.empty:
            st.write(f"Number of low-risk outliers found: {len(low_outliers_df)}")
            st.write("Descriptive statistics for low-risk outlier scores:")
            st.dataframe(low_outliers_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
            with st.expander("View Low-Risk Outlier Credits & Scores"):
                st.dataframe(low_outliers_df.style.format({"risk_score": "{:.4f}"}))
        else:
            st.write("No low-risk outliers found.")

        st.markdown("---")
        st.subheader("Download Outlier Details from 'ListadoCreditos'")
        if listado_creditos_loaded and listado_creditos_df is not None and numero_credito_col_name in listado_creditos_df.columns:
            if not high_outliers_df.empty or not low_outliers_df.empty:
                output_outliers = io.BytesIO()
                with pd.ExcelWriter(output_outliers, engine='xlsxwriter') as writer:
                    if not high_outliers_df.empty:
                        high_risk_details = listado_creditos_df.merge(
                            high_outliers_df[['credito', 'risk_score']],
                            left_on=numero_credito_col_name,
                            right_on='credito',
                            how='inner'
                        )
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
                st.info("No risk score outliers found to download details for.")
        elif uploaded_file is not None :
             st.warning(f"Cannot provide outlier details as 'ListadoCreditos' sheet was not loaded successfully or '{numero_credito_col_name}' column is missing. Check processing log.")
    elif uploaded_file is not None:
        st.warning("Risk scores are not available for outlier analysis. Ensure scores are calculated in Tab 1.")
    else:
        st.write("Upload a file and calculate risk scores to see outlier analysis.")

with tab4: # Customer Data Quality
    st.header(tab_titles[3] + " (Sheet: 'ListadoCreditos')")

    if uploaded_file is None:
        st.write("Upload an Excel file using the sidebar to analyze 'ListadoCreditos' data.")
    elif not listado_creditos_loaded or listado_creditos_df is None:
        st.error("'ListadoCreditos' sheet could not be loaded or is empty. Please check the uploaded file and the processing log in the sidebar.")
    else: # listado_creditos_df is available
        df_dqa = listado_creditos_df # Use a shorter alias for convenience

        st.subheader("1. General Overview")
        st.write(f"Number of records (rows): {df_dqa.shape[0]}")
        st.write(f"Number of features (columns): {df_dqa.shape[1]}")

        with st.expander("Column Names and Data Types"):
            st.dataframe(df_dqa.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

        st.subheader("2. Missing Values Analysis")
        missing_summary = df_dqa.isnull().sum().reset_index()
        missing_summary.columns = ['Column', 'Missing Count']
        missing_summary['Missing Percentage (%)'] = (missing_summary['Missing Count'] / len(df_dqa)) * 100
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values(by='Missing Percentage (%)', ascending=False)

        if not missing_summary.empty:
            st.write("Columns with missing values:")
            st.dataframe(missing_summary.style.format({'Missing Percentage (%)': "{:.2f}%"}))

            # Bar chart for missing percentages
            fig_missing, ax_missing = plt.subplots(figsize=(10, max(5, len(missing_summary) * 0.3))) # Dynamic height
            sns.barplot(x='Missing Percentage (%)', y='Column', data=missing_summary, ax=ax_missing, palette="viridis")
            ax_missing.set_title('Percentage of Missing Values per Column')
            ax_missing.set_xlabel('Percentage Missing (%)')
            plt.tight_layout()
            st.pyplot(fig_missing)
        else:
            st.success("No missing values found in the 'ListadoCreditos' data! 🎉")

        st.subheader("3. Duplicate Records Analysis")
        num_total_duplicates = df_dqa.duplicated().sum()
        st.write(f"Number of completely duplicate rows (all columns identical): {num_total_duplicates}")
        if num_total_duplicates > 0:
            with st.expander("View Completely Duplicate Rows"):
                st.dataframe(df_dqa[df_dqa.duplicated(keep=False)]) # Show all occurrences of duplicates

        if numero_credito_col_name in df_dqa.columns:
            num_id_duplicates = df_dqa.duplicated(subset=[numero_credito_col_name]).sum()
            st.write(f"Number of duplicate entries in '{numero_credito_col_name}' column: {num_id_duplicates}")
            if num_id_duplicates > 0:
                st.warning(f"'{numero_credito_col_name}' should ideally be unique for each customer/loan record.")
                with st.expander(f"View Duplicate '{numero_credito_col_name}' Entries"):
                    st.dataframe(df_dqa[df_dqa.duplicated(subset=[numero_credito_col_name], keep=False)].sort_values(by=numero_credito_col_name))
        else:
            st.warning(f"'{numero_credito_col_name}' column not found. Cannot check for ID duplicates.")

        st.subheader("4. Column-wise Detailed Analysis")
        # Allow user to select columns for detailed analysis to avoid overwhelming display
        cols_for_detail = st.multiselect(
            "Select columns for detailed analysis:",
            options=df_dqa.columns.tolist(),
            default=df_dqa.columns.tolist()[:min(5, len(df_dqa.columns))] # Default to first 5 or all if less
        )

        for col in cols_for_detail:
            with st.expander(f"Analysis for Column: '{col}' (Type: {df_dqa[col].dtype})"):
                st.write(f"**Data Type:** {df_dqa[col].dtype}")
                st.write(f"**Number of Unique Values:** {df_dqa[col].nunique()}")
                st.write(f"**Missing Values:** {df_dqa[col].isnull().sum()} ({df_dqa[col].isnull().sum() / len(df_dqa) * 100:.2f}%)")

                if pd.api.types.is_numeric_dtype(df_dqa[col]):
                    st.write("**Descriptive Statistics:**")
                    st.dataframe(df_dqa[col].describe().to_frame().T)
                    
                    # Simple histogram for numeric data
                    fig_num, ax_num = plt.subplots(figsize=(6,4))
                    try:
                        sns.histplot(df_dqa[col].dropna(), kde=False, ax=ax_num, bins=min(30, df_dqa[col].nunique())) # Cap bins
                        ax_num.set_title(f"Distribution of {col}")
                        st.pyplot(fig_num)
                    except Exception as e:
                        st.caption(f"Could not plot histogram for {col}: {e}")
                    plt.close(fig_num)


                elif pd.api.types.is_object_dtype(df_dqa[col]) or pd.api.types.is_categorical_dtype(df_dqa[col]):
                    st.write("**Value Counts (Top 20):**")
                    st.dataframe(df_dqa[col].value_counts().nlargest(20).reset_index().rename(columns={'index': 'Value', col: 'Count'}))
                    if df_dqa[col].nunique() > 20:
                        st.caption("Note: Displaying top 20 unique values. More exist.")
                
                # Add more specific checks if needed, e.g., for date columns if parsed
                # elif pd.api.types.is_datetime64_any_dtype(df_dqa[col]):
                # st.write(f"**Date Range:** From {df_dqa[col].min()} to {df_dqa[col].max()}")


st.markdown("---")
st.markdown("App developed by your Expert Data Scientist, Antonio Medrano, CepSA.")
