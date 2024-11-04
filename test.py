import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
file_path = 'consolidated_planilla.xlsx'
data = pd.read_excel(file_path)

# Streamlit App
st.title("Time Usage Analysis for Commercial Department Staff")

# Create tabs for different views
tab1, tab2 = st.tabs(["Data Analysis", "Filtered Data"])

with tab1:
    st.header("Analyze Duration by Area and Position")

    # Sidebar Filters for Duration Analysis
    area_options = data["Area"].unique()
    position_options = data["Posicion"].unique()

    selected_area = st.selectbox("Select Area", options=area_options)
    selected_position = st.selectbox("Select Position", options=position_options)

    # Filter data based on user selection
    filtered_data = data[(data["Area"] == selected_area) & (data["Posicion"] == selected_position)]

    # Display total duration for selected Area and Position
    total_duration = filtered_data["Duración Calculada"].sum()
    st.write(f"Total Duration for **{selected_area}** in Position **{selected_position}**: {total_duration:.2f} hours")

    # Plotting Total Duration by Date for Selected Area and Position
    st.subheader("Total Duration by Date")
    fig, ax = plt.subplots(figsize=(10, 6))
    duration_by_date = filtered_data.groupby("Fecha")["Duración Calculada"].sum()
    duration_by_date.plot(kind='bar', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Duration (Hours)")
    ax.set_title(f"Total Duration Over Time for {selected_area} - {selected_position}")
    st.pyplot(fig)

with tab2:
    st.header("Filtered Data View")
    
    # Multi-select filters for a broader view
    selected_areas = st.multiselect("Filter by Area(s)", options=area_options, default=area_options)
    selected_positions = st.multiselect("Filter by Position(s)", options=position_options, default=position_options)

    # Filter data based on broader multi-selection
    multi_filtered_data = data[(data["Area"].isin(selected_areas)) & (data["Posicion"].isin(selected_positions))]
    
    # Display filtered data
    st.write(multi_filtered_data)

    # Summary Statistics
    st.subheader("Duration Statistics")
    st.write(multi_filtered_data["Duración Calculada"].describe())
