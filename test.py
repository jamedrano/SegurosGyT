import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'consolidated_planilla.xlsx'
data = pd.read_excel(file_path)

# Streamlit App
st.title("Análisis de Tiempo para el Staff del Departamento Comercial")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Análisis de Datos", "Datos Filtrados", "Análisis por Fecha"])

with tab1:
    st.header("Análisis de Duración de Actividades por Area, por Posición y por Tipo de Actividad")

    # Add "Todos" to area and position options
    area_options = ["Todos"] + list(data["Area"].unique())
    position_options = ["Todos"] + list(data["Posicion"].unique())

    selected_area = st.selectbox("Select Area", options=area_options)
    selected_position = st.selectbox("Select Position", options=position_options)

    # Filter data based on user selection, including "Todos" option
    if selected_area != "Todos" and selected_position != "Todos":
        filtered_data = data[(data["Area"] == selected_area) & (data["Posicion"] == selected_position)]
    elif selected_area == "Todos" and selected_position != "Todos":
        filtered_data = data[data["Posicion"] == selected_position]
    elif selected_area != "Todos" and selected_position == "Todos":
        filtered_data = data[data["Area"] == selected_area]
    else:
        filtered_data = data  # If both are "Todos", select all data

    # Display total duration for the selected filters
    total_duration = filtered_data["Duración Calculada"].sum()
    if selected_area == "Todos" and selected_position == "Todos":
        st.write(f"Total Duration for **All Areas and Positions**: {total_duration:.2f} hours")
    elif selected_area == "Todos":
        st.write(f"Total Duration for **All Areas** in Position **{selected_position}**: {total_duration:.2f} hours")
    elif selected_position == "Todos":
        st.write(f"Total Duration for **{selected_area}** in **All Positions**: {total_duration:.2f} hours")
    else:
        st.write(f"Total Duration for **{selected_area}** in Position **{selected_position}**: {total_duration:.2f} hours")

    # Plotting Total Duration by Activity Type for Selected Area and Position
    st.subheader("Total Duration by Activity Type")
    fig, ax = plt.subplots(figsize=(10, 6))
    duration_by_activity = filtered_data.groupby("Tipo de Actividad")["Duración Calculada"].sum()
    duration_by_activity = duration_by_activity.sort_values(ascending=False)  # Sort from largest to smallest
    duration_by_activity.plot(kind='bar', ax=ax)
    ax.set_xlabel("Tipo de Actividad")
    ax.set_ylabel("Duración Total (Horas)")
    title = f"Duración Total por Tipo de Actividad para {selected_area} - {selected_position}".replace("Todos", "All")
    ax.set_title(title)
    st.pyplot(fig)

with tab2:
    st.header("Resumen Estadístico de Datos por Area y por Posición")
    
    # Dropdowns for filtering
    selected_area_summary = st.selectbox("Seleccionar Area para Resumen", options=area_options, key="area_summary")
    selected_position_summary = st.selectbox("Seleccionar Posición para Resumen", options=position_options, key="position_summary")

    # Filter data for statistical summary based on dropdown selections
    if selected_area_summary != "Todos" and selected_position_summary != "Todos":
        summary_data = data[(data["Area"] == selected_area_summary) & (data["Posicion"] == selected_position_summary)]
    elif selected_area_summary == "Todos" and selected_position_summary != "Todos":
        summary_data = data[data["Posicion"] == selected_position_summary]
    elif selected_area_summary != "Todos" and selected_position_summary == "Todos":
        summary_data = data[data["Area"] == selected_area_summary]
    else:
        summary_data = data  # If both are "Todos", select all data

    # Display statistical summary for Duración Calculada
    st.subheader("Statistical Summary")
    st.write(f"Resumen estadístico para **{selected_area_summary}** - **{selected_position_summary}**:")
    st.write(summary_data["Duración Calculada"].describe())

with tab3:
    st.header("Análisis de Duración por Día")

    # Dropdowns for filtering
    selected_area_timeseries = st.selectbox("Seleccione Area", options=area_options, key="area_timeseries")
    selected_position_timeseries = st.selectbox("Seleccione Posición", options=position_options, key="position_timeseries")

    # Filter data for time series analysis based on dropdown selections
    if selected_area_timeseries != "Todos" and selected_position_timeseries != "Todos":
        timeseries_data = data[(data["Area"] == selected_area_timeseries) & (data["Posicion"] == selected_position_timeseries)]
    elif selected_area_timeseries == "Todos" and selected_position_timeseries != "Todos":
        timeseries_data = data[data["Posicion"] == selected_position_timeseries]
    elif selected_area_timeseries != "Todos" and selected_position_timeseries == "Todos":
        timeseries_data = data[data["Area"] == selected_area_timeseries]
    else:
        timeseries_data = data  # If both are "Todos", select all data

    # Aggregating data by date for time series analysis
    duration_by_date = timeseries_data.groupby("Fecha")["Duración Calculada"].sum()

    # Plotting the time series data
    fig, ax = plt.subplots(figsize=(10, 6))
    duration_by_date.plot(ax=ax)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Duración Total (Horas)")
    title = f"Duración Total en el Tiempo Para {selected_area_timeseries} - {selected_position_timeseries}".replace("Todos", "All")
    ax.set_title(title)
    st.pyplot(fig)
