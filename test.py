import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
file_path = 'consolidated_planilla.xlsx'
data = pd.read_excel(file_path)

# Streamlit App
st.title("Time Usage Analysis for Commercial Department Staff")

# Sidebar Filters
st.sidebar.header("Filters")
area = st.sidebar.multiselect("Select Area(s)", options=data["Area"].unique(), default=data["Area"].unique())
position = st.sidebar.multiselect("Select Position(s)", options=data["Posicion"].unique(), default=data["Posicion"].unique())
priority = st.sidebar.multiselect("Select Priority Level(s)", options=data["Prioridad"].unique(), default=data["Prioridad"].unique())

# Apply Filters
filtered_data = data[(data["Area"].isin(area)) & (data["Posicion"].isin(position)) & (data["Prioridad"].isin(priority))]

# Display filtered data
st.subheader("Filtered Data")
st.write(filtered_data)

# Visualization: Total Duration by Area and Position
st.subheader("Total Duration Analysis")

# Aggregating data for visualization
duration_by_area = filtered_data.groupby(["Area", "Posicion"])["Duración Calculada"].sum().reset_index()
fig = px.bar(duration_by_area, x="Area", y="Duración Calculada", color="Posicion",
             title="Total Duration by Area and Position",
             labels={"Duración Calculada": "Total Duration (Hours)"})

st.plotly_chart(fig)

# Individual Duration Analysis
st.subheader("Individual Task Duration Analysis")
individual_duration_fig = px.scatter(filtered_data, x="Fecha", y="Duración Calculada", color="Posicion", 
                                     hover_data=["Nombre", "Descripción de la Actividad"],
                                     title="Individual Task Duration Over Time",
                                     labels={"Duración Calculada": "Task Duration (Hours)", "Fecha": "Date"})

st.plotly_chart(individual_duration_fig)

# Duration Statistics
st.subheader("Duration Statistics")
st.write(filtered_data["Duración Calculada"].describe())
  
