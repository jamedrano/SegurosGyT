import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'consolidated_planilla.xlsx'
data = pd.read_excel(file_path)

# Streamlit App
st.title("Time Usage Analysis for Commercial Department Staff")

# Create tabs for different views
tab1, tab2 = st.tabs(["Data Analysis", "Filtered Data"])

with tab1:
    st.header("Analyze Duration by Area and Position")

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
    duration_by_activity.plot(kind='bar', ax=ax)
    ax.set_xlabel("Activity Type")
    ax.set_ylabel("Total Duration (Hours)")
    title = f"Total Duration by Activity Type for {selected_area} - {selected_position}".replace("Todos", "All")
    ax.set_title(title)
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
