import streamlit as st
import pandas as pd
import plotly.express as px
import io  # For handling file-like objects
from st_aggrid import AgGrid, GridOptionsBuilder  # For interactive table

# --- Helper Functions ---
def display_data_grid(df):
    """Displays a dataframe using AgGrid with interactive features."""
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #enable multi-select
    gridOptions = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT',
        update_mode='MODEL_CHANGED',
        fit_columns_on_grid_load=False,
        theme='streamlit',  # Add theme color to the table
        enable_enterprise_modules=True,
        height=350,
        width='100%',
        reload_data=True
    )

    selected_rows = grid_response['selected_rows']
    selected_df = pd.DataFrame(selected_rows)  #convert to df

    return selected_df


# Streamlit App
st.title("Interactive Data Analysis Dashboard")

# Step 1: Upload the dataset
st.subheader("1. Upload Your Data")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Step 2: Load the dataset
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()  #get the extension
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file) #using pd.read_csv directly
        elif file_extension == 'xlsx':
             df = pd.read_excel(uploaded_file)  # using pd.read_excel directly
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()  # Stop execution if the file format is unsupported

        st.success("File successfully uploaded and processed!")

        # Step 3: Display Dataset with interactive features
        st.subheader("2. Interactive Dataset Preview")
        selected_df = display_data_grid(df) # Display with AgGrid
        if not selected_df.empty:
             st.write("Rows Selected:",selected_df.shape[0])
             with st.expander("View Selected Data"):
                  st.dataframe(selected_df)

        # Step 4: Basic data analysis
        st.subheader("3. Basic Data Analysis")
        st.write("**Shape of the dataset:**", df.shape)
        st.write("**Columns in the dataset:**", list(df.columns))

        if not df.empty: #check the df not empty before display
             st.write("**Summary statistics:**")
             st.write(df.describe())
        else:
           st.warning("DataFrame is empty, no summary statistics to show.")


        # Step 5: Interactive Visualization
        st.subheader("4. Interactive Visualizations")

        # Select columns for visualization
        numeric_columns = df.select_dtypes(include=["number"]).columns
        categorical_columns = df.select_dtypes(include=["object"]).columns

        # Handle cases where no numeric or categorical columns exist
        if numeric_columns.empty and categorical_columns.empty:
             st.warning("No numeric or categorical columns found in your dataset. Cannot create visualizations.")
             st.stop() #Stop execution if there are no columns
        elif numeric_columns.empty:
           st.warning("No numeric columns found in your dataset. Only bar chart with categorical columns is available.")
           plot_type = "Bar Chart" # force bar chart if no numeric
        else:
           plot_type = st.selectbox("Select Plot Type", ["Bar Chart", "Scatter Plot", "Histogram", "Line Plot"])


        if plot_type == "Bar Chart":
             if categorical_columns.empty:
                 st.warning("No categorical columns to create a bar chart.")
             else:
              x_axis = st.selectbox("X-axis (Categorical)", categorical_columns)
              y_axis = st.selectbox("Y-axis (Numeric)", numeric_columns) if not numeric_columns.empty else None
              if y_axis: #check if y-axis is selected (only numeric columns are present)
                 chart = px.bar(df, x=x_axis, y=y_axis, title=f"{plot_type} of {y_axis} by {x_axis}")
                 st.plotly_chart(chart)
              else:
                  chart = px.bar(df, x=x_axis, title=f"{plot_type} of {x_axis}") #bar chart with just categorical
                  st.plotly_chart(chart)



        elif plot_type == "Scatter Plot":
           if len(numeric_columns) < 2:
              st.warning("Scatter plots require at least two numeric columns.")
           else:
              x_axis = st.selectbox("X-axis", numeric_columns)
              y_axis = st.selectbox("Y-axis", numeric_columns, index=1)
              chart = px.scatter(df, x=x_axis, y=y_axis, title=f"{plot_type} of {y_axis} vs {x_axis}")
              st.plotly_chart(chart)


        elif plot_type == "Histogram":
            x_axis = st.selectbox("Column", numeric_columns)
            bins = st.slider("Number of bins", 5, 50, 10)
            chart = px.histogram(df, x=x_axis, nbins=bins, title=f"{plot_type} of {x_axis}")
            st.plotly_chart(chart)

        elif plot_type == "Line Plot":
             if len(numeric_columns) < 2:
               st.warning("Line plots require at least two numeric columns.")
             else:
              x_axis = st.selectbox("X-axis (Time/Sequential)", numeric_columns)
              y_axis = st.selectbox("Y-axis", numeric_columns)
              chart = px.line(df, x=x_axis, y=y_axis, title=f"{plot_type} of {y_axis} over {x_axis}")
              st.plotly_chart(chart)


    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Upload a dataset to get started!")