import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from src.data import data_loader
from src.visualizations import plots
from src.utils.reporting import get_csv_download, get_pdf_download

def render_data_exploration():
    """Render the data exploration page"""
    st.title("Data Exploration")
    st.subheader("Explore and visualize agricultural datasets")
    
    # Dataset selection
    dataset = st.selectbox(
        "Select Dataset to Explore",
        ["Crop Recommendation", "Yield Data", "Temperature Data", "Rainfall Data", "Pesticides Data"]
    )
    
    # Load the selected dataset
    df = None
    if dataset == "Crop Recommendation":
        df = data_loader.load_crop_recommendation_data()
        # Ensure label column is treated as categorical
        if df is not None and 'label' in df.columns:
            df['label'] = df['label'].astype('category')
    elif dataset == "Yield Data":
        df = data_loader.load_yield_data()
    elif dataset == "Temperature Data":
        df = data_loader.load_temperature_data()
    elif dataset == "Rainfall Data":
        df = data_loader.load_rainfall_data()
    elif dataset == "Pesticides Data":
        df = data_loader.load_pesticides_data()
    
    if df is None:
        st.error(f"Failed to load {dataset} dataset. Please check if the file exists.")
        return
    
    # Display basic information
    st.write("### Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    
    # Display dataset preview
    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(df.head(10))
    
    # Display column information
    with st.expander("Column Information"):
        st.write("#### Column Details")
        for col in df.columns:
            st.write(f"**{col}**: {df[col].dtype}")
            
        st.write("#### Missing Values")
        missing_values = df.isnull().sum().reset_index()
        missing_values.columns = ['Column', 'Missing Values']
        missing_values['Percentage'] = (missing_values['Missing Values'] / len(df)) * 100
        
        # Display missing values
        if missing_values['Missing Values'].sum() > 0:
            st.dataframe(missing_values[missing_values['Missing Values'] > 0])
            
            # Plot missing values
            fig = px.bar(
                missing_values[missing_values['Missing Values'] > 0], 
                x='Column', 
                y='Percentage', 
                title='Missing Values Percentage',
                color='Percentage'
            )
            st.plotly_chart(fig)
        else:
            st.success("No missing values in this dataset!")
    
    # Statistical summary
    with st.expander("Statistical Summary"):
        # Get only numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe())
            
            # Distribution of numerical columns
            st.write("#### Distributions")
            
            # Let user select a column to visualize
            selected_col = st.selectbox("Select column to visualize", numeric_cols)
            
            # Plot distribution
            fig = plots.plot_distribution(df, selected_col)
            st.pyplot(fig)
        else:
            st.warning("No numerical columns found in this dataset.")
    
    # Correlation analysis
    if df.select_dtypes(include=['int64', 'float64']).shape[1] > 1:
        with st.expander("Correlation Analysis"):
            st.write("#### Correlation Heatmap")
            
            # Generate correlation heatmap
            fig = plots.plot_correlation_heatmap(df)
            st.pyplot(fig)
            
            # Pairplot for selected features
            st.write("#### Scatter Matrix")
            
            # Let user select columns for pairplot (max 4)
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_cols) > 4:
                selected_cols = st.multiselect(
                    "Select columns for scatter matrix (max 4)",
                    numeric_cols,
                    default=numeric_cols[:4]
                )
                
                if len(selected_cols) > 4:
                    st.warning("Too many columns selected. Using only the first 4.")
                    selected_cols = selected_cols[:4]
                
                if selected_cols:
                    with st.spinner("Generating scatter matrix..."):
                        # Limit to 1000 samples to avoid overcrowding
                        samples = min(1000, len(df))
                        fig = plots.plot_scatter_matrix(df, features=selected_cols, samples=samples)
                        st.pyplot(fig)
            else:
                with st.spinner("Generating scatter matrix..."):
                    # Limit to 1000 samples to avoid overcrowding
                    samples = min(1000, len(df))
                    fig = plots.plot_scatter_matrix(df, samples=samples)
                    st.pyplot(fig)
    
    # Dataset-specific visualizations
    with st.expander("Dataset-Specific Visualizations"):
        if dataset == "Crop Recommendation":
            # Crop count visualization
            st.write("#### Crop Distribution")
            if 'label' in df.columns:
                crop_counts = df['label'].value_counts().reset_index()
                crop_counts.columns = ['Crop', 'Count']
                
                fig = px.bar(
                    crop_counts, 
                    x='Crop', 
                    y='Count', 
                    title='Distribution of Crops',
                    color='Count'
                )
                st.plotly_chart(fig)
                
                # Feature comparison by crop
                st.write("#### Feature Comparison by Crop")
                
                # Let user select a feature to compare
                feature_options = [col for col in df.columns if col != 'label']
                selected_feature = st.selectbox("Select feature to compare", feature_options)
                
                # Box plot
                fig = px.box(
                    df, 
                    x='label', 
                    y=selected_feature, 
                    title=f'{selected_feature} by Crop Type',
                    color='label'
                )
                st.plotly_chart(fig)
                
                # Create a radar chart for selected crops
                st.write("#### Crop Requirements Comparison")
                
                # Let user select crops to compare
                available_crops = df['label'].unique().tolist()
                selected_crops = st.multiselect(
                    "Select crops to compare (max 5)",
                    available_crops,
                    default=available_crops[:3] if len(available_crops) > 2 else available_crops
                )
                
                if len(selected_crops) > 5:
                    st.warning("Too many crops selected. Using only the first 5.")
                    selected_crops = selected_crops[:5]
                
                if selected_crops:
                    # Calculate mean values for each feature by crop
                    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
                    radar_df = df[df['label'].isin(selected_crops)].groupby('label')[feature_cols].mean().reset_index()
                    
                    # Create radar chart data
                    fig = go.Figure()
                    
                    for crop in radar_df['label']:
                        crop_data = radar_df[radar_df['label'] == crop].iloc[0]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=[crop_data[col] for col in feature_cols],
                            theta=feature_cols,
                            fill='toself',
                            name=crop
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                            )
                        ),
                        title="Crop Requirements Comparison"
                    )
                    
                    st.plotly_chart(fig)
        
        elif dataset == "Rainfall Data":
            # Time series visualization for rainfall data
            st.write("#### Rainfall Trends")
            
            if 'Area' in df.columns and 'Year' in df.columns and 'average_rain_fall_mm_per_year' in df.columns:
                # Get top 10 countries by average rainfall
                top_countries = df.groupby('Area')['average_rain_fall_mm_per_year'].mean().nlargest(10).index.tolist()
                
                # Let user select countries to compare
                selected_countries = st.multiselect(
                    "Select countries to compare",
                    df['Area'].unique().tolist(),
                    default=top_countries[:5] if top_countries else []
                )
                
                if selected_countries:
                    # Filter data for selected countries
                    filtered_df = df[df['Area'].isin(selected_countries)]
                    
                    # Create time series plot
                    fig = px.line(
                        filtered_df,
                        x='Year',
                        y='average_rain_fall_mm_per_year',
                        color='Area',
                        title='Average Rainfall by Year',
                        markers=True
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Rainfall distribution by country
                    st.write("#### Rainfall Distribution by Country")
                    
                    fig = px.box(
                        filtered_df,
                        x='Area',
                        y='average_rain_fall_mm_per_year',
                        color='Area',
                        title='Rainfall Distribution by Country'
                    )
                    
                    st.plotly_chart(fig)
        
        elif dataset == "Temperature Data":
            # Temperature trends visualization
            st.write("#### Temperature Trends")
            
            if 'country' in df.columns and 'year' in df.columns and 'avg_temp' in df.columns:
                # Get top 10 countries with most data points
                top_countries = df.groupby('country').size().nlargest(10).index.tolist()
                
                # Let user select countries to compare
                selected_countries = st.multiselect(
                    "Select countries to compare",
                    df['country'].unique().tolist(),
                    default=top_countries[:5] if top_countries else []
                )
                
                if selected_countries:
                    # Filter data for selected countries
                    filtered_df = df[df['country'].isin(selected_countries)]
                    
                    # Create time series plot
                    fig = px.line(
                        filtered_df,
                        x='year',
                        y='avg_temp',
                        color='country',
                        title='Average Temperature by Year',
                        markers=True
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Recent temperature trends (last 50 years)
                    st.write("#### Recent Temperature Trends (Last 50 Years)")
                    
                    # Calculate the most recent year in the dataset
                    most_recent_year = df['year'].max()
                    
                    # Filter data for the last 50 years
                    recent_df = filtered_df[filtered_df['year'] >= most_recent_year - 50]
                    
                    # Create time series plot for recent data
                    fig = px.line(
                        recent_df,
                        x='year',
                        y='avg_temp',
                        color='country',
                        title=f'Average Temperature Trends ({most_recent_year-50} to {most_recent_year})',
                        markers=True
                    )
                    
                    # Add trendlines
                    fig.update_traces(mode='markers+lines')
                    
                    st.plotly_chart(fig)
        
        elif dataset == "Yield Data":
            # Yield visualization
            st.write("#### Crop Yield Analysis")
            
            # This is a placeholder as we don't know the exact structure of the yield dataset
            # We'll adapt based on the columns available
            
            st.info("Select columns to analyze in the yield dataset")
            
            # Let user select columns for analysis
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols and categorical_cols:
                # Let user select one categorical and one numerical column
                selected_cat_col = st.selectbox("Select categorical column", categorical_cols)
                selected_num_col = st.selectbox("Select numerical column for analysis", numeric_cols)
                
                # Create a grouped bar chart
                fig = px.bar(
                    df,
                    x=selected_cat_col,
                    y=selected_num_col,
                    title=f'{selected_num_col} by {selected_cat_col}',
                    color=selected_cat_col
                )
                
                st.plotly_chart(fig)
                
                # Create a box plot
                fig = px.box(
                    df,
                    x=selected_cat_col,
                    y=selected_num_col,
                    title=f'Distribution of {selected_num_col} by {selected_cat_col}',
                    color=selected_cat_col
                )
                
                st.plotly_chart(fig)
        
        elif dataset == "Pesticides Data":
            # Pesticides visualization
            st.write("#### Pesticides Usage Analysis")
            
            # This is a placeholder as we don't know the exact structure of the pesticides dataset
            # We'll adapt based on the columns available
            
            st.info("Select columns to analyze in the pesticides dataset")
            
            # Let user select columns for analysis
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols and categorical_cols:
                # Let user select one categorical and one numerical column
                selected_cat_col = st.selectbox("Select categorical column", categorical_cols)
                selected_num_col = st.selectbox("Select numerical column for analysis", numeric_cols)
                
                # Create a grouped bar chart for top 15 categories
                top_categories = df.groupby(selected_cat_col)[selected_num_col].mean().nlargest(15).index.tolist()
                filtered_df = df[df[selected_cat_col].isin(top_categories)]
                
                fig = px.bar(
                    filtered_df,
                    x=selected_cat_col,
                    y=selected_num_col,
                    title=f'Top 15 {selected_cat_col} by {selected_num_col}',
                    color=selected_cat_col
                )
                
                st.plotly_chart(fig)
                
                # Create a time series if time column is available
                time_cols = [col for col in df.columns if any(time_word in col.lower() for time_word in ['year', 'date', 'time'])]
                
                if time_cols:
                    time_col = st.selectbox("Select time column for trend analysis", time_cols)
                    
                    # Check if the time column is usable
                    try:
                        # Try to convert to datetime if it's not already
                        if df[time_col].dtype == 'object':
                            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                        
                        # Group by time and selected category
                        grouped_df = df.groupby([time_col, selected_cat_col])[selected_num_col].mean().reset_index()
                        
                        # Create a line chart
                        fig = px.line(
                            grouped_df,
                            x=time_col,
                            y=selected_num_col,
                            color=selected_cat_col,
                            title=f'{selected_num_col} Trends by {selected_cat_col}',
                            markers=True
                        )
                        
                        st.plotly_chart(fig)
                    except:
                        st.warning(f"Could not create time series with column '{time_col}'.")
    
    # Download dataset option
    st.write("### Download Dataset")
    
    # Convert dataframe to CSV
    csv = df.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"{dataset.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    ) 

    if 'df' in locals() or 'df' in globals():
        get_csv_download(df, filename="data_exploration.csv")
        get_pdf_download(df.to_string(index=False), filename="data_exploration.pdf") 