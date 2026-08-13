import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import streamlit as st

# Set the style for matplotlib plots - wrapped in try-except to handle warnings module issues
try:
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")
except Exception as e:
    # Fallback to default style if there's an issue
    pass

def plot_correlation_heatmap(df, cmap='coolwarm', figsize=(10, 8)):
    """
    Plot correlation heatmap for numerical features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    cmap : str, default='coolwarm'
        Colormap for the heatmap
    figsize : tuple, default=(10, 8)
        Figure size
    
    Returns:
    --------
    matplotlib figure
    """
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap=cmap, 
        linewidths=0.5, 
        ax=ax, 
        fmt='.2f',
        mask=np.triu(corr_matrix, k=1)  # Show only lower triangle
    )
    
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importance_df, figsize=(12, 8), top_n=15):
    """
    Plot feature importance
    
    Parameters:
    -----------
    feature_importance_df : pandas DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    figsize : tuple, default=(12, 8)
        Figure size
    top_n : int, default=15
        Number of top features to display
        
    Returns:
    --------
    matplotlib figure
    """
    # Get top N features
    top_features = feature_importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    bars = ax.barh(
        top_features['Feature'], 
        top_features['Importance'],
        color=sns.color_palette('viridis', len(top_features))
    )
    
    # Add values to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.3f}', 
            ha='left', 
            va='center'
        )
    
    # Set labels and title
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance', fontsize=14)
    
    # Invert y-axis to have the highest importance at the top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    return fig

def plot_scatter_matrix(df, features=None, figsize=(15, 15), samples=None, hue=None):
    """
    Plot scatter matrix for selected features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    features : list, default=None
        List of features to include (default: all numerical)
    figsize : tuple, default=(15, 15)
        Figure size
    samples : int, default=None
        Number of samples to use (to avoid overcrowding)
    hue : str, default=None
        Column name for color encoding
        
    Returns:
    --------
    matplotlib figure
    """
    # If features not specified, use all numerical columns
    if features is None:
        features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Limit to a maximum of 6 features to avoid overcrowding
    if len(features) > 6:
        features = features[:6]
    
    # If hue is specified, add it to the features if not already included
    plot_df = df[features].copy()
    if hue is not None and hue not in features:
        plot_df[hue] = df[hue]
    
    # If samples specified, take a sample
    if samples is not None and samples < len(plot_df):
        plot_df = plot_df.sample(samples, random_state=42)
    
    # Create figure
    fig = sns.pairplot(
        plot_df, 
        hue=hue, 
        diag_kind='kde', 
        plot_kws={'alpha': 0.6},
        height=figsize[0]/len(features)
    )
    
    fig.fig.suptitle('Scatter Matrix', y=1.02, fontsize=16)
    plt.tight_layout()
    
    return fig.fig

def plot_distribution(df, column, bins=30, figsize=(12, 6)):
    """
    Plot distribution of a numerical column
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    column : str
        Column name to plot
    bins : int, default=30
        Number of bins for histogram
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with KDE
    sns.histplot(df[column], bins=bins, kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {column}', fontsize=12)
    ax1.set_xlabel(column, fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    
    # Box plot
    sns.boxplot(y=df[column], ax=ax2)
    ax2.set_title(f'Box Plot of {column}', fontsize=12)
    ax2.set_ylabel(column, fontsize=10)
    
    plt.tight_layout()
    
    return fig

def plot_time_series(df, x, y, title=None, figsize=(12, 6)):
    """
    Plot time series data
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    x : str
        Column name for x-axis (usually time)
    y : str or list
        Column name(s) for y-axis
    title : str, default=None
        Plot title
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle single or multiple y columns
    if isinstance(y, list):
        for col in y:
            ax.plot(df[x], df[col], marker='o', linewidth=2, label=col)
        ax.legend()
    else:
        ax.plot(df[x], df[y], marker='o', linewidth=2, color='blue')
    
    # Set labels and title
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y if not isinstance(y, list) else 'Value', fontsize=12)
    ax.set_title(title or f'Time Series Plot of {y}', fontsize=14)
    
    # Rotate x-axis labels if they are categorical or dates
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def plot_pred_vs_actual(y_true, y_pred, title=None, figsize=(10, 6)):
    """
    Plot predicted vs actual values
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str, default=None
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, color='blue')
    
    # Diagonal line (perfect predictions)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Set labels and title
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title or 'Predicted vs Actual Values', fontsize=14)
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(
        0.05, 0.95, 
        f'Correlation: {correlation:.3f}', 
        transform=ax.transAxes, 
        fontsize=12, 
        verticalalignment='top'
    )
    
    plt.tight_layout()
    
    return fig

def plot_residuals(y_true, y_pred, figsize=(12, 6)):
    """
    Plot residuals
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib figure
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot of residuals vs predicted values
    ax1.scatter(y_pred, residuals, alpha=0.6, color='blue')
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residuals vs Predicted Values', fontsize=14)
    
    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Residuals', fontsize=14)
    
    plt.tight_layout()
    
    return fig

def plot_interactive_map(df, lat_col, lon_col, color_col=None, size_col=None, hover_name=None, zoom=3):
    """
    Create an interactive map using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with geographical data
    lat_col : str
        Column name for latitude
    lon_col : str
        Column name for longitude
    color_col : str, default=None
        Column name for color encoding
    size_col : str, default=None
        Column name for marker size
    hover_name : str, default=None
        Column name for hover information
    zoom : int, default=3
        Initial zoom level
        
    Returns:
    --------
    plotly figure
    """
    # Create figure
    fig = px.scatter_mapbox(
        df, 
        lat=lat_col, 
        lon=lon_col, 
        color=color_col,
        size=size_col,
        hover_name=hover_name,
        zoom=zoom,
        mapbox_style="open-street-map"
    )
    
    fig.update_layout(
        margin={"r":0, "t":0, "l":0, "b":0},
        height=600
    )
    
    return fig

def plot_grouped_bar(df, x, y, group_by, title=None, figsize=(12, 6)):
    """
    Create a grouped bar chart
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    group_by : str
        Column name for grouping
    title : str, default=None
        Plot title
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Grouped bar chart
    sns.barplot(x=x, y=y, hue=group_by, data=df, ax=ax)
    
    # Set labels and title
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.set_title(title or f'{y} by {x} and {group_by}', fontsize=14)
    
    # Rotate x-axis labels if they are long
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def plot_interactive_scatter(df, x, y, color=None, size=None, hover_name=None, title=None):
    """
    Create an interactive scatter plot using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    color : str, default=None
        Column name for color encoding
    size : str, default=None
        Column name for marker size
    hover_name : str, default=None
        Column name for hover information
    title : str, default=None
        Plot title
        
    Returns:
    --------
    plotly figure
    """
    # Create figure
    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        color=color,
        size=size,
        hover_name=hover_name,
        title=title or f'{y} vs {x}'
    )
    
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        height=600
    )
    
    return fig

def plot_interactive_line(df, x, y, color=None, line_dash=None, title=None):
    """
    Create an interactive line chart using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    x : str
        Column name for x-axis
    y : str or list
        Column name(s) for y-axis
    color : str, default=None
        Column name for color encoding
    line_dash : str, default=None
        Column name for line dash pattern
    title : str, default=None
        Plot title
        
    Returns:
    --------
    plotly figure
    """
    # Create figure
    fig = px.line(
        df, 
        x=x, 
        y=y, 
        color=color,
        line_dash=line_dash,
        title=title or f'{y} over {x}'
    )
    
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y if not isinstance(y, list) else 'Value',
        height=500
    )
    
    return fig

def plot_streamlit_metric_cards(metrics_dict, n_cols=3):
    """
    Display metrics as Streamlit metric cards
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metrics with format {name: (value, delta)}
        where delta is the change (optional)
    n_cols : int, default=3
        Number of columns
    """
    # Create columns
    cols = st.columns(n_cols)
    
    # Fill columns with metrics
    for i, (metric_name, metric_value) in enumerate(metrics_dict.items()):
        col_idx = i % n_cols
        
        if isinstance(metric_value, tuple) and len(metric_value) == 2:
            # Value and delta provided
            value, delta = metric_value
            cols[col_idx].metric(metric_name, value, delta)
        else:
            # Only value provided
            cols[col_idx].metric(metric_name, metric_value)
            
def create_yield_comparison_chart(df, crop_type=None, region=None, years=None):
    """
    Create a chart comparing yield over years for selected crop and region
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with yield data
    crop_type : str, default=None
        Crop type to filter
    region : str, default=None
        Region to filter
    years : list, default=None
        Years to include
        
    Returns:
    --------
    plotly figure
    """
    # Filter data based on parameters
    filtered_df = df.copy()
    
    if crop_type:
        filtered_df = filtered_df[filtered_df['crop_type'] == crop_type]
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if years:
        filtered_df = filtered_df[filtered_df['year'].isin(years)]
    
    # Create figure
    if 'yield' in filtered_df.columns and 'year' in filtered_df.columns:
        fig = px.line(
            filtered_df,
            x='year',
            y='yield',
            color='crop_type' if not crop_type else None,
            line_group='region' if not region else None,
            markers=True,
            title=f"Crop Yield Over Time {f'for {crop_type}' if crop_type else ''} {f'in {region}' if region else ''}"
        )
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Yield (tons/hectare)",
            legend_title="Crop Type" if not crop_type else "Region",
            height=500
        )
        
        return fig
    else:
        # Create a dummy figure if the required columns are not available
        fig = go.Figure()
        fig.add_annotation(
            text="Yield comparison data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig 