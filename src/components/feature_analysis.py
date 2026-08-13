import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import plotly.graph_objects as go

from src.data import data_loader
from src.utils import preprocessing
from src.visualizations import plots

def render_feature_analysis():
    """Render the feature analysis page"""
    st.title("Feature Analysis")
    st.subheader("Analyze how different factors affect crop yield")
    
    # Dataset selection
    dataset = st.selectbox(
        "Select Dataset for Analysis",
        ["Crop Recommendation", "Yield Data"]
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
    
    if df is None:
        st.error(f"Failed to load {dataset} dataset. Please check if the file exists.")
        return
    
    # Main analysis options
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Model Comparison & Hybrid Evaluation", "Feature Correlation", "Feature Importance", "Feature Distribution", "Dimensionality Reduction"]
    )
    
    if analysis_type == "Model Comparison & Hybrid Evaluation":
        show_model_comparison_and_hybrid_eval()
    elif analysis_type == "Feature Correlation":
        show_feature_correlation(df, dataset)
    elif analysis_type == "Feature Importance":
        show_feature_importance(df, dataset)
    elif analysis_type == "Feature Distribution":
        show_feature_distribution(df, dataset)
    elif analysis_type == "Dimensionality Reduction":
        show_dimensionality_reduction(df, dataset)

import os
import json

def show_model_comparison_and_hybrid_eval():
    """Show comprehensive evaluation of ML models, DNN, and Hybrid ML-DL Model."""
    st.write("### 📊 Model Benchmark & Hybrid ML-DL Evaluation")
    st.write("Comparative evaluation across conventional ML models, Tabular DNN, and the Hybrid ML-DL architecture.")

    metrics_file = os.path.join('models', 'model_comparison_metrics.json')
    crop_metrics_file = os.path.join('models', 'crop_specific_metrics.json')

    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_list = json.load(f)
        
        comp_df = pd.DataFrame(metrics_list)
        st.write("#### 🏆 Model Performance Summary Table")
        st.dataframe(comp_df.style.format({'R2': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}'}), use_container_width=True)

        # Bar chart comparison for R2 score
        fig_r2 = px.bar(
            comp_df,
            x='Model',
            y='R2',
            color='R2',
            title='Model R² Score Comparison (Higher is better)',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_r2, use_container_width=True)

        # Bar chart for RMSE
        fig_rmse = px.bar(
            comp_df,
            x='Model',
            y='RMSE',
            color='RMSE',
            title='Model Root Mean Squared Error (RMSE) (Lower is better)',
            color_continuous_scale='Reds_r'
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    else:
        st.info("No benchmark metrics file found. Run model training to populate metrics.")

    # Render Crop-Specific Evaluation if available
    st.markdown("---")
    st.write("### 🌾 Crop-Specific Hybrid Model Evaluation")
    st.write("Comparing Global Hybrid ML-DL Model vs Crop-Specific Trained Hybrid Models for key crops.")

    if os.path.exists(crop_metrics_file):
        with open(crop_metrics_file, 'r') as f:
            crop_metrics = json.load(f)
        
        crop_df = pd.DataFrame(crop_metrics)
        st.dataframe(crop_df.style.format({
            'Global Hybrid R2': '{:.4f}',
            'Crop-Specific Hybrid R2': '{:.4f}',
            'Global Hybrid RMSE': '{:.2f}',
            'Crop-Specific Hybrid RMSE': '{:.2f}',
            'Global Hybrid MAE': '{:.2f}',
            'Crop-Specific Hybrid MAE': '{:.2f}'
        }), use_container_width=True)

        # Grouped bar chart comparing MAE
        crop_chart_df = pd.melt(
            crop_df,
            id_vars=['Crop'],
            value_vars=['Global Hybrid MAE', 'Crop-Specific Hybrid MAE'],
            var_name='Model Type',
            value_name='MAE Error'
        )
        fig_crop = px.bar(
            crop_chart_df,
            x='Crop',
            y='MAE Error',
            color='Model Type',
            barmode='group',
            title='Global vs Crop-Specific Hybrid Model MAE Error'
        )
        st.plotly_chart(fig_crop, use_container_width=True)

def show_feature_correlation(df, dataset):
    """Show feature correlation analysis"""
    st.write("### Feature Correlation Analysis")
    st.write("Examine how different features are correlated with each other.")
    
    # Get only numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Not enough numerical features for correlation analysis.")
        return
    
    # Correlation method selection
    corr_method = st.selectbox(
        "Select Correlation Method",
        ["Pearson", "Spearman", "Kendall"],
        index=0
    )
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr(method=corr_method.lower())
    
    # Show heatmap
    st.write("#### Correlation Heatmap")
    
    # Create heatmap using plotly for interactivity
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        labels=dict(color="Correlation"),
        title=f"{corr_method} Correlation Matrix"
    )
    
    st.plotly_chart(fig)
    
    # Show pairwise scatter plots
    st.write("#### Pairwise Relationship Explorer")
    
    # Let user select features to explore
    selected_features = st.multiselect(
        "Select features to explore (max 4)",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))]
    )
    
    if len(selected_features) > 4:
        st.warning("Too many features selected. Using only the first 4.")
        selected_features = selected_features[:4]
    
    if len(selected_features) >= 2:
        # If we have a target column (like 'label' in crop recommendation)
        hue_col = None
        if dataset == "Crop Recommendation" and 'label' in df.columns:
            hue_col = 'label'
        
        # Create pairplot
        with st.spinner("Generating pairwise plots..."):
            # Limit to 1000 samples to avoid overcrowding
            samples = min(1000, len(df))
            fig = plots.plot_scatter_matrix(df, features=selected_features, samples=samples, hue=hue_col)
            st.pyplot(fig)
    else:
        st.info("Please select at least 2 features to create pairwise plots.")

def show_feature_importance(df, dataset):
    """Show feature importance analysis"""
    st.write("### Feature Importance Analysis")
    st.write("Identify which features have the strongest influence on the target variable.")
    
    target_col = None
    
    # Identify target column based on dataset
    if dataset == "Crop Recommendation":
        if 'label' in df.columns:
            st.info("For Crop Recommendation dataset, we'll analyze feature importance for distinguishing crop types.")
            target_col = 'label'
        else:
            st.error("Could not find 'label' column in the Crop Recommendation dataset.")
            return
    elif dataset == "Yield Data":
        # Try to identify yield column - this may need to be adapted based on the actual dataset structure
        potential_yield_cols = [col for col in df.columns if 'yield' in col.lower()]
        
        if potential_yield_cols:
            target_col = st.selectbox("Select target variable (yield)", potential_yield_cols)
        else:
            # Let user select the target column
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("Select target variable (yield)", numeric_cols)
            else:
                st.error("No numerical columns found for yield prediction.")
                return
    
    if not target_col:
        st.error("Could not identify a target column for feature importance analysis.")
        return
    
    # Get numeric features (excluding target for yield data)
    if dataset == "Yield Data":
        feature_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != target_col]
    else:  # For Crop Recommendation, we use all numeric features
        feature_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(feature_cols) < 1:
        st.warning("Not enough numerical features for importance analysis.")
        return
    
    # Feature importance methods
    method = st.selectbox(
        "Select Feature Importance Method",
        ["Mutual Information", "Random Forest", "XGBoost"],
        index=0
    )
    
    # Calculate and display feature importance
    if dataset == "Yield Data":
        show_regression_feature_importance(df, feature_cols, target_col, method)
    else:  # Crop Recommendation
        show_classification_feature_importance(df, feature_cols, target_col, method)

def show_regression_feature_importance(df, feature_cols, target_col, method):
    """Show feature importance for regression problems"""
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Calculate feature importance based on selected method
    feature_importance = None
    if method == "Mutual Information":
        with st.spinner("Calculating mutual information..."):
            mi_scores = mutual_info_regression(X, y)
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': mi_scores
            })
    
    elif method == "Random Forest":
        with st.spinner("Training Random Forest model..."):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            })
    
    elif method == "XGBoost":
        with st.spinner("Training XGBoost model..."):
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            })
    
    if feature_importance is not None:
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Create interactive bar chart
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Feature Importance using {method}",
            color='Importance'
        )
        
        fig.update_layout(
            yaxis=dict(autorange="reversed"),  # Highest importance at the top
            height=500
        )
        
        st.plotly_chart(fig)
        
        # Analyze top features
        top_features = feature_importance.head(3)['Feature'].tolist()
        
        st.write("#### Top Features Analysis")
        st.write(f"The top 3 most important features are: {', '.join(top_features)}")
        
        # Show scatter plots of top features vs target
        for feature in top_features:
            fig = px.scatter(
                df,
                x=feature,
                y=target_col,
                title=f"{feature} vs {target_col}",
                trendline="ols"
            )
            
            st.plotly_chart(fig)

def show_classification_feature_importance(df, feature_cols, target_col, method):
    """Show feature importance for classification problems (like crop recommendation)"""
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Replace categorical target with numeric codes temporarily for mutual information
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Calculate feature importance based on selected method
    feature_importance = None
    if method == "Mutual Information":
        with st.spinner("Calculating mutual information..."):
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(X, y_encoded)
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': mi_scores
            })
    
    elif method == "Random Forest":
        with st.spinner("Training Random Forest model..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            })
    
    elif method == "XGBoost":
        with st.spinner("Training XGBoost model..."):
            model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            })
    
    if feature_importance is not None:
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Create interactive bar chart
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Feature Importance using {method}",
            color='Importance'
        )
        
        fig.update_layout(
            yaxis=dict(autorange="reversed"),  # Highest importance at the top
            height=500
        )
        
        st.plotly_chart(fig)
        
        # Analyze top features
        top_features = feature_importance.head(3)['Feature'].tolist()
        
        st.write("#### Top Features Analysis")
        st.write(f"The top 3 most important features are: {', '.join(top_features)}")
        
        # Show box plots of top features by crop type
        for feature in top_features:
            fig = px.box(
                df,
                x=target_col,
                y=feature,
                title=f"{feature} by Crop Type",
                color=target_col
            )
            
            st.plotly_chart(fig)

def show_feature_distribution(df, dataset):
    """Show feature distribution analysis"""
    st.write("### Feature Distribution Analysis")
    st.write("Examine the distribution of features across different categories.")
    
    # Get categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle dataset-specific analysis
    if dataset == "Crop Recommendation" and 'label' in df.columns:
        # For crop recommendation, analyze numerical features across different crops
        
        # Let user select numerical features to analyze
        selected_features = st.multiselect(
            "Select features to analyze",
            numerical_cols,
            default=numerical_cols[:min(3, len(numerical_cols))]
        )
        
        if not selected_features:
            st.info("Please select at least one feature to analyze.")
            return
        
        # Let user select crops to analyze
        available_crops = df['label'].unique().tolist()
        selected_crops = st.multiselect(
            "Select crops to compare",
            available_crops,
            default=available_crops[:min(5, len(available_crops))]
        )
        
        if not selected_crops:
            st.info("Please select at least one crop to analyze.")
            return
        
        # Filter data for selected crops
        filtered_df = df[df['label'].isin(selected_crops)]
        
        # Create distribution plots for each selected feature
        for feature in selected_features:
            st.write(f"#### Distribution of {feature}")
            
            # Create histogram with density curves
            fig = px.histogram(
                filtered_df,
                x=feature,
                color='label',
                marginal='violin',
                opacity=0.7,
                barmode='overlay',
                histnorm='probability density',
                title=f"Distribution of {feature} by Crop Type"
            )
            
            st.plotly_chart(fig)
            
            # Create box plot
            fig = px.box(
                filtered_df,
                x='label',
                y=feature,
                color='label',
                title=f"Box Plot of {feature} by Crop Type"
            )
            
            st.plotly_chart(fig)
    
    elif dataset == "Yield Data":
        # For yield data, let user select categorical and numerical columns to analyze
        
        if not categorical_cols:
            st.warning("No categorical columns found in the yield dataset.")
            return
        
        if not numerical_cols:
            st.warning("No numerical columns found in the yield dataset.")
            return
        
        # Let user select a categorical column for grouping
        groupby_col = st.selectbox(
            "Select categorical column for grouping",
            categorical_cols
        )
        
        # Let user select numerical features to analyze
        selected_features = st.multiselect(
            "Select features to analyze",
            numerical_cols,
            default=numerical_cols[:min(3, len(numerical_cols))]
        )
        
        if not selected_features:
            st.info("Please select at least one feature to analyze.")
            return
        
        # Get top categories (to avoid overcrowding with too many categories)
        top_categories = df[groupby_col].value_counts().nlargest(10).index.tolist()
        
        # Let user select categories to analyze
        selected_categories = st.multiselect(
            f"Select {groupby_col} values to compare",
            df[groupby_col].unique().tolist(),
            default=top_categories[:min(5, len(top_categories))]
        )
        
        if not selected_categories:
            st.info(f"Please select at least one {groupby_col} value to analyze.")
            return
        
        # Filter data for selected categories
        filtered_df = df[df[groupby_col].isin(selected_categories)]
        
        # Create distribution plots for each selected feature
        for feature in selected_features:
            st.write(f"#### Distribution of {feature}")
            
            # Create histogram with density curves
            fig = px.histogram(
                filtered_df,
                x=feature,
                color=groupby_col,
                marginal='violin',
                opacity=0.7,
                barmode='overlay',
                histnorm='probability density',
                title=f"Distribution of {feature} by {groupby_col}"
            )
            
            st.plotly_chart(fig)
            
            # Create box plot
            fig = px.box(
                filtered_df,
                x=groupby_col,
                y=feature,
                color=groupby_col,
                title=f"Box Plot of {feature} by {groupby_col}"
            )
            
            st.plotly_chart(fig)

def show_dimensionality_reduction(df, dataset):
    """Show dimensionality reduction analysis"""
    st.write("### Dimensionality Reduction Analysis")
    st.write("Visualize high-dimensional data in a lower-dimensional space.")
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numerical_cols) < 3:
        st.warning("Not enough numerical features for dimensionality reduction.")
        return
    
    # Select method
    method = st.selectbox(
        "Select Dimensionality Reduction Method",
        ["PCA", "t-SNE"],
        index=0
    )
    
    # Let user select features for analysis
    selected_features = st.multiselect(
        "Select features for analysis",
        numerical_cols,
        default=numerical_cols
    )
    
    if len(selected_features) < 3:
        st.info("Please select at least 3 features for meaningful dimensionality reduction.")
        return
    
    # Prepare data
    X = df[selected_features]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Color encoding
    color_col = None
    if dataset == "Crop Recommendation" and 'label' in df.columns:
        color_col = 'label'
    else:
        # Let user select a categorical column for color encoding if available
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            color_col = st.selectbox(
                "Select column for color encoding",
                ['None'] + categorical_cols
            )
            
            if color_col == 'None':
                color_col = None
    
    # Apply dimensionality reduction
    if method == "PCA":
        with st.spinner("Applying PCA..."):
            pca = PCA(n_components=3)
            X_reduced = pca.fit_transform(X_scaled)
            
            # Create a DataFrame with the reduced dimensions
            pca_df = pd.DataFrame(
                X_reduced,
                columns=['PC1', 'PC2', 'PC3']
            )
            
            # Add color column if available
            if color_col:
                pca_df[color_col] = df[color_col].values
            
            # Explained variance
            explained_variance = pca.explained_variance_ratio_
            
            # Create 3D scatter plot
            if color_col:
                fig = px.scatter_3d(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color=color_col,
                    title=f"PCA Visualization",
                    labels={
                        'PC1': f"PC1 ({explained_variance[0]:.2%})",
                        'PC2': f"PC2 ({explained_variance[1]:.2%})",
                        'PC3': f"PC3 ({explained_variance[2]:.2%})"
                    }
                )
            else:
                fig = px.scatter_3d(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    title=f"PCA Visualization",
                    labels={
                        'PC1': f"PC1 ({explained_variance[0]:.2%})",
                        'PC2': f"PC2 ({explained_variance[1]:.2%})",
                        'PC3': f"PC3 ({explained_variance[2]:.2%})"
                    }
                )
            
            fig.update_layout(height=700)
            st.plotly_chart(fig)
            
            # Show explained variance
            st.write("#### Explained Variance")
            
            fig = px.bar(
                x=[f"PC{i+1}" for i in range(len(explained_variance))],
                y=explained_variance,
                labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
                title="Explained Variance by Principal Component"
            )
            
            # Add cumulative explained variance
            cumulative_variance = np.cumsum(explained_variance)
            fig.add_trace(
                go.Scatter(
                    x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
                    y=cumulative_variance,
                    mode='lines+markers',
                    name='Cumulative Explained Variance',
                    yaxis='y2'
                )
            )
            
            fig.update_layout(
                yaxis2=dict(
                    title='Cumulative Explained Variance',
                    overlaying='y',
                    side='right'
                )
            )
            
            st.plotly_chart(fig)
            
            # Show feature loadings
            st.write("#### Feature Loadings")
            st.write("This shows how each original feature contributes to the principal components.")
            
            loadings = pca.components_.T
            loadings_df = pd.DataFrame(
                loadings,
                columns=[f"PC{i+1}" for i in range(loadings.shape[1])],
                index=selected_features
            )
            
            # Create heatmap of loadings
            fig = px.imshow(
                loadings_df,
                labels=dict(x="Principal Component", y="Feature", color="Loading"),
                x=loadings_df.columns,
                y=loadings_df.index,
                color_continuous_scale='RdBu_r',
                title="Feature Loadings"
            )
            
            st.plotly_chart(fig)
    
    elif method == "t-SNE":
        with st.spinner("Applying t-SNE (this may take a while)..."):
            from sklearn.manifold import TSNE
            
            # t-SNE parameters
            perplexity = st.slider("Perplexity", 5, 50, 30)
            learning_rate = st.slider("Learning Rate", 10, 1000, 200)
            
            # Apply t-SNE
            tsne = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
            X_reduced = tsne.fit_transform(X_scaled)
            
            # Create a DataFrame with the reduced dimensions
            tsne_df = pd.DataFrame(
                X_reduced,
                columns=['TSNE1', 'TSNE2', 'TSNE3']
            )
            
            # Add color column if available
            if color_col:
                tsne_df[color_col] = df[color_col].values
            
            # Create 3D scatter plot
            if color_col:
                fig = px.scatter_3d(
                    tsne_df,
                    x='TSNE1',
                    y='TSNE2',
                    z='TSNE3',
                    color=color_col,
                    title=f"t-SNE Visualization"
                )
            else:
                fig = px.scatter_3d(
                    tsne_df,
                    x='TSNE1',
                    y='TSNE2',
                    z='TSNE3',
                    title=f"t-SNE Visualization"
                )
            
            fig.update_layout(height=700)
            st.plotly_chart(fig) 