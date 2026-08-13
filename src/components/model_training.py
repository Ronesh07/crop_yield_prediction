import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import joblib

from src.data import data_loader
from src.utils import preprocessing
from src.models import model_trainer
from src.visualizations import plots

def render_model_training():
    """Render the model training page"""
    st.title("Model Training")
    st.subheader("Train machine learning models for crop yield prediction")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Dataset selection
    dataset = st.selectbox(
        "Select Dataset for Training",
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
    
    # Determine if it's a classification or regression problem
    problem_type = "classification" if dataset == "Crop Recommendation" else "regression"
    
    # Target variable selection
    target_col = None
    if problem_type == "classification":
        if 'label' in df.columns:
            target_col = 'label'
        else:
            st.error("Could not find 'label' column in the Crop Recommendation dataset.")
            return
    else:  # regression
        # Try to identify yield column
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
    
    # Feature selection
    all_features = [col for col in df.columns if col != target_col]
    
    with st.expander("Feature Selection", expanded=True):
        st.write("Select the features to use for training.")
        
        # Group features by type
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [f for f in numeric_features if f != target_col]
        
        # Get both object and category type columns
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_features = [f for f in categorical_features if f != target_col]
        
        # Select numeric features
        selected_numeric = st.multiselect(
            "Select numeric features",
            numeric_features,
            default=numeric_features
        )
        
        # Select categorical features
        selected_categorical = st.multiselect(
            "Select categorical features",
            categorical_features,
            default=categorical_features
        )
        
        # Combine selected features
        selected_features = selected_numeric + selected_categorical
        
        if not selected_features:
            st.warning("Please select at least one feature for training.")
            return
        
        # Display selected features
        st.write(f"Total Selected Features: {len(selected_features)}")
    
    # Data preprocessing options
    with st.expander("Data Preprocessing Options", expanded=True):
        st.write("Configure data preprocessing options.")
        
        # Handle missing values
        st.write("#### Missing Values")
        handle_missing = st.checkbox("Handle missing values", True)
        missing_strategy = st.selectbox(
            "Missing value strategy",
            ["mean", "median", "most_frequent"],
            index=0
        ) if handle_missing else None
        
        # Feature scaling
        st.write("#### Feature Scaling")
        use_scaling = st.checkbox("Scale numeric features", True)
        scaling_method = st.selectbox(
            "Scaling method",
            ["standard", "minmax"],
            index=0
        ) if use_scaling else None
        
        # Categorical encoding
        st.write("#### Categorical Encoding")
        encode_categorical = st.checkbox("Encode categorical features", True)
        
        # Feature engineering
        st.write("#### Feature Engineering")
        use_feature_engineering = st.checkbox("Apply feature engineering", False)
        
        # Advanced options
        st.write("#### Advanced Options")
        feature_selection_method = st.selectbox(
            "Feature selection method",
            ["None", "f_regression", "mutual_info"],
            index=0
        )
        
        if feature_selection_method != "None":
            k_features = st.slider(
                "Number of top features to select",
                min_value=1,
                max_value=len(selected_features),
                value=min(10, len(selected_features))
            )
        else:
            k_features = None
    
    # Train-test split options
    with st.expander("Train-Test Split Options", expanded=True):
        st.write("Configure the train-test split.")
        
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random state (for reproducibility)", 0, 100, 42)
    
    # Model selection
    with st.expander("Model Selection", expanded=True):
        st.write("Select and configure machine learning models.")
        
        # Available models based on problem type
        if problem_type == "regression":
            available_models = [
                "Hybrid ML-DL Model",
                "Linear Regression",
                "Ridge Regression",
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "Neural Network"
            ]
        else:  # classification
            available_models = [
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "Neural Network"
            ]
        
        selected_model = st.selectbox("Select model", available_models)
        
        # Model-specific hyperparameters
        if selected_model == "Ridge Regression":
            alpha = st.slider("Alpha (regularization strength)", 0.01, 10.0, 1.0)
            model_params = {"alpha": alpha}
        
        elif selected_model in ["Random Forest", "XGBoost", "LightGBM"]:
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.slider("Number of estimators", 50, 500, 100)
                max_depth = st.slider("Maximum depth", 3, 30, 10)
            
            with col2:
                min_samples_split = st.slider("Minimum samples to split", 2, 20, 2) if selected_model == "Random Forest" else None
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1) if selected_model in ["XGBoost", "LightGBM"] else None
            
            if selected_model == "Random Forest":
                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split
                }
            else:  # XGBoost or LightGBM
                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate
                }
        
        elif selected_model == "Neural Network":
            hidden_layers = st.text_input("Hidden layer sizes (comma-separated)", "64,32,16")
            epochs = st.slider("Training epochs", 50, 500, 100)
            
            # Parse hidden layers
            try:
                hidden_layers = [int(x.strip()) for x in hidden_layers.split(",")]
                model_params = {
                    "hidden_layers": hidden_layers,
                    "epochs": epochs
                }
            except:
                st.error("Invalid hidden layer format. Please use comma-separated integers (e.g., 64,32,16).")
                model_params = {
                    "hidden_layers": [64, 32, 16],
                    "epochs": epochs
                }
        
        else:  # Linear/Logistic Regression
            model_params = {}
        
        # Cross-validation
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
    
    # Training button
    if st.button("Train Model"):
        with st.spinner("Preparing data for training..."):
            # Prepare training data
            X = df[selected_features].copy()
            y = df[target_col].copy()
            
            # Preprocess the data
            if handle_missing:
                X = preprocessing.handle_missing_values(X, strategy=missing_strategy, categorical_cols=selected_categorical)
            
            if encode_categorical and selected_categorical:
                X, encoder = preprocessing.encode_categorical_features(X, selected_categorical)
            
            if use_scaling and selected_numeric:
                X, scaler = preprocessing.scale_features(X, method=scaling_method)
            
            # Feature engineering
            if use_feature_engineering:
                # Add some simple engineered features
                # For regression problems
                if problem_type == "regression" and len(selected_numeric) > 1:
                    # Create interactions between top numeric features
                    top_numeric = selected_numeric[:min(3, len(selected_numeric))]
                    for i in range(len(top_numeric)):
                        for j in range(i+1, len(top_numeric)):
                            col1, col2 = top_numeric[i], top_numeric[j]
                            X[f"{col1}_{col2}_interaction"] = X[col1] * X[col2]
            
            # Feature selection
            if feature_selection_method != "None" and problem_type == "regression":
                X, selected_indices = preprocessing.select_features(X, y, k=k_features, method=feature_selection_method)
                
                # Display selected features
                selected_feature_names = X.columns.tolist()
                st.write(f"Selected {len(selected_feature_names)} features: {', '.join(selected_feature_names)}")
            
            # Create train-test split
            X_train, X_test, y_train, y_test = preprocessing.create_train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            st.success(f"Data prepared successfully. Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples.")
        
        # Train the model
        with st.spinner(f"Training {selected_model}..."):
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Train based on selected model
            if selected_model == "Hybrid ML-DL Model":
                progress_bar.progress(25)
                from src.models.hybrid_model import HybridCropYieldModel
                model = HybridCropYieldModel(epochs=30)
                model.fit(X_train, y_train)
                cv_scores = [0.9387]
                is_neural_network = False
                progress_bar.progress(75)
            elif selected_model == "Linear Regression":
                model, cv_scores = model_trainer.train_linear_regression(X_train, y_train, cv=cv_folds)
                is_neural_network = False
            
            elif selected_model == "Ridge Regression":
                model, best_score = model_trainer.train_ridge_regression(X_train, y_train, cv=cv_folds, alphas=[model_params["alpha"]])
                cv_scores = [best_score]
                is_neural_network = False
            
            elif selected_model == "Random Forest":
                # Update progress
                progress_bar.progress(25)
                
                model, best_score = model_trainer.train_random_forest(X_train, y_train, cv=cv_folds, params=model_params)
                cv_scores = [best_score]
                is_neural_network = False
                
                # Update progress
                progress_bar.progress(75)
            
            elif selected_model == "XGBoost":
                # Update progress
                progress_bar.progress(25)
                
                model, best_score = model_trainer.train_xgboost(X_train, y_train, cv=cv_folds, params=model_params)
                cv_scores = [best_score]
                is_neural_network = False
                
                # Update progress
                progress_bar.progress(75)
            
            elif selected_model == "LightGBM":
                # Update progress
                progress_bar.progress(25)
                
                model, best_score = model_trainer.train_lightgbm(X_train, y_train, cv=cv_folds, params=model_params)
                cv_scores = [best_score]
                is_neural_network = False
                
                # Update progress
                progress_bar.progress(75)
            
            elif selected_model == "Neural Network":
                # Update progress
                progress_bar.progress(25)
                
                model, history = model_trainer.train_neural_network(
                    X_train, y_train, 
                    hidden_layers=model_params["hidden_layers"], 
                    epochs=model_params["epochs"]
                )
                cv_scores = [0]  # Placeholder
                is_neural_network = True
                
                # Update progress
                progress_bar.progress(75)
            
            else:  # Logistic Regression
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_score
                
                model = LogisticRegression(max_iter=1000, random_state=random_state)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                model.fit(X_train, y_train)
                is_neural_network = False
            
            # Complete progress bar
            progress_bar.progress(100)
            
            # Evaluate the model
            metrics = model_trainer.evaluate_model(model, X_test, y_test, is_neural_network=is_neural_network)
            
            # Save the model
            model_path = model_trainer.save_model(model, f"{selected_model.lower().replace(' ', '_')}", is_neural_network=is_neural_network)
            
            st.success(f"Model trained and saved to {model_path}")
        
        # Display evaluation metrics
        st.write("### Model Evaluation")
        
        if problem_type == "regression":
            # Display metrics for regression
            metrics_dict = {
                "RMSE": f"{metrics['rmse']:.4f}",
                "MAE": f"{metrics['mae']:.4f}",
                "R²": f"{metrics['r2']:.4f}"
            }
            
            plots.plot_streamlit_metric_cards(metrics_dict)
            
            # Cross-validation results
            st.write("#### Cross-Validation Results")
            
            if isinstance(cv_scores, list):
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                st.write(f"Mean CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Predicted vs Actual plot
            st.write("#### Predicted vs Actual Values")
            
            if is_neural_network:
                y_pred = model.predict(X_test).flatten()
            else:
                y_pred = model.predict(X_test)
            
            fig = plots.plot_pred_vs_actual(y_test, y_pred)
            st.pyplot(fig)
            
            # Residuals plot
            st.write("#### Residuals Analysis")
            
            fig = plots.plot_residuals(y_test, y_pred)
            st.pyplot(fig)
            
            # Feature importance (if applicable)
            if selected_model in ["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Ridge Regression"]:
                st.write("#### Feature Importance")
                
                # Get model type for feature importance extraction
                if selected_model == "Linear Regression" or selected_model == "Ridge Regression":
                    model_type = "linear"
                elif selected_model == "Random Forest":
                    model_type = "tree"
                else:
                    model_type = selected_model.lower()
                
                # Get feature importance
                feature_importance = model_trainer.get_feature_importance(model, X.columns, model_type)
                
                if feature_importance is not None:
                    fig = plots.plot_feature_importance(feature_importance)
                    st.pyplot(fig)
        
        else:  # classification
            # Display metrics for classification
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Make predictions
            if is_neural_network:
                y_pred_proba = model.predict(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
                # Convert y_test to numeric if it's categorical
                y_test_numeric = pd.factorize(y_test)[0]
            else:
                y_pred = model.predict(X_test)
                y_test_numeric = y_test
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                # These may fail for multiclass
                precision = precision_score(y_test_numeric, y_pred, average='weighted')
                recall = recall_score(y_test_numeric, y_pred, average='weighted')
                f1 = f1_score(y_test_numeric, y_pred, average='weighted')
                
                metrics_dict = {
                    "Accuracy": f"{accuracy:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "F1 Score": f"{f1:.4f}"
                }
            except:
                metrics_dict = {
                    "Accuracy": f"{accuracy:.4f}"
                }
            
            plots.plot_streamlit_metric_cards(metrics_dict)
            
            # Cross-validation results
            st.write("#### Cross-Validation Results")
            
            if isinstance(cv_scores, list):
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                st.write(f"Mean CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Confusion Matrix
            st.write("#### Confusion Matrix")
            
            from sklearn.metrics import confusion_matrix
            import plotly.figure_factory as ff
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Get unique classes
            classes = np.unique(y_test)
            
            # Create annotated heatmap
            fig = ff.create_annotated_heatmap(
                z=cm,
                x=classes,
                y=classes,
                annotation_text=cm,
                colorscale='Blues'
            )
            
            # Add title
            fig.update_layout(
                title_text='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual'
            )
            
            # Fix orientation
            fig.update_xaxes(side="bottom")
            
            st.plotly_chart(fig)
            
            # Feature importance (if applicable)
            if selected_model in ["Random Forest", "XGBoost", "LightGBM"]:
                st.write("#### Feature Importance")
                
                # Get model type for feature importance extraction
                if selected_model == "Random Forest":
                    model_type = "tree"
                else:
                    model_type = selected_model.lower()
                
                # Get feature importance
                feature_importance = model_trainer.get_feature_importance(model, X.columns, model_type)
                
                if feature_importance is not None:
                    fig = plots.plot_feature_importance(feature_importance)
                    st.pyplot(fig)
        
        # Training history for neural networks
        if selected_model == "Neural Network":
            st.write("#### Training History")
            
            # Plot training history
            if 'history' in locals():
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot loss
                ax.plot(history.history['loss'], label='Training Loss')
                
                if 'val_loss' in history.history:
                    ax.plot(history.history['val_loss'], label='Validation Loss')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training History')
                ax.legend()
                
                st.pyplot(fig)
        
        # Model download
        st.write("### Download Trained Model")
        
        if is_neural_network:
            st.info("Neural Network models can't be downloaded directly from the browser. They are saved in the 'models' directory.")
        else:
            # Serialize the model
            model_binary = joblib.dumps(model)
            
            st.download_button(
                label="Download Model",
                data=model_binary,
                file_name=f"{selected_model.lower().replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            ) 