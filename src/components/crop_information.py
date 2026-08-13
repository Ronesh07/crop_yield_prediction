# importing libraries
# importing part 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
from PIL import Image
import json

from src.data import data_loader
from src.visualizations import plots

# Path for local crop images
CROP_IMAGES_PATH = "static/images/crops"

# Dictionary with crop information
CROP_INFO = {
    "rice": {
        "description": "Rice is the seed of the grass species Oryza sativa. It is the most widely consumed staple food for a large part of the world's human population, especially in Asia.",
        "growing_conditions": "Requires warm climate, heavy rainfall, humidity and waterlogged soil. Optimal temperature is between 20-35°C.",
        "nutritional_value": "Good source of carbohydrates, provides some protein, very little fat, and negligible amounts of vitamins and minerals.",
        "typical_yield": "4-6 tons per hectare",
        "major_producers": ["China", "India", "Indonesia", "Bangladesh", "Vietnam"],
        "image_url": "https://source.unsplash.com/800x600/?rice,field"
    },
    "wheat": {
        "description": "Wheat is a grass widely cultivated for its seed, a cereal grain which is a worldwide staple food.",
        "growing_conditions": "Can be grown in a wide range of soils, but thrives in well-drained loamy soil. Requires moderate rainfall and cool weather.",
        "nutritional_value": "Rich in carbohydrates, with moderate amounts of protein, minimal fat, and several vitamins and minerals.",
        "typical_yield": "3-4 tons per hectare",
        "major_producers": ["China", "India", "Russia", "United States", "France"],
        "image_url": "https://source.unsplash.com/800x600/?wheat,field"
    },
    "maize": {
        "description": "Maize, also known as corn, is a cereal grain first domesticated by indigenous peoples in southern Mexico.",
        "growing_conditions": "Requires warm weather, plenty of sunshine, and moderate rainfall. Optimal temperature is between 18-32°C.",
        "nutritional_value": "High in carbohydrates, provides some protein and minimal fat. Good source of dietary fiber and several vitamins.",
        "typical_yield": "5-8 tons per hectare",
        "major_producers": ["United States", "China", "Brazil", "Argentina", "Mexico"],
        "image_url": "https://source.unsplash.com/800x600/?corn,maize,field"
    },
    "potato": {
        "description": "The potato is a root vegetable native to the Americas, a starchy tuber of the plant Solanum tuberosum.",
        "growing_conditions": "Grows best in cool climates with well-drained, loose soil. Requires consistent moisture.",
        "nutritional_value": "Good source of carbohydrates, vitamin C, potassium, and dietary fiber.",
        "typical_yield": "15-30 tons per hectare",
        "major_producers": ["China", "India", "Russia", "Ukraine", "United States"],
        "image_url": "https://source.unsplash.com/800x600/?potato,field"
    },
    "cotton": {
        "description": "Cotton is a soft, fluffy staple fiber that grows in a boll around the seeds of the cotton plant.",
        "growing_conditions": "Requires a long frost-free period, plenty of sunshine, and moderate rainfall. Grows best in warm climates.",
        "nutritional_value": "Not a food crop. Used primarily for textile production.",
        "typical_yield": "1.5-2.5 tons per hectare",
        "major_producers": ["China", "India", "United States", "Pakistan", "Brazil"],
        "image_url": "https://source.unsplash.com/800x600/?cotton,field"
    },
    "sugarcane": {
        "description": "Sugarcane is a perennial grass of the genus Saccharum used for sugar production.",
        "growing_conditions": "Requires tropical or subtropical climate, abundant sunshine, and high humidity. Optimal temperature is between 20-35°C.",
        "nutritional_value": "Primarily processed for sugar. Raw sugarcane juice contains vitamins and minerals.",
        "typical_yield": "60-80 tons per hectare",
        "major_producers": ["Brazil", "India", "China", "Thailand", "Pakistan"],
        "image_url": "https://source.unsplash.com/800x600/?sugarcane,field"
    },
    "coffee": {
        "description": "Coffee is a brewed drink prepared from roasted coffee beans, the seeds of berries from certain Coffea species.",
        "growing_conditions": "Grows best in tropical climates with rich soil and moderate rainfall. Prefers shade and elevation.",
        "nutritional_value": "Contains caffeine, antioxidants, and minimal nutritional value as a food.",
        "typical_yield": "0.5-1 ton per hectare",
        "major_producers": ["Brazil", "Vietnam", "Colombia", "Indonesia", "Ethiopia"],
        "image_url": "https://source.unsplash.com/800x600/?coffee,plantation"
    },
    "muskmelon": {
        "description": "Muskmelon (Cucumis melo) is a species of melon that includes cantaloupes, honeydew melons, and Persian melons.",
        "growing_conditions": "Requires warm temperatures, full sun, and well-drained soil. Sensitive to frost and excessive moisture.",
        "nutritional_value": "Rich in vitamins A and C, potassium, and antioxidants. Low in calories.",
        "typical_yield": "15-25 tons per hectare",
        "major_producers": ["China", "Turkey", "Iran", "Egypt", "India"],
        "image_url": "https://source.unsplash.com/800x600/?muskmelon,cantaloupe"
    },
    "watermelon": {
        "description": "Watermelon (Citrullus lanatus) is a flowering plant species of the Cucurbitaceae family and the name of its edible fruit.",
        "growing_conditions": "Requires warm temperatures, full sun, and well-drained soil. Needs consistent water supply.",
        "nutritional_value": "High water content, good source of vitamins A and C, and contains lycopene. Low in calories.",
        "typical_yield": "20-40 tons per hectare",
        "major_producers": ["China", "Turkey", "Iran", "Brazil", "Egypt"],
        "image_url": "https://source.unsplash.com/800x600/?watermelon,field"
    },
    "apple": {
        "description": "The apple is a pome fruit from the apple tree (Malus domestica), one of the most widely cultivated tree fruits.",
        "growing_conditions": "Grows in temperate climates with cold winters and mild summers. Requires well-drained soil and full sun.",
        "nutritional_value": "Good source of dietary fiber, vitamin C, and various antioxidants.",
        "typical_yield": "15-30 tons per hectare",
        "major_producers": ["China", "United States", "Turkey", "Poland", "Italy"],
        "image_url": "https://source.unsplash.com/800x600/?apple,orchard"
    },
    "grapes": {
        "description": "Grapes are the fruit of the woody vine of the genus Vitis. They can be eaten fresh as table grapes or used for making wine, jam, juice, or dried as raisins.",
        "growing_conditions": "Thrives in Mediterranean-like climates with warm, dry summers and mild winters. Requires well-drained soil.",
        "nutritional_value": "Contains resveratrol, antioxidants, and vitamins K and C. Provides natural sugars.",
        "typical_yield": "8-15 tons per hectare",
        "major_producers": ["China", "Italy", "United States", "Spain", "France"],
        "image_url": "https://source.unsplash.com/800x600/?grapes,vineyard"
    },
    "mango": {
        "description": "Mango is a juicy stone fruit produced from numerous species of tropical trees belonging to the flowering plant genus Mangifera.",
        "growing_conditions": "Requires tropical climate, full sun, and well-drained soil. Optimal temperature is between 24-30°C.",
        "nutritional_value": "Rich in vitamins A and C, and various antioxidants. Good source of dietary fiber.",
        "typical_yield": "10-15 tons per hectare",
        "major_producers": ["India", "China", "Thailand", "Indonesia", "Mexico"],
        "image_url": "https://source.unsplash.com/800x600/?mango,tree"
    },
    "banana": {
        "description": "Banana is an edible fruit produced by several kinds of large herbaceous flowering plants in the genus Musa.",
        "growing_conditions": "Requires tropical climate, abundant rainfall, and high humidity. Grows best in rich, well-drained soil.",
        "nutritional_value": "Good source of potassium, vitamin B6, vitamin C, and dietary fiber.",
        "typical_yield": "20-40 tons per hectare",
        "major_producers": ["India", "China", "Indonesia", "Brazil", "Ecuador"],
        "image_url": "https://source.unsplash.com/800x600/?banana,plantation"
    },
    "orange": {
        "description": "The orange is the fruit of the citrus species Citrus × sinensis. It is a hybrid between pomelo and mandarin.",
        "growing_conditions": "Grows best in subtropical climates with warm temperatures and moderate rainfall. Requires well-drained soil.",
        "nutritional_value": "Excellent source of vitamin C, also provides folate, thiamine, and potassium.",
        "typical_yield": "15-30 tons per hectare",
        "major_producers": ["Brazil", "China", "India", "United States", "Mexico"],
        "image_url": "https://source.unsplash.com/800x600/?orange,orchard"
    }
}

# Sample global yield data
def get_global_crop_data():
    # This would normally be loaded from a real dataset
    # Creating sample data for demonstration
    countries = [
        "United States", "China", "India", "Brazil", "Russia", 
        "Argentina", "Ukraine", "France", "Indonesia", "Bangladesh", 
        "Vietnam", "Thailand", "Pakistan", "Australia", "Mexico", 
        "Nigeria", "Ethiopia", "Egypt", "Turkey", "Iran"
    ]
    
    crops = ["Rice", "Wheat", "Maize", "Potato", "Cotton", "Sugarcane"]
    
    data = []
    
    for country in countries:
        for crop in crops:
            # Generate random yield data
            base_yield = {
                "Rice": 4.5,
                "Wheat": 3.2,
                "Maize": 5.5,
                "Potato": 20.0,
                "Cotton": 2.1,
                "Sugarcane": 70.0
            }[crop]
            
            # Add some variation
            variation = np.random.normal(0, base_yield * 0.3) 
            actual_yield = max(0.5, base_yield + variation)
            
            # Generate data for multiple years
            for year in range(2015, 2023):
                year_variation = np.random.normal(0, base_yield * 0.1)
                year_yield = actual_yield + year_variation + (year - 2015) * (base_yield * 0.02)  # Slight upward trend
                
                data.append({
                    "Country": country,
                    "Crop": crop,
                    "Year": year,
                    "Yield (tons/ha)": max(0.5, year_yield),
                    "Production (million tons)": max(0.1, year_yield * np.random.uniform(0.5, 5))
                })
    
    return pd.DataFrame(data)

def render_crop_information():
    """Render the crop information page"""
    st.title("Crop Information Center")
    st.subheader("Learn about different crops and global yield data")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Crop Encyclopedia", "Global Yield Data", "Crop Predictor", "Add New Crop Details"])
    
    with tab1:
        render_crop_encyclopedia()
    
    with tab2:
        render_global_yield_data()
    
    with tab3:
        render_crop_predictor()
    
    with tab4:
        render_add_crop_details()

def render_crop_encyclopedia():
    """Render the crop encyclopedia section"""
    st.write("### Crop Encyclopedia")
    st.write("Explore detailed information about various crops.")

    # --- Load crops from crop_info.json dynamically ---
    crop_info_path = os.path.join("data", "crop_info.json")
    if os.path.exists(crop_info_path):
        with open(crop_info_path, "r") as f:
            dynamic_crop_info = json.load(f)
    else:
        dynamic_crop_info = {}

    crop_options = list(dynamic_crop_info.keys())
    if not crop_options:
        st.info("No crops available. Please add a crop.")
        return
    selected_crop = st.selectbox("Select a crop", crop_options)

    # Display crop information
    if selected_crop in dynamic_crop_info:
        crop_data = dynamic_crop_info[selected_crop]
        col1, col2 = st.columns([1, 1])
        with col1:
            # Check if local image exists first
            extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_found = False
            for ext in extensions:
                test_path = os.path.join(CROP_IMAGES_PATH, f"{selected_crop}{ext}")
                if os.path.exists(test_path):
                    local_image_path = test_path
                    image_found = True
                    break
            if image_found:
                try:
                    image = Image.open(local_image_path)
                    st.image(image, caption=selected_crop.capitalize(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.info("No image available for this crop.")
            # Add option to upload a new image for existing crop
            st.write("#### Upload a custom image")
            uploaded_file = st.file_uploader(f"Upload a new image for {selected_crop}", type=["jpg", "jpeg", "png", "webp"], key=f"upload_{selected_crop}")
            if uploaded_file is not None:
                try:
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    os.makedirs(CROP_IMAGES_PATH, exist_ok=True)
                    save_path = os.path.join(CROP_IMAGES_PATH, f"{selected_crop}{file_ext}")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Image saved successfully at {save_path}")
                    image = Image.open(save_path)
                    st.image(image, caption=f"New image for {selected_crop.capitalize()}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error saving image: {e}")
        with col2:
            st.subheader(selected_crop.capitalize())
            st.write(crop_data.get("description", "No description available."))
            st.write("**Growing Conditions:**")
            st.write(crop_data.get("growing_conditions", "N/A"))
            st.write("**Nutritional Value:**")
            st.write(crop_data.get("nutritional_value", "N/A"))
            st.write("**Typical Yield:**")
            st.write(crop_data.get("typical_yield", "N/A"))
            st.write("**Major Producers:**")
            for i, country in enumerate(crop_data.get("major_producers", []), 1):
                st.write(f"{i}. {country}")
        st.subheader(f"Best Practices for {selected_crop.capitalize()} Cultivation")
        practices = {
            "Soil Preparation": "Prepare the soil by plowing and leveling. Ensure proper drainage.",
            "Sowing": f"Sow {selected_crop} seeds at the recommended spacing and depth based on variety.",
            "Irrigation": "Maintain optimal soil moisture throughout the growing season.",
            "Fertilization": "Apply balanced fertilizers based on soil test results.",
            "Pest & Disease Management": "Monitor regularly for pests and diseases. Use integrated pest management.",
            "Harvesting": "Harvest at proper maturity for maximum yield and quality."
        }
        for practice, description in practices.items():
            st.write(f"**{practice}:** {description}")

def render_global_yield_data():
    """Render the global yield data section"""
    st.write("### Global Crop Yield Data")
    st.write("Explore crop yields around the world.")
    
    # Load global crop data
    df = get_global_crop_data()
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        selected_crops = st.multiselect(
            "Select crops", 
            df["Crop"].unique(),
            default=["Rice", "Wheat", "Maize"]
        )
    
    with col2:
        selected_year = st.slider(
            "Select year",
            int(df["Year"].min()),
            int(df["Year"].max()),
            int(df["Year"].max())
        )
    
    if not selected_crops:
        st.warning("Please select at least one crop.")
        return
    
    # Filter data
    filtered_df = df[(df["Crop"].isin(selected_crops)) & (df["Year"] == selected_year)]
    
    # Display global map
    st.write(f"#### Crop Yields in {selected_year}")
    
    # Create a choropleth map
    for crop in selected_crops:
        crop_df = filtered_df[filtered_df["Crop"] == crop]
        
        fig = px.choropleth(
            crop_df,
            locations="Country",
            locationmode="country names",
            color="Yield (tons/ha)",
            hover_name="Country",
            hover_data=["Production (million tons)"],
            title=f"{crop} Yield (tons/ha) in {selected_year}",
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=400
        )
        
        st.plotly_chart(fig)
    
    # Display yield trends over time
    st.write("#### Yield Trends Over Time")
    
    # Select countries
    selected_countries = st.multiselect(
        "Select countries",
        df["Country"].unique(),
        default=["United States", "China", "India", "Brazil"]
    )
    
    if not selected_countries:
        st.warning("Please select at least one country.")
        return
    
    # Filter data for trends
    trend_df = df[
        (df["Crop"].isin(selected_crops)) & 
        (df["Country"].isin(selected_countries))
    ]
    
    # Create line chart
    fig = px.line(
        trend_df,
        x="Year",
        y="Yield (tons/ha)",
        color="Country",
        facet_col="Crop",
        facet_col_wrap=2,
        title="Crop Yield Trends by Country",
        labels={"Yield (tons/ha)": "Yield (tons/ha)"}
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig)
    
    # Yield comparison bar chart
    st.write("#### Yield Comparison Between Countries")
    
    # Calculate average yield for selected year
    comparison_df = filtered_df[filtered_df["Country"].isin(selected_countries)]
    
    if len(comparison_df) > 0:
        fig = px.bar(
            comparison_df,
            x="Country",
            y="Yield (tons/ha)",
            color="Crop",
            barmode="group",
            title=f"Crop Yields by Country in {selected_year}",
            labels={"Yield (tons/ha)": "Yield (tons/ha)"}
        )
        
        fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Yield (tons/ha)",
            legend_title="Crop"
        )
        
        st.plotly_chart(fig)
    else:
        st.info("No data available for the selected criteria.")

def render_crop_predictor():
    """Render the crop predictor section"""
    st.write("### Crop Recommendation System")
    st.write("Enter soil and environmental parameters to get crop recommendations.")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        nitrogen = st.slider("Nitrogen Content (kg/ha)", 0, 140, 50)
        phosphorus = st.slider("Phosphorus Content (kg/ha)", 5, 145, 50)
        potassium = st.slider("Potassium Content (kg/ha)", 5, 205, 50)
    
    with col2:
        temperature = st.slider("Temperature (°C)", 8.0, 45.0, 25.0)
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 71.0)
        ph = st.slider("pH Value", 3.5, 10.0, 6.5)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)
    
    # Predict button
    if st.button("Recommend Suitable Crops"):
        # This would normally use a trained model
        # Creating a simple rule-based system for demonstration
        
        # Load some sample data to show a realistic output
        try:
            df = data_loader.load_crop_recommendation_data()
            if df is not None and 'label' in df.columns:
                df['label'] = df['label'].astype('category')
                
                # Simple prediction logic (would normally use a trained model)
                # For demo, find closest matches in the dataset using Euclidean distance
                features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
                input_values = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall])
                
                # Standardize the input
                scaler = StandardScaler()
                df_features = df[features].copy()
                scaler.fit(df_features)
                
                # Standardize input values
                input_scaled = scaler.transform(input_values.reshape(1, -1))
                
                # Standardize dataset
                df_scaled = scaler.transform(df_features)
                
                # Calculate distances
                distances = np.sqrt(((df_scaled - input_scaled) ** 2).sum(axis=1))
                
                # Get top 3 closest matches
                top_3_indices = distances.argsort()[:3]
                
                # Get the predicted crops
                predicted_crops = df.iloc[top_3_indices]['label'].values
                
                # Calculate confidence scores based on distance (inverse relationship)
                confidence_scores = 1 / (1 + distances[top_3_indices])
                
                # Normalize confidence scores to sum to 1
                confidence_scores = confidence_scores / confidence_scores.sum()
                
                # Display results
                st.success("Here are your crop recommendations based on the soil and climate parameters:")
                
                results = []
                for i, (crop, score) in enumerate(zip(predicted_crops, confidence_scores)):
                    crop_name = str(crop)
                    results.append({
                        "Crop": crop_name,
                        "Suitability Score": float(score) * 100
                    })
                
                # Create a DataFrame
                results_df = pd.DataFrame(results)
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create a bar chart
                    fig = px.bar(
                        results_df,
                        x="Crop",
                        y="Suitability Score",
                        color="Suitability Score",
                        labels={"Suitability Score": "Suitability Score (%)"},
                        title="Crop Recommendations",
                        color_continuous_scale="Viridis"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Crop",
                        yaxis_title="Suitability Score (%)",
                        yaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig)
                
                with col2:
                    st.write("### Top Recommendations")
                    for idx, row in enumerate(results_df.itertuples(index=False), 1):
                        st.write(f"**{idx}. {getattr(row, 'Crop').capitalize()}** ({getattr(row, 'Suitability Score'):.1f}%)")
                
                # Display detailed information for the top recommended crop
                st.write("### Details about the top recommended crop:")
                top_crop = results_df.iloc[0]["Crop"].lower()
                
                if top_crop in CROP_INFO:
                    st.write(f"#### {top_crop.capitalize()}")
                    st.write(CROP_INFO[top_crop]["description"])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(CROP_INFO[top_crop]["image_url"], caption=top_crop.capitalize(), use_container_width=True)
                    
                    with col2:
                        st.write("**Growing Conditions:**")
                        st.write(CROP_INFO[top_crop]["growing_conditions"])
                        
                        st.write("**Typical Yield:**")
                        st.write(CROP_INFO[top_crop]["typical_yield"])
                        
                        st.write("**Major Producers:**")
                        st.write(", ".join(CROP_INFO[top_crop]["major_producers"]))
            else:
                st.error("Could not load crop recommendation data. Using sample recommendations instead.")
                sample_recommendations()
        except Exception as e:
            st.error(f"Error predicting crops: {e}")
            sample_recommendations()

def render_add_crop_details():
    """Render the add new crop details form as a separate tab."""
    st.write("### Add a New Crop")
    with st.form("add_crop_form"):
        new_crop_name = st.text_input("Crop Name (lowercase, no spaces)")
        new_crop_desc = st.text_area("Description")
        new_crop_growing = st.text_area("Growing Conditions")
        new_crop_nutrition = st.text_area("Nutritional Value")
        new_crop_yield = st.text_input("Typical Yield")
        new_crop_producers = st.text_area("Major Producers (comma separated)")
        new_crop_image = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png", "webp"])
        submit_crop = st.form_submit_button("Add Crop")

        if submit_crop:
            if not (new_crop_name and new_crop_desc and new_crop_growing and new_crop_nutrition and new_crop_yield and new_crop_producers and new_crop_image):
                st.error("Please fill all fields and upload an image.")
            else:
                # Save image
                ext = os.path.splitext(new_crop_image.name)[1].lower()
                image_save_path = os.path.join(CROP_IMAGES_PATH, f"{new_crop_name}{ext}")
                os.makedirs(CROP_IMAGES_PATH, exist_ok=True)
                with open(image_save_path, "wb") as f:
                    f.write(new_crop_image.getbuffer())

                # Update crop_info.json
                crop_info_path = os.path.join("data", "crop_info.json")
                try:
                    if os.path.exists(crop_info_path):
                        with open(crop_info_path, "r") as f:
                            crop_info = json.load(f)
                    else:
                        crop_info = {}
                    crop_info[new_crop_name] = {
                        "description": new_crop_desc,
                        "growing_conditions": new_crop_growing,
                        "nutritional_value": new_crop_nutrition,
                        "typical_yield": new_crop_yield,
                        "major_producers": [x.strip() for x in new_crop_producers.split(",") if x.strip()]
                    }
                    with open(crop_info_path, "w") as f:
                        json.dump(crop_info, f, indent=4)
                    st.success(f"Crop '{new_crop_name}' added successfully!")
                except Exception as e:
                    st.error(f"Error saving crop info: {e}")

def sample_recommendations():
    """Display sample crop recommendations if real prediction isn't available"""
    results = [
        {"Crop": "rice", "Suitability Score": 82.5},
        {"Crop": "maize", "Suitability Score": 64.3},
        {"Crop": "cotton", "Suitability Score": 53.1}
    ]
    
    results_df = pd.DataFrame(results)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a bar chart
        fig = px.bar(
            results_df,
            x="Crop",
            y="Suitability Score",
            color="Suitability Score",
            labels={"Suitability Score": "Suitability Score (%)"},
            title="Sample Crop Recommendations",
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            xaxis_title="Crop",
            yaxis_title="Suitability Score (%)",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig)
    
    with col2:
        st.write("### Top Recommendations")
        for idx, row in enumerate(results_df.itertuples(index=False), 1):
            st.write(f"**{idx}. {getattr(row, 'Crop').capitalize()}** ({getattr(row, 'Suitability Score'):.1f}%)")
    
    # Display detailed information for the top recommended crop
    st.write("### Details about the top recommended crop:")
    top_crop = results_df.iloc[0]["Crop"].lower()
    
    if top_crop in CROP_INFO:
        st.write(f"#### {top_crop.capitalize()}")
        st.write(CROP_INFO[top_crop]["description"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(CROP_INFO[top_crop]["image_url"], caption=top_crop.capitalize(), use_container_width=True)
        
        with col2:
            st.write("**Growing Conditions:**")
            st.write(CROP_INFO[top_crop]["growing_conditions"])
            
            st.write("**Typical Yield:**")
            st.write(CROP_INFO[top_crop]["typical_yield"])
            
            st.write("**Major Producers:**")
            st.write(", ".join(CROP_INFO[top_crop]["major_producers"])) 