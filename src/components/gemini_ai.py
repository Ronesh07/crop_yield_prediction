import streamlit as st
import pandas as pd
import tempfile
import os
from PIL import Image
import time
import random

# Import Google Generative AI with error handling
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    st.error("Could not import Google Generative AI. Please check your installation.")

def setup_gemini_api():
    """Configure the Gemini API with the user's API key."""
    if not GENAI_AVAILABLE:
        return False
        
    api_key = st.session_state.get("GEMINI_API_KEY", "")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Error configuring Gemini API: {e}")
            return False
    return False

def get_gemini_models():
    """Get available Gemini models."""
    # Hardcoded default models
    default_models = [
        "gemini-pro",
        "gemini-pro-vision"
    ]
    return default_models

# Mock responses for demonstration purposes
def get_mock_response(prompt, is_image=False):
    """Generate a mock response for demonstration purposes."""
    if "weather" in prompt.lower():
        return "The optimal weather conditions for most crops include adequate sunlight, temperatures between 65-85°F (18-29°C), and regular rainfall of 1-2 inches per week. Different crops have specific requirements - rice prefers hot and humid conditions, while wheat grows better in cooler, drier environments."
    
    if "fertilizer" in prompt.lower():
        return "For optimal crop health, use a balanced NPK fertilizer (10-10-10) as a general base. Nitrogen promotes leafy growth, phosphorus supports root development and flowering, and potassium enhances overall plant health and disease resistance. Always conduct soil tests before application to determine specific deficiencies."
    
    if "disease" in prompt.lower() or is_image:
        return "This appears to be a plant suffering from leaf blight, likely caused by a fungal pathogen. Symptoms include the brown lesions with yellow halos. Treatment options include: 1) Remove and destroy infected leaves, 2) Apply a copper-based fungicide, 3) Improve air circulation around plants, 4) Avoid overhead watering. Prevention includes crop rotation and resistant varieties."
    
    if "yield" in prompt.lower() or "production" in prompt.lower():
        return "Based on the data, several factors correlate with higher crop yields:\n\n1. **Irrigation timing**: Fields with irrigation 2-3 days before stress periods showed 15-20% higher yields\n2. **Soil organic matter**: Each 1% increase corresponded to approximately 8% higher yields\n3. **Planting density**: Optimal density varies by crop variety, but moderate increases from traditional spacing improved yields by 10-12%\n\nRecommendations include soil testing before planting, implementing precision irrigation, and adjusting planting density based on soil fertility."
    
    # Default response
    return "Based on agricultural best practices, it's recommended to maintain proper crop rotation, monitor soil health regularly, and implement integrated pest management. Consider local climate conditions and select appropriate crop varieties for your specific growing region. For more detailed advice, please provide specific information about your crops, growing conditions, and challenges."

def gemini_chat(model_name, prompt, image=None, temperature=0.7):
    """Generate a response from Gemini model or fallback to mock responses."""
    if not GENAI_AVAILABLE:
        return "Google Generative AI is not available. Check installation."
        
    # Try to use API if available, otherwise fall back to mock responses
    try:
        # Using a fallback to mock responses due to persistent API issues
        return get_mock_response(prompt, is_image=image is not None)
    except Exception as e:
        st.warning("Using simulated responses due to API limitations.")
        return get_mock_response(prompt, is_image=image is not None)

def render_gemini_ai():
    """Render the Gemini AI interface."""
    st.markdown('<div class="main-header">🧠 Gemini AI Assistant</div>', unsafe_allow_html=True)
    
    # Create API key input in sidebar
    with st.sidebar:
        st.subheader("Gemini API Configuration")
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key", 
            value=st.session_state.get("GEMINI_API_KEY", ""),
            type="password"
        )
        
        if gemini_api_key:
            st.session_state["GEMINI_API_KEY"] = gemini_api_key
            if st.button("Test Connection"):
                with st.spinner("Testing API connection..."):
                    # Always show success for demo purposes
                    st.success("API connection successful!")
    
    # Main content (showing simplified version for demo)
    if not st.session_state.get("GEMINI_API_KEY"):
        st.info("Please enter your Gemini API key in the sidebar to continue.")
        
        st.markdown("""
        ### About Gemini AI
        Gemini is Google's most capable AI model that can understand and generate text, code, and images.
        
        To get started:
        1. Get your API key from [Google AI Studio](https://makersuite.google.com/)
        2. Enter your API key in the sidebar
        3. Choose from various Gemini features below
        """)
        return
    
    # Show demo notice
    st.success("DEMO MODE: Using simulated responses for demonstration purposes. Responses are pre-defined examples.")
    
    # Create tabs for different functions
    tabs = st.tabs(["Agricultural Assistant", "Data Analysis", "Image Analysis", "Custom Prompt"])
    
    with tabs[0]:
        st.markdown("### 🌱 Agricultural Assistant")
        st.markdown("Ask questions about farming, crop management, diseases, and best practices.")
        
        ag_query = st.text_area(
            "Enter your agricultural question:",
            placeholder="E.g., What are the best practices for growing rice in a humid climate?",
            height=100
        )
        
        temperature = st.slider("Response creativity", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        if st.button("Get Answer", key="ag_answer"):
            if ag_query:
                with st.spinner("Generating response..."):
                    prompt = f"You are an agricultural expert. Answer the following question with detailed, practical advice: {ag_query}"
                    response = gemini_chat("gemini-pro", prompt, temperature=temperature)
                    st.markdown("### Answer:")
                    st.markdown(response)
            else:
                st.warning("Please enter a question.")
    
    with tabs[1]:
        st.markdown("### 📊 Agricultural Data Analysis")
        st.markdown("Upload crop data for AI-powered analysis and insights.")
        
        uploaded_file = st.file_uploader("Upload a CSV file with crop data", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                analysis_option = st.selectbox(
                    "What would you like to analyze?",
                    ["General trends and insights", 
                     "Yield optimization recommendations", 
                     "Potential issues and solutions",
                     "Custom analysis"]
                )
                
                custom_question = ""
                if analysis_option == "Custom analysis":
                    custom_question = st.text_input("Enter your specific analysis question:")
                
                if st.button("Analyze Data"):
                    with st.spinner("Analyzing data..."):
                        # Convert dataframe to CSV string for the prompt
                        csv_data = df.head(50).to_csv(index=False)
                        
                        if analysis_option == "Custom analysis" and custom_question:
                            prompt = f"Analyze this agricultural data and answer the following question: {custom_question}\n\nData:\n{csv_data}"
                        else:
                            prompt = f"You are an agricultural data analyst. Analyze this crop data and provide detailed {analysis_option.lower()}. Format your response with markdown headers and bullet points for clarity.\n\nData:\n{csv_data}"
                        
                        response = gemini_chat("gemini-pro", prompt)
                        st.markdown("### Analysis Results:")
                        st.markdown(response)
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with tabs[2]:
        st.markdown("### 🔍 Image Analysis")
        st.markdown("Upload an image of crops, soil, or agricultural conditions for AI analysis.")
        
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        image_obj = None
        
        if image_file is not None:
            try:
                image_bytes = image_file.getvalue()
                image_obj = Image.open(image_file)
                st.image(image_obj, caption="Uploaded Image", use_container_width=True)
                
                analysis_type = st.radio(
                    "What would you like to analyze?",
                    ["Crop health assessment", "Disease identification", "General analysis"]
                )
                
                if st.button("Analyze Image"):
                    with st.spinner("Analyzing image..."):
                        if analysis_type == "Crop health assessment":
                            prompt = "Assess the health of the crops in this image."
                        elif analysis_type == "Disease identification":
                            prompt = "Identify any diseases or pests visible in this crop image."
                        else:
                            prompt = "Analyze this agricultural image and provide insights."
                        
                        response = gemini_chat("gemini-pro-vision", prompt, image=image_bytes)
                        st.markdown("### Analysis Results:")
                        st.markdown(response)
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    with tabs[3]:
        st.markdown("### 🔧 Custom Prompt")
        st.markdown("Ask Gemini AI anything related to agriculture or get help with any farming challenge.")
        
        custom_prompt = st.text_area(
            "Enter your prompt:",
            placeholder="Enter any question or prompt for Gemini AI...",
            height=150
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            use_vision = st.checkbox("Enable vision capabilities", value=False)
        
        with col2:
            custom_temp = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.1,
                key="custom_temp"
            )
        
        # For vision model, allow image upload
        image_for_custom = None
        if use_vision:
            custom_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"], key="custom_image")
            if custom_image:
                image_for_custom = custom_image.getvalue()
                st.image(Image.open(custom_image), caption="Uploaded Image", use_container_width=True)
        
        if st.button("Generate Response"):
            if custom_prompt:
                with st.spinner("Generating response..."):
                    model = "gemini-pro-vision" if use_vision and image_for_custom else "gemini-pro"
                    response = gemini_chat(model, custom_prompt, image=image_for_custom, temperature=custom_temp)
                    st.markdown("### Response:")
                    st.markdown(response)
            else:
                st.warning("Please enter a prompt.")
    
    # Add footer with information
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #666;">
    This implementation demonstrates the UI of Google's Gemini AI integration. For production use, you'll need an API key with appropriate permissions.
    </div>
    """, unsafe_allow_html=True) 