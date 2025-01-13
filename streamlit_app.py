import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import pdfplumber

# Page config
st.set_page_config(
    page_title="Femicide News Classifier",
    page_icon="📰",
    layout="wide"
)

# Title and description
st.title("Femicide Swahili News Classification")
st.write("Upload swahili news articles to predict whether they are related to femicide.")

# Model loading with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_logistic_regression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')
        return model, scaler, pca
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Load models
model, scaler, pca = load_models()

# File uploader
uploaded_files = st.file_uploader(
    "Upload news articles (TXT, CSV, or PDF files)", 
    type=['txt', 'csv', 'pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Number of files uploaded: {len(uploaded_files)}")
    
    # Process each file
    for file in uploaded_files:
        try:
            # Create a container for each file
            with st.expander(f"File: {file.name}", expanded=True):
                # Read file content based on type
                if file.name.endswith('.txt'):
                    content = file.read().decode('utf-8')
                    st.write("Content preview:")
                    st.text(content[:500] + "..." if len(content) > 500 else content)
                
                elif file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                    st.write("CSV content preview:")
                    st.dataframe(df.head())
                
                elif file.name.endswith('.pdf'):
                    st.write("PDF content preview:")
                    text = extract_text_from_pdf(file)
                    st.text(text[:500] + "..." if len(text) > 500 else text)
                
                if st.button(f"Analyze {file.name}", key=file.name):
                    with st.spinner("Processing..."):
                        # Add your prediction pipeline here
                        # This is where you'll need to:
                        # 1. Extract text from the file
                        # 2. Preprocess the text
                        # 3. Make predictions
                        st.success("Analysis complete!")
                        
                        # Placeholder for prediction results
                        st.write("Prediction results would go here")
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

# Add instructions
with st.expander("How to use this classifier"):
    st.write("""
    1. Use the file uploader to upload one or more news articles
    2. Supported formats:
        - TXT: Plain text files
        - CSV: Files containing news articles in tabular format
        - PDF: PDF documents containing news articles
    3. Click 'Analyze' for each file to get predictions
    """)

# Debug information
with st.expander("Debug Information"):
    st.write("Current directory:", os.getcwd())
    st.write("Files in directory:", os.listdir())