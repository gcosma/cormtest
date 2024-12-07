# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
from pathlib import Path
import tempfile

# Configure page
st.set_page_config(
    page_title="Disease Trajectory Analysis",
    layout="wide"
)

# Define helper function
def parse_iqr(iqr_string):
    """Parse IQR string of format 'median [Q1-Q3]' into (median, q1, q3)"""
    try:
        median_str, iqr = iqr_string.split(' [')
        q1, q3 = iqr.strip(']').split('-')
        return float(median_str), float(q1), float(q3)
    except:
        return 0.0, 0.0, 0.0

def load_data(uploaded_file):
    """Load and process uploaded data"""
    try:
        data = pd.read_csv(uploaded_file)
        total_patients = data['TotalPatientsInGroup'].iloc[0]
        
        filename = uploaded_file.name.lower()
        gender = 'Unknown'
        age_group = 'Unknown'
        
        if 'females' in filename:
            gender = 'Female'
        elif 'males' in filename:
            gender = 'Male'
            
        if 'below45' in filename:
            age_group = '<45'
        elif '45to64' in filename:
            age_group = '45-64'
        elif '65plus' in filename:
            age_group = '65+'
            
        return data, total_patients, gender, age_group
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None, None

def main():
    st.title("Disease Trajectory Analysis")
    st.write("Upload your data to begin analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data, total_patients, gender, age_group = load_data(uploaded_file)
        
        if data is not None:
            st.success(f"Data loaded successfully! Total patients: {total_patients}")
            st.write(f"Gender: {gender}")
            st.write(f"Age Group: {age_group}")
            
            # Show data sample
            st.subheader("Data Preview")
            st.dataframe(data.head())

if __name__ == "__main__":
    main()
