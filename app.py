import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
from pathlib import Path
import tempfile
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Configure page
st.set_page_config(
    page_title="Disease Trajectory Analysis",
    layout="wide"
)

# Disease system categories
condition_categories = {
    "Addisons Disease": "Endocrine",
    "Anaemia": "Blood",
    "Barretts Oesophagus": "Digestive",
    "Bronchiectasis": "Respiratory",
    "Cancer": "Neoplasms",
    "Cardiac Arrhythmias": "Cardiovascular",
    "Cerebral Palsy": "Nervous",
    "Chronic Airway Diseases": "Respiratory",
    "Chronic Arthritis": "Musculoskeletal",
    "Chronic Constipation": "Digestive",
    "Chronic Diarrhoea": "Digestive",
    "Chronic Kidney Disease": "Genitourinary",
    "Chronic Pain Conditions": "Musculoskeletal",
    "Chronic Pneumonia": "Respiratory",
    "Cirrhosis": "Digestive",
    "Coronary Heart Disease": "Cardiovascular",
    "Dementia": "Mental health",
    "Diabetes": "Endocrine",
    "Dysphagia": "Digestive",
    "Epilepsy": "Nervous",
    "Heart Failure": "Cardiovascular",
    "Hearing Loss": "Ear",
    "Hypertension": "Cardiovascular",
    "Inflammatory Bowel Disease": "Digestive",
    "Insomnia": "Nervous",
    "Interstitial Lung Disease": "Respiratory",
    "Mental Illness": "Mental",
    "Menopausal and Perimenopausal": "Genitourinary",
    "Multiple Sclerosis": "Nervous",
    "Neuropathic Pain": "Nervous",
    "Osteoporosis": "Musculoskeletal",
    "Parkinsons": "Nervous",
    "Peripheral Vascular Disease": "Circulatory",
    "Polycystic Ovary Syndrome": "Endocrine",
    "Psoriasis": "Skin",
    "Reflux Disorders": "Digestive",
    "Stroke": "Nervous",
    "Thyroid Disorders": "Endocrine",
    "Tourette": "Mental health",
    "Visual Impairment": "Eye"
}

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

@st.cache_data
def perform_sensitivity_analysis(data):
    """Perform sensitivity analysis with corrected calculations"""
    or_thresholds = [2.0, 3.0, 4.0, 5.0]
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    # Get top patterns
    top_patterns = data.nlargest(5, 'OddsRatio')[
        ['ConditionA', 'ConditionB', 'OddsRatio', 'PairFrequency',
         'MedianDurationYearsWithIQR', 'DirectionalPercentage']
    ].to_dict('records')

    for threshold in or_thresholds:
        filtered_data = data[data['OddsRatio'] >= threshold].copy()
        n_trajectories = len(filtered_data)

        total_pairs = filtered_data['PairFrequency'].sum()
        estimated_unique_patients = total_pairs / 2
        coverage = min((estimated_unique_patients / total_patients) * 100, 100.0)

        system_pairs = set()
        for _, row in filtered_data.iterrows():
            sys_a = condition_categories.get(row['ConditionA'], 'Other')
            sys_b = condition_categories.get(row['ConditionB'], 'Other')
            if sys_a != sys_b:
                system_pairs.add(tuple(sorted([sys_a, sys_b])))

        duration_stats = filtered_data['MedianDurationYearsWithIQR'].apply(parse_iqr)
        medians = [x[0] for x in duration_stats if x[0] > 0]
        q1s = [x[1] for x in duration_stats if x[1] > 0]
        q3s = [x[2] for x in duration_stats if x[2] > 0]

        results.append({
            'OR_Threshold': threshold,
            'Num_Trajectories': n_trajectories,
            'Coverage_Percent': round(coverage, 2),
            'System_Pairs': len(system_pairs),
            'Median_Duration': round(np.median(medians) if medians else 0, 2),
            'Q1_Duration': round(np.median(q1s) if q1s else 0, 2),
            'Q3_Duration': round(np.median(q3s) if q3s else 0, 2),
            'Top_Patterns': top_patterns
        })

    return pd.DataFrame(results)

def create_sensitivity_plot(results):
    """Create sensitivity analysis visualization"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x_vals = results['OR_Threshold'].values
    bar_heights = results['Num_Trajectories']

    # Plot bars and lines
    bars = ax1.bar(x_vals, bar_heights, alpha=0.3, color='navy')
    line = ax2.plot(x_vals, results['Coverage_Percent'], 'r-o', linewidth=2)

    # Add scatter plot
    sizes = (results['System_Pairs'] / results['System_Pairs'].max()) * 500
    scatter = ax2.scatter(x_vals, results['Coverage_Percent'], s=sizes, alpha=0.5, color='darkred')

    # Add annotations
    for i, row in results.iterrows():
        ax1.text(row['OR_Threshold'], bar_heights[i] * 0.5,
                f"Median: {row['Median_Duration']:.1f}y\nIQR: [{row['Q1_Duration']:.1f}-{row['Q3_Duration']:.1f}]",
                ha='center', va='center', fontsize=10)

    # Labels and legend
    ax1.set_xlabel('Minimum Odds Ratio Threshold')
    ax1.set_ylabel('Number of Disease Trajectories')
    ax2.set_ylabel('Population Coverage (%)')

    legend_elements = [
        patches.Patch(facecolor='navy', alpha=0.3, label='Number of Trajectories'),
        Line2D([0], [0], color='r', marker='o', label='Population Coverage %'),
        Line2D([0], [0], marker='o', color='darkred', alpha=0.5, 
               label='System Pairs', markersize=10, linestyle='None')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.title('Impact of Odds Ratio Threshold on Disease Trajectory Analysis')
    plt.tight_layout()
    return fig

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
            
            # Create tabs
            tabs = st.tabs(["Sensitivity Analysis", "Data Preview"])
            
            # Sensitivity Analysis Tab
            with tabs[0]:
                st.header("Sensitivity Analysis")
                st.markdown("""
                Explore how different odds ratio thresholds affect the number of disease
                trajectories and population coverage.
                """)
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    st.markdown("### Control Panel")
                    analyze_button = st.button(
                        "Run Analysis",
                        key="run_sensitivity",
                        help="Click to perform sensitivity analysis"
                    )

                with col1:
                    if analyze_button:
                        with st.spinner("Analyzing data..."):
                            results = perform_sensitivity_analysis(data)
                            
                            st.subheader("Analysis Results")
                            display_df = results.drop('Top_Patterns', axis=1)
                            st.dataframe(
                                display_df.style.background_gradient(cmap='YlOrRd', subset=['Coverage_Percent'])
                            )

                            st.subheader("Top 5 Strongest Trajectories")
                            patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                            st.dataframe(
                                patterns_df.style.background_gradient(cmap='YlOrRd', subset=['OddsRatio'])
                            )

                            fig = create_sensitivity_plot(results)
                            st.pyplot(fig)

                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="sensitivity_analysis_results.csv",
                                mime="text/csv"
                            )
            
            # Data Preview Tab
            with tabs[1]:
                st.subheader("Data Preview")
                st.dataframe(data.head())

if __name__ == "__main__":
    main()
