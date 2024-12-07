import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from pyvis.network import Network
import math
import random
import tempfile
import base64
from pathlib import Path
from matplotlib import patches
from itertools import combinations
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Configure page settings
st.set_page_config(
    page_title="Clinical Trajectory Analysis Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling for clinical interface
st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        height: 3.5rem;
        background-color: #2e6d9e;
        color: white;
        font-size: 1.1rem;
        font-weight: 500;
        border: none;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #1e4e79;
    }
    .data-metrics {
        padding: 1rem;
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    .metric-value {
        color: #2e6d9e;
        font-size: 1.5rem;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #1e4e79;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2e6d9e;
        color: white;
    }
    .alert {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .alert-info {
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        color: #1976d2;
    }
    .alert-warning {
        background-color: #fff3e0;
        border: 1px solid #ffe0b2;
        color: #f57c00;
    }
    .dataframe {
        font-size: 0.9rem;
    }
    .dataframe th {
        background-color: #f8f9fa;
        font-weight: 600;
    }
    .risk-badge {
        padding: 0.3rem 0.6rem;
        border-radius: 3px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .high-risk {
        background-color: #ef5350;
        color: white;
    }
    .moderate-risk {
        background-color: #ff9800;
        color: white;
    }
    .low-risk {
        background-color: #4caf50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Disease Systems Categories with clinical classifications
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

# Clinical color scheme for systems
SYSTEM_COLORS = {
    "Endocrine": "#7E57C2",      # Deep Purple
    "Blood": "#EF5350",          # Red
    "Digestive": "#66BB6A",      # Green
    "Respiratory": "#42A5F5",    # Blue
    "Neoplasms": "#EC407A",      # Pink
    "Cardiovascular": "#FF7043", # Deep Orange
    "Nervous": "#FFCA28",        # Amber
    "Musculoskeletal": "#26A69A", # Teal
    "Genitourinary": "#8D6E63",  # Brown
    "Mental health": "#78909C",  # Blue Grey
    "Mental": "#90A4AE",         # Grey
    "Ear": "#5C6BC0",           # Indigo
    "Eye": "#29B6F6",           # Light Blue
    "Circulatory": "#EF5350",    # Red
    "Skin": "#FF8A65"           # Light Orange
}

# Clinical risk level definitions with evidence thresholds
RISK_LEVELS = {
    "High": {
        "threshold": 5.0,
        "color": "#ef5350",
        "description": "Strong evidence of association (OR ≥ 5.0)"
    },
    "Moderate": {
        "threshold": 3.0,
        "color": "#ff9800",
        "description": "Moderate evidence of association (OR ≥ 3.0)"
    },
    "Low": {
        "threshold": 2.0,
        "color": "#4caf50",
        "description": "Weak evidence of association (OR ≥ 2.0)"
    }
}

# Data validation schema
REQUIRED_COLUMNS = {
    'TotalPatientsInGroup': 'integer',
    'ConditionA': 'string',
    'ConditionB': 'string',
    'OddsRatio': 'float',
    'PairFrequency': 'integer',
    'MedianDurationYearsWithIQR': 'string',
    'DirectionalPercentage': 'float',
    'Precedence': 'string'
}

def validate_data(df):
    """Validate input data against required schema"""
    validation_errors = []
    
    # Check required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        validation_errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    if not validation_errors:
        # Check data types
        for col, dtype in REQUIRED_COLUMNS.items():
            try:
                if dtype == 'string':
                    df[col].astype(str)
                elif dtype == 'float':
                    pd.to_numeric(df[col])
                elif dtype == 'integer':
                    pd.to_numeric(df[col], downcast='integer')
            except:
                validation_errors.append(f"Invalid data type in column {col}. Expected {dtype}.")
        
        # Check value ranges
        if (df['OddsRatio'] < 0).any():
            validation_errors.append("Odds ratios cannot be negative.")
        if (df['PairFrequency'] < 0).any():
            validation_errors.append("Frequencies cannot be negative.")
        if (df['DirectionalPercentage'] < 0).any() or (df['DirectionalPercentage'] > 100).any():
            validation_errors.append("Directional percentages must be between 0 and 100.")
            
    return validation_errors

def parse_iqr(iqr_string):
    """Parse IQR string with error handling"""
    try:
        if pd.isna(iqr_string):
            return 0.0, 0.0, 0.0
        median_str, iqr = iqr_string.split(' [')
        q1, q3 = iqr.strip(']').split('-')
        return float(median_str), float(q1), float(q3)
    except:
        return 0.0, 0.0, 0.0

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and validate clinical data"""
    try:
        data = pd.read_csv(uploaded_file)
        
        # Validate data
        validation_errors = validate_data(data)
        if validation_errors:
            error_message = "\n".join(validation_errors)
            st.error(f"Data Validation Errors:\n{error_message}")
            return None, None, None, None
        
        # Extract metadata
        total_patients = data['TotalPatientsInGroup'].iloc[0]
        filename = uploaded_file.name.lower()
        
        # Determine cohort characteristics
        gender = next((g for g in ['Female', 'Male'] 
                      if g.lower() in filename), 'Unknown Gender')
        
        age_groups = {
            'below45': '<45',
            '45to64': '45-64',
            '65plus': '65+'
        }
        age_group = next((ag for pattern, ag in age_groups.items() 
                         if pattern in filename), 'Unknown Age Group')
        
        return data, total_patients, gender, age_group
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None, None

def analyze_condition_combinations(data, min_prev, min_freq):
    """Analyze condition combinations and identify patterns"""
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]
    
    filtered_data = data[
        (data['PairFrequency'] >= min_freq) & 
        (data['PairFrequency'] / total_patients * 100 >= min_prev)
    ].copy()
    
    for _, row in filtered_data.iterrows():
        median, q1, q3 = parse_iqr(row['MedianDurationYearsWithIQR'])
        risk_level, _ = get_risk_level(row['OddsRatio'])
        
        results.append({
            'Condition Pair': f"{row['ConditionA']} → {row['ConditionB']}",
            'System Interaction': f"{condition_categories.get(row['ConditionA'], 'Other')} → {condition_categories.get(row['ConditionB'], 'Other')}",
            'Risk Level': risk_level,
            'Odds Ratio': row['OddsRatio'],
            'Cases': row['PairFrequency'],
            'Prevalence (%)': row['PairFrequency'] / total_patients * 100,
            'Median Time (years)': median,
            'Direction Confidence (%)': row['DirectionalPercentage']
        })
    
    return pd.DataFrame(results)

def get_risk_level(odds_ratio):
    """Calculate clinical risk level based on evidence thresholds"""
    for level, info in RISK_LEVELS.items():
        if odds_ratio >= info["threshold"]:
            return level, info["color"]
    return "Low", RISK_LEVELS["Low"]["color"]

@st.cache_data
def perform_sensitivity_analysis(data):
    """Perform comprehensive sensitivity analysis with clinical risk stratification"""
    or_thresholds = [2.0, 3.0, 4.0, 5.0]
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    # Identify top patterns with risk assessment
    top_patterns = data.nlargest(5, 'OddsRatio')[
        ['ConditionA', 'ConditionB', 'OddsRatio', 'PairFrequency',
         'MedianDurationYearsWithIQR', 'DirectionalPercentage', 'Precedence']
    ].to_dict('records')

    for pattern in top_patterns:
        risk_level, _ = get_risk_level(pattern['OddsRatio'])
        pattern['RiskLevel'] = risk_level

    # Analyze different thresholds
    for threshold in or_thresholds:
        filtered_data = data[data['OddsRatio'] >= threshold].copy()
        
        # Calculate metrics
        n_trajectories = len(filtered_data)
        total_pairs = filtered_data['PairFrequency'].sum()
        estimated_unique_patients = total_pairs / 2
        coverage = min((estimated_unique_patients / total_patients) * 100, 100.0)

        # Analyze system interactions
        system_pairs = set()
        risk_distribution = {'High': 0, 'Moderate': 0, 'Low': 0}
        system_interaction_counts = {}

        for _, row in filtered_data.iterrows():
            sys_a = condition_categories.get(row['ConditionA'], 'Other')
            sys_b = condition_categories.get(row['ConditionB'], 'Other')
            
            if sys_a != sys_b:
                pair = tuple(sorted([sys_a, sys_b]))
                system_pairs.add(pair)
                system_interaction_counts[pair] = system_interaction_counts.get(pair, 0) + 1

            risk_level, _ = get_risk_level(row['OddsRatio'])
            risk_distribution[risk_level] += 1

        # Calculate temporal metrics
        # Calculate temporal metrics
        duration_stats = filtered_data['MedianDurationYearsWithIQR'].apply(parse_iqr)
        medians = [x[0] for x in duration_stats if x[0] > 0]
        q1s = [x[1] for x in duration_stats if x[1] > 0]
        q3s = [x[2] for x in duration_stats if x[2] > 0]

        # Compile results
        results.append({
            'OR_Threshold': threshold,
            'Num_Trajectories': n_trajectories,
            'Coverage_Percent': round(coverage, 2),
            'System_Pairs': len(system_pairs),
            'System_Interactions': dict(system_interaction_counts),
            'Median_Duration': round(np.median(medians) if medians else 0, 2),
            'Q1_Duration': round(np.median(q1s) if q1s else 0, 2),
            'Q3_Duration': round(np.median(q3s) if q3s else 0, 2),
            'High_Risk_Count': risk_distribution['High'],
            'Moderate_Risk_Count': risk_distribution['Moderate'],
            'Low_Risk_Count': risk_distribution['Low'],
            'Top_Patterns': top_patterns
        })

    return pd.DataFrame(results)

def create_clinical_metrics_dashboard(data, results_df):
    """Create summary dashboard with key clinical metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="data-metrics">
            <div class="metric-label">Total Trajectories</div>
            <div class="metric-value">{results_df.iloc[0]['Num_Trajectories']:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="data-metrics">
            <div class="metric-label">High Risk Patterns</div>
            <div class="metric-value">{results_df.iloc[0]['High_Risk_Count']:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="data-metrics">
            <div class="metric-label">Population Coverage</div>
            <div class="metric-value">{results_df.iloc[0]['Coverage_Percent']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="data-metrics">
            <div class="metric-label">Median Progression Time</div>
            <div class="metric-value">{results_df.iloc[0]['Median_Duration']:.1f} years</div>
        </div>
        """, unsafe_allow_html=True)

def create_clinical_report(data, results_df, gender, age_group):
    """Generate comprehensive clinical analysis report"""
    report = f"""
    # Clinical Trajectory Analysis Report
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    **Cohort:** {gender}s, Age Group {age_group}
    **Total Patients:** {data['TotalPatientsInGroup'].iloc[0]:,}

    ## Key Findings

    ### 1. Risk Distribution
    - High Risk Trajectories (OR ≥ 5): {results_df.iloc[0]['High_Risk_Count']}
    - Moderate Risk Trajectories (OR ≥ 3): {results_df.iloc[0]['Moderate_Risk_Count']}
    - Low Risk Trajectories (OR ≥ 2): {results_df.iloc[0]['Low_Risk_Count']}

    ### 2. Population Coverage
    - Maximum Coverage: {results_df['Coverage_Percent'].max():.1f}%
    - Minimum OR 2.0 Coverage: {results_df.iloc[0]['Coverage_Percent']:.1f}%

    ### 3. Temporal Characteristics
    - Median Progression Time: {results_df.iloc[0]['Median_Duration']:.1f} years
    - Interquartile Range: [{results_df.iloc[0]['Q1_Duration']:.1f} - {results_df.iloc[0]['Q3_Duration']:.1f}] years

    ### 4. System Interactions
    """
    
    # Add system interaction analysis
    system_interactions = results_df.iloc[0]['System_Interactions']
    if system_interactions:
        report += "**Most Common System Interactions:**\n"
        for (sys1, sys2), count in sorted(system_interactions.items(), 
                                        key=lambda x: x[1], reverse=True)[:5]:
            report += f"- {sys1} ↔ {sys2}: {count} trajectories\n"

    return report

def main():
    """Main application interface"""
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1>🏥 Clinical Trajectory Analysis Platform</h1>
        <p style='font-size: 1.2rem; color: #666;'>
            Advanced analysis of disease progression patterns for clinical research
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Clinical Data (CSV)",
        type="csv",
        help="Upload a CSV file containing patient trajectory data with required clinical parameters"
    )
    
    if uploaded_file is not None:
        with st.spinner("Validating and processing clinical data..."):
            data, total_patients, gender, age_group = load_and_process_data(uploaded_file)
            
            if data is not None:
                st.markdown(f"""
                <div class="alert alert-info">
                    <h3 style='margin-top: 0;'>📊 Cohort Information</h3>
                    <ul style='list-style-type: none; padding-left: 0;'>
                        <li><strong>Total Patients:</strong> {total_patients:,}</li>
                        <li><strong>Gender:</strong> {gender}</li>
                        <li><strong>Age Group:</strong> {age_group}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                # Create analysis tabs
                tabs = st.tabs([
                    "📈 Sensitivity Analysis",
                    "🔄 Trajectory Prediction",
                    "🔍 Clinical Patterns",
                    "📊 Clinical Trial Analysis",
                    "📋 Summary Report"
                ])

                # Sensitivity Analysis Tab
                with tabs[0]:
                    st.markdown("""
                    <h2>Clinical Sensitivity Analysis</h2>
                    <p>Analyze the relationship between evidence thresholds and trajectory patterns</p>
                    """, unsafe_allow_html=True)
                    
                    analysis_col1, analysis_col2 = st.columns([3, 1])
                    
                    with analysis_col2:
                        st.markdown("### Analysis Controls")
                        analyze_button = st.button(
                            "Run Analysis",
                            help="Perform comprehensive sensitivity analysis"
                        )
                    
                    with analysis_col1:
                        if analyze_button:
                            with st.spinner("Performing clinical trajectory analysis..."):
                                results = perform_sensitivity_analysis(data)
                                create_clinical_metrics_dashboard(data, results)

                # Trajectory Prediction Tab
                with tabs[1]:
                    st.markdown("""
                    <h2>Trajectory Prediction</h2>
                    <p>Predict potential disease progression pathways</p>
                    """, unsafe_allow_html=True)
                    
                    pred_col1, pred_col2 = st.columns([2, 1])
                    
                    with pred_col2:
                        st.markdown("### Prediction Parameters")
                        # Add prediction controls here

                # Clinical Patterns Tab
                with tabs[2]:
                    st.markdown("""
                    <h2>Clinical Patterns</h2>
                    <p>Analyze disease progression patterns and relationships</p>
                    """, unsafe_allow_html=True)
                    
                    pattern_col1, pattern_col2 = st.columns([2, 1])
                    
                    with pattern_col2:
                        st.markdown("### Pattern Analysis")
                        # Add pattern analysis controls here

                # Clinical Trial Analysis Tab
                with tabs[3]:
                    st.markdown("""
                    <h2>Clinical Trial Analysis</h2>
                    <p>Analyze clinical trial outcomes and safety data</p>
                    """, unsafe_allow_html=True)
                    
                    trial_col1, trial_col2 = st.columns([2, 1])
                    
                    with trial_col2:
                        st.markdown("### Trial Parameters")
                        trial_phase = st.selectbox(
                            "Trial Phase",
                            ["Phase I", "Phase II", "Phase III", "Phase IV"]
                        )
                        
                        outcome_metric = st.selectbox(
                            "Primary Outcome",
                            ["Efficacy", "Safety", "Both"]
                        )
                        
                        analysis_period = st.slider(
                            "Analysis Period (months)",
                            1, 60, 12
                        )

                # Summary Report Tab
                with tabs[4]:
                    st.markdown("""
                    <h2>Summary Report</h2>
                    <p>Generate comprehensive clinical analysis report</p>
                    """, unsafe_allow_html=True)
                    
                    if 'results' in locals():
                        report = create_clinical_report(data, results, gender, age_group)
                        st.markdown(report)

if __name__ == "__main__":
    main()
