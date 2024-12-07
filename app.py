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

def create_personalized_analysis_html(data, patient_conditions, total_patients, time_horizon=None, time_margin=None, min_or=2.0):
    """Create HTML for personalized trajectory analysis"""
    def get_risk_level(odds_ratio):
        if odds_ratio >= 5:
            return "High", "#dc3545"
        elif odds_ratio >= 3:
            return "Moderate", "#ffc107"
        else:
            return "Low", "#28a745"

    filtered_data = data[data['OddsRatio'] >= min_or].copy()

    html = """
    <style>
        .patient-analysis {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 1200px;
            margin: 20px auto;
        }
        .condition-section {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .condition-header {
            font-size: 1.2em;
            color: #2c5282;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }
        .trajectory-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            background-color: white;
        }
        .trajectory-table th {
            background-color: #f5f5f5;
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }
        .trajectory-table td {
            padding: 10px;
            border: 1px solid #ddd;
        }
        .risk-badge {
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
            font-weight: bold;
        }
        .system-tag {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: #e2e8f0;
            font-size: 0.9em;
            margin-right: 5px;
        }
        .timeline-indicator {
            font-style: italic;
            color: #666;
        }
    </style>
    <div class="patient-analysis">
        <h2>Personalized Disease Trajectory Analysis</h2>
        <p>Based on current conditions: """ + ", ".join(patient_conditions) + "</p>"

    for condition_a in patient_conditions:
        time_filtered_data = filtered_data[filtered_data['ConditionA'] == condition_a]
        if time_horizon and time_margin:
            time_filtered_data = time_filtered_data[
                time_filtered_data['MedianDurationYearsWithIQR'].apply(
                    lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin)
            ]

        if not time_filtered_data.empty:
            system_a = condition_categories.get(condition_a, 'Other')
            html += f"""
            <div class="condition-section">
                <div class="condition-header">
                    <span class="system-tag">{system_a}</span>
                    Progression Paths from {condition_a}
                </div>
                <table class="trajectory-table">
                    <thead>
                        <tr>
                            <th>Risk Level</th>
                            <th>Potential Progression</th>
                            <th>Expected Timeline</th>
                            <th>Statistical Support</th>
                            <th>Progression Details</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for _, row in time_filtered_data.sort_values('OddsRatio', ascending=False).iterrows():
                condition_b = row['ConditionB']
                if condition_b not in patient_conditions:
                    system_b = condition_categories.get(condition_b, 'Other')
                    median, q1, q3 = parse_iqr(row['MedianDurationYearsWithIQR'])
                    prevalence = (row['PairFrequency'] / total_patients) * 100
                    risk_level, color = get_risk_level(row['OddsRatio'])

                    if row['DirectionalPercentage'] >= 50:
                        direction = f"{condition_a} → {condition_b}"
                        confidence = row['DirectionalPercentage']
                    else:
                        direction = f"{condition_b} → {condition_a}"
                        confidence = 100 - row['DirectionalPercentage']

                    html += f"""
                        <tr>
                            <td><span class="risk-badge" style="background-color: {color}">{risk_level}</span></td>
                            <td>
                                <strong>{condition_b}</strong><br>
                                <span class="system-tag">{system_b}</span>
                            </td>
                            <td class="timeline-indicator">
                                Typically {median:.1f} years<br>
                                Range: {q1:.1f} to {q3:.1f} years
                            </td>
                            <td>
                                OR: {row['OddsRatio']:.1f}<br>
                                {row['PairFrequency']} cases ({prevalence:.1f}%)
                            </td>
                            <td>
                                {confidence:.1f}% confidence in progression order<br>
                                {direction}
                            </td>
                        </tr>
                    """

            html += """
                    </tbody>
                </table>
            </div>
            """

    html += """
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
            <h4>Understanding This Analysis:</h4>
            <ul>
                <li><strong>Risk Level:</strong> Based on odds ratio strength (High: OR≥5, Moderate: OR≥3, Low: OR≥2)</li>
                <li><strong>Expected Timeline:</strong> Median years and range between which progression typically occurs</li>
                <li><strong>Statistical Support:</strong> Odds ratio and number of observed cases in the population</li>
                <li><strong>Progression Details:</strong> Confidence in the order of disease progression</li>
            </ul>
        </div>
    </div>
    """

    return html

def main():
    # Page configuration
    st.set_page_config(
        page_title="Multimorbidity Analysis Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            width: 100%;
            height: 3rem;
            margin: 1rem 0;
            background-color: #ff4b4b;
            color: white;
        }
        .stButton > button:hover {
            background-color: #ff3333;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            padding: 1rem 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 4rem;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e6e9ef;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #ff4b4b;
            color: white;
        }
        div[data-testid="stSidebarNav"] {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
        }
        div[data-testid="stFileUploader"] {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 4px;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🏥 Multimorbidity Analysis Tool")
    st.markdown("""
    This tool helps analyze disease trajectories and comorbidity patterns in patient data.
    Upload your data file to begin analysis.
    """)

    # File uploader in a container
    with st.container():
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file containing your patient data"
        )

    if uploaded_file is not None:
        # Load and process data
        data, total_patients, gender, age_group = load_and_process_data(uploaded_file)

        if data is not None:
            # Data summary in an info box
            st.info(f"""
            📊 **Data Summary**
            - Total Patients: {total_patients:,}
            - Gender: {gender}
            - Age Group: {age_group}
            """)

            # Create tabs with icons
            tabs = st.tabs([
                "📈 Sensitivity Analysis",
                "🔄 Trajectory Prediction",
                "📋 Personalized Analysis",
                "🔍 Condition Combinations"
            ])

            # Sensitivity Analysis Tab
            with tabs[0]:
                st.header("Sensitivity Analysis")
                st.markdown("""
                Explore how different odds ratio thresholds affect the number of disease
                trajectories and population coverage.
                """)
                
                analysis_col1, analysis_col2 = st.columns([3, 1])
                
                with analysis_col2:
                    st.markdown("### Control Panel")
                    analyze_button = st.button(
                        "🚀 Run Analysis",
                        key="run_sensitivity",
                        help="Click to perform sensitivity analysis"
                    )

                with analysis_col1:
                    if analyze_button:
                        with st.spinner("💫 Analyzing data..."):
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

                            # Visualization
                            fig = create_sensitivity_plot(results)
                            st.pyplot(fig)

                            # Download button
                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="sensitivity_analysis_results.csv",
                                mime="text/csv"
                            )

            # Trajectory Prediction Tab
            with tabs[1]:
                st.header("Trajectory Prediction")
                
                viz_col, param_col = st.columns([3, 1])
                
                with param_col:
                    st.markdown("### Parameters")
                    with st.container():
                        min_or = st.slider(
                            "Minimum Odds Ratio",
                            1.0, 10.0, 2.0, 0.5,
                            help="Filter trajectories by minimum odds ratio"
                        )
                        
                        unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
                        selected_conditions = st.multiselect(
                            "Select Initial Conditions",
                            unique_conditions,
                            help="Choose the starting conditions for trajectory analysis"
                        )

                        if selected_conditions:
                            max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]).max())
                            time_horizon = st.slider(
                                "Time Horizon (years)",
                                1, max_years, min(5, max_years),
                                help="Maximum time period to consider"
                            )
                            
                            time_margin = st.slider(
                                "Time Margin",
                                0.0, 0.5, 0.1, 0.05,
                                help="Allowable variation in time predictions"
                            )

                            generate_button = st.button(
                                "🔄 Generate Network",
                                help="Click to generate trajectory network"
                            )

                with viz_col:
                    if selected_conditions and generate_button:
                        with st.spinner("🌐 Generating network..."):
                            try:
                                html_content = create_network_graph(
                                    data,
                                    selected_conditions,
                                    min_or,
                                    time_horizon,
                                    time_margin
                                )
                                st.components.v1.html(html_content, height=800)
                                
                                st.download_button(
                                    label="📥 Download Network",
                                    data=html_content,
                                    file_name="trajectory_network.html",
                                    mime="text/html"
                                )
                            except Exception as e:
                                st.error(f"Failed to generate network: {str(e)}")

            # Personalized Analysis Tab
            with tabs[2]:
                st.header("Personalized Trajectory Analysis")
                st.markdown("""
                This analysis provides a detailed view of potential disease progressions based on 
                a patient's current conditions, with risk levels and timelines.
                """)
                
                analysis_col1, analysis_col2 = st.columns([3, 1])
                
                with analysis_col2:
                    st.markdown("### Analysis Parameters")
                    
                    unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
                    selected_conditions = st.multiselect(
                        "Select Current Conditions",
                        unique_conditions,
                        help="Choose the patient's current conditions",
                        key="personalized_conditions"
                    )
                    
                    min_or = st.slider(
                        "Minimum Odds Ratio",
                        1.0, 10.0, 2.0, 0.5,
                        help="Filter trajectories by minimum odds ratio",
                        key="personalized_min_or"
                    )
                    
                    if selected_conditions:
                        max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]).max())
                        time_horizon = st.slider(
                            "Time Horizon (years)",
                            1, max_years, min(5, max_years),
                            help="Maximum time period to consider",
                            key="personalized_time_horizon"
                        )
                        
                        time_margin = st.slider(
                            "Time Margin",
                            0.0, 0.5, 0.1, 0.05,
                            help="Allowable variation in time predictions",
                            key="personalized_time_margin"
                        )
                    
                    analyze_button = st.button(
                        "🔍 Analyze Trajectories",
                        help="Generate personalized trajectory analysis",
                        key="personalized_analyze"
                    )

                with analysis_col1:
                    if selected_conditions and analyze_button:
                        with st.spinner("Generating personalized analysis..."):
                            html_content = create_personalized_analysis_html(
                                data,
                                selected_conditions,
                                total_patients,
                                time_horizon,
                                time_margin,
                                min_or
                            )
                            
                            st.components.v1.html(html_content, height=800, scrolling=True)
                            
                            st.download_button(
                                label="📥 Download Analysis",
                                data=html_content,
                                file_name="personalized_trajectory_analysis.html",
                                mime="text/html"
                            )

            # Condition Combinations Tab
            with tabs[3]:
                st.header("Condition Combinations Analysis")
                
                param_col, results_col = st.columns([1, 3])
                
                with param_col:
                    st.markdown("### Analysis Parameters")
                    
                    min_freq_range = (data['PairFrequency'].min(), data['PairFrequency'].max())
                    min_frequency = st.slider(
                        "Minimum Pair Frequency",
                        int(min_freq_range[0]),
                        int(min_freq_range[1]),
                        int(min_freq_range[0]),
                        help="Minimum number of occurrences required"
                    )
                    
                    min_percentage_range = (data['Percentage'].min(), data['Percentage'].max())
                    min_percentage = st.slider(
                        "Minimum Prevalence (%)",
                        float(min_percentage_range[0]),
                        float(min_percentage_range[1]),
                        float(min_percentage_range[0]),
                        0.1,
                        help="Minimum percentage of population affected"
                    )

                    analyze_combinations_button = st.button(
                        "🔍 Analyze Combinations",
                        help="Click to analyze condition combinations"
                    )

                with results_col:
                    if analyze_combinations_button:
                        with st.spinner("🔄 Analyzing combinations..."):
                            results_df = analyze_condition_combinations(
                                data,
                                min_percentage,
                                min_frequency
                            )
                            
                            if not results_df.empty:
                                st.subheader(f"Analysis Results ({len(results_df)} combinations)")
                                st.dataframe(
                                    results_df.style.background_gradient(
                                        cmap='YlOrRd',
                                        subset=['Prevalence of the combination (%)']
                                    )
                                )

                                fig = create_combinations_plot(results_df)
                                st.pyplot(fig)

                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name="condition_combinations.csv",
                                    mime="text/html"
                                )
                            else:
                                st.warning("No combinations found matching the criteria.")

if __name__ == "__main__":
    main()
