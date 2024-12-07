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

# Disease category mappings
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

# System colors for visualization
SYSTEM_COLORS = {
    "Endocrine": "#BA55D3",
    "Blood": "#DC143C",
    "Digestive": "#32CD32", 
    "Respiratory": "#48D1CC",
    "Neoplasms": "#800080",
    "Cardiovascular": "#FF4500",
    "Nervous": "#FFD700",
    "Musculoskeletal": "#4682B4",
    "Genitourinary": "#DAA520",
    "Mental health": "#8B4513",
    "Mental": "#A0522D",
    "Ear": "#4169E1",
    "Eye": "#20B2AA",
    "Circulatory": "#FF6347",
    "Skin": "#F08080"
}

def parse_iqr(iqr_string):
    """Parse IQR string of format 'median [Q1-Q3]' into (median, q1, q3)"""
    try:
        median_str, iqr = iqr_string.split(' [')
        q1, q3 = iqr.strip(']').split('-')
        return float(median_str), float(q1), float(q3)
    except:
        return 0.0, 0.0, 0.0

def load_and_process_data(uploaded_file):
    """Load and process the uploaded CSV file"""
    try:
        data = pd.read_csv(uploaded_file)
        total_patients = data['TotalPatientsInGroup'].iloc[0]

        filename = uploaded_file.name.lower()

        if 'females' in filename:
            gender = 'Female'
        elif 'males' in filename:
            gender = 'Male'
        else:
            gender = 'Unknown Gender'

        if 'below45' in filename:
            age_group = '<45'
        elif '45to64' in filename:
            age_group = '45-64'
        elif '65plus' in filename:
            age_group = '65+'
        else:
            age_group = 'Unknown Age Group'

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

    # Get top 5 patterns from full dataset first
    top_patterns = data.nlargest(5, 'OddsRatio')[
        ['ConditionA', 'ConditionB', 'OddsRatio', 'PairFrequency',
         'MedianDurationYearsWithIQR', 'DirectionalPercentage', 'Precedence']
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

@st.cache_data
def create_network_graph(data, patient_conditions, min_or, time_horizon=None, time_margin=None):
    """Create network graph for trajectory visualization with legend"""
    # Create HTML for the legend
    legend_html = """
    <div style="position: absolute; top: 10px; right: 10px; background: white; 
                padding: 10px; border: 1px solid #ddd; border-radius: 5px; z-index: 1000;">
        <h3 style="margin-top: 0; margin-bottom: 10px;">Legend</h3>
        <div style="margin-bottom: 10px;">
            <strong>Node Types:</strong><br>
            ★ Initial Condition<br>
            ○ Related Condition
        </div>
        <div>
            <strong>Body Systems:</strong><br>
    """
    
    for system, color in SYSTEM_COLORS.items():
        legend_html += f"""
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 15px; height: 15px; background-color: {color}50; 
                 border: 1px solid {color}; margin-right: 5px;"></div>
            <span>{system}</span>
        </div>
        """
    
    legend_html += """
        </div>
        <div style="margin-top: 10px;">
            <strong>Edge Information:</strong><br>
            • Edge thickness indicates strength of association<br>
            • Arrow indicates typical progression direction<br>
            • Hover over edges for detailed statistics
        </div>
    </div>
    """

    net = Network(height="800px", width="100%", bgcolor='white', font_color='black', directed=True)
    
    # Network options
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 16},
            "scaling": {"min": 10, "max": 30}
        },
        "edges": {
            "color": {"inherit": false},
            "font": {
                "size": 12,
                "align": "middle",
                "multi": true,
                "background": "rgba(255, 255, 255, 0.8)"
            },
            "smooth": {
                "type": "continuous",
                "roundness": 0.2
            }
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -4000,
                "centralGravity": 0.1,
                "springLength": 250,
                "springConstant": 0.03,
                "damping": 0.1,
                "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "stabilization": {
                "enabled": true,
                "iterations": 1000,
                "updateInterval": 25
            }
        }
    }
    """)
    
    # Filter data and identify connected conditions
    filtered_data = data[data['OddsRatio'] >= min_or].copy()
    connected_conditions = set()
    
    for condition_a in patient_conditions:
        time_filtered_data = filtered_data
        if time_horizon and time_margin:
            time_filtered_data = filtered_data[
                (filtered_data['ConditionA'] == condition_a) &
                (filtered_data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin))
            ]
        conditions_b = set(time_filtered_data[time_filtered_data['ConditionA'] == condition_a]['ConditionB'])
        connected_conditions.update(conditions_b)
    
    active_conditions = set(patient_conditions) | connected_conditions
    active_categories = {condition_categories[cond] for cond in active_conditions if cond in condition_categories}

    # Organize conditions by system
    system_conditions = {}
    for condition in active_conditions:
        category = condition_categories.get(condition, "Other")
        if category not in system_conditions:
            system_conditions[category] = []
        system_conditions[category].append(condition)

    # Calculate layout positions
    angle_step = (2 * math.pi) / len(active_categories)
    radius = 500
    system_centers = {}

    for i, category in enumerate(sorted(active_categories)):
        angle = i * angle_step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        system_centers[category] = (x, y)

    # Add nodes
    for category, conditions in system_conditions.items():
        center_x, center_y = system_centers[category]
        sub_radius = radius / (len(conditions) + 1)
        
        for j, condition in enumerate(conditions):
            sub_angle = (j / len(conditions)) * (2 * math.pi)
            node_x = center_x + sub_radius * math.cos(sub_angle)
            node_y = center_y + sub_radius * math.sin(sub_angle)
            
            base_color = SYSTEM_COLORS[category]
            
            if condition in patient_conditions:
                net.add_node(
                    condition,
                    label=f"★ {condition}",
                    title=f"{condition}\nCategory: {category}",
                    size=30,
                    x=node_x,
                    y=node_y,
                    color={'background': f"{base_color}50", 'border': '#000000'},
                    physics=True,
                    fixed=False
                )
            else:
                net.add_node(
                    condition,
                    label=condition,
                    title=f"{condition}\nCategory: {category}",
                    size=20,
                    x=node_x,
                    y=node_y,
                    color={'background': f"{base_color}50", 'border': base_color},
                    physics=True,
                    fixed=False
                )

    # Add edges with trajectory information
    total_patients = data['TotalPatientsInGroup'].iloc[0]
    for condition_a in patient_conditions:
        relevant_data = filtered_data[filtered_data['ConditionA'] == condition_a]
        if time_horizon and time_margin:
            relevant_data = relevant_data[
                relevant_data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin)
            ]

        for _, row in relevant_data.iterrows():
            condition_b = row['ConditionB']
            if condition_b not in patient_conditions:
                edge_width = max(1, min(8, math.log2(row['OddsRatio'] + 1)))
                prevalence = (row['PairFrequency'] / total_patients) * 100
                directional_percentage = row['DirectionalPercentage']

                if directional_percentage >= 50:
                    source, target = condition_a, condition_b
                else:
                    source, target = condition_b, condition_a
                    directional_percentage = 100 - directional_percentage

                edge_label = (f"OR: {row['OddsRatio']:.1f}\n"
                            f"Years: {row['MedianDurationYearsWithIQR']}\n"
                            f"n={row['PairFrequency']} ({prevalence:.1f}%)\n"
                            f"Proceeds: {directional_percentage:.1f}%")

                net.add_edge(
                    source,
                    target,
                    label=edge_label,
                    title=edge_label,
                    width=edge_width,
                    arrows={'to': {'enabled': True, 'scaleFactor': 1}},
                    color={'color': 'rgba(128,128,128,0.7)', 'highlight': 'black'},
                    smooth={'type': 'curvedCW', 'roundness': 0.2}
                )

    # Generate final HTML with legend
    network_html = net.generate_html()
    final_html = network_html.replace('</body>', f'{legend_html}</body>')
    
    return final_html

@st.cache_data
def analyze_condition_combinations(data, min_percentage, min_frequency):
    """Analyze combinations of conditions"""
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    filtered_data = data[
        (data['Percentage'] >= min_percentage) &
        (data['PairFrequency'] >= min_frequency)
    ].copy()

    # Clean condition names
    for col in ['ConditionA', 'ConditionB']:
        filtered_data[col] = (filtered_data[col]
                            .str.replace(r'\s*\([^)]*\)', '', regex=True)
                            .str.replace('_', ' '))

    unique_conditions = pd.unique(filtered_data[['ConditionA', 'ConditionB']].values.ravel('K'))
    
    # Calculate frequencies
    pair_frequency_map = {}
    condition_frequency_map = {}
    
    for _, row in filtered_data.iterrows():
        for key in [f"{row['ConditionA']}_{row['ConditionB']}", 
                   f"{row['ConditionB']}_{row['ConditionA']}"]:
            pair_frequency_map[key] = row['PairFrequency']
        
        for condition in [row['ConditionA'], row['ConditionB']]:
            condition_frequency_map[condition] = (
                condition_frequency_map.get(condition, 0) + row['PairFrequency']
            )

    # Analyze combinations
    result_data = []
    for k in range(3, min(8, len(unique_conditions) + 1)):
        for comb in combinations(unique_conditions, k):
            pair_frequencies = [
                pair_frequency_map.get(f"{a}_{b}", 0) 
                for a, b in combinations(comb, 2)
            ]
            
            frequency = min(pair_frequencies)
            prevalence = (frequency / total_patients) * 100
            
            # Calculate odds ratio
            observed = frequency
            expected = total_patients
            for condition in comb:
                expected *= (condition_frequency_map[condition] / total_patients)
            
            odds_ratio = observed / expected if expected != 0 else float('inf')
            
            result_data.append({
                'Combination': ' + '.join(comb),
                'NumConditions': len(comb),
                'Minimum Pair Frequency': frequency,
                'Prevalence of the combination (%)': prevalence,
                'Total odds ratio': odds_ratio
            })

    results_df = pd.DataFrame(result_data)
    results_df = (results_df[results_df['Prevalence of the combination (%)'] > 0]
                 .sort_values('Prevalence of the combination (%)', ascending=False))
    
    return results_df

def create_sensitivity_plot(results):
    """Create the sensitivity analysis visualization"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x_vals = results['OR_Threshold'].values
    bar_heights = results['Num_Trajectories']

    # Plot bars and lines
    bars = ax1.bar(x_vals, bar_heights, alpha=0.3, color='navy')
    line = ax2.plot(x_vals, results['Coverage_Percent'], 'r-o', linewidth=2)

    # Add scatter plot with variable sizes
    sizes = (results['System_Pairs'] / results['System_Pairs'].max()) * 500
    scatter = ax2.scatter(x_vals, results['Coverage_Percent'], s=sizes, alpha=0.5, color='darkred')

    # Add text annotations
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

def create_combinations_plot(results_df):
    """Create the combinations analysis visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    top_10 = results_df.nlargest(10, 'Prevalence of the combination (%)')
    bars = ax.bar(range(len(top_10)), top_10['Prevalence of the combination (%)'])
    
    # Customize the plot
    ax.set_xticks(range(len(top_10)))
    ax.set_xticklabels(top_10['Combination'], rotation=45, ha='right')
    ax.set_title('Top 10 Condition Combinations by Prevalence')
    ax.set_xlabel('Condition Combinations')
    ax.set_ylabel('Prevalence (%)')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

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

            # Condition Combinations Tab
            with tabs[2]:
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
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No combinations found matching the criteria.")

if __name__ == "__main__":
    main()
