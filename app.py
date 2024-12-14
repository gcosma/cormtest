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

def clear_session_state():
    """Clear all analysis results from session state when a new file is uploaded"""
    st.session_state.sensitivity_results = None
    st.session_state.network_html = None
    st.session_state.combinations_results = None
    st.session_state.combinations_fig = None
    st.session_state.personalized_html = None
    st.session_state.selected_conditions = []
    st.session_state.min_or = 2.0
    st.session_state.time_horizon = 5
    st.session_state.time_margin = 0.1
    st.session_state.min_frequency = None
    st.session_state.min_percentage = None
    st.session_state.unique_conditions = []


# Disease category mappings
condition_categories = {
   "Anaemia": "Blood",
    "Cardiac Arrhythmias": "Circulatory",
    "Coronary Heart Disease": "Circulatory",
    "Heart Failure": "Circulatory",
    "Hypertension": "Circulatory",
    "Peripheral Vascular Disease": "Circulatory",
    "Stroke": "Nervous",
    "Barretts Oesophagus": "Digestive",
    "Chronic Constipation": "Digestive",
    "Chronic Diarrhoea": "Digestive",
    "Cirrhosis": "Digestive",
    "Dysphagia": "Digestive",
    "Inflammatory Bowel Disease": "Digestive",
    "Reflux Disorders": "Digestive",
    "Hearing Loss": "Ear",
    "Addisons Disease": "Endocrine",
    "Diabetes": "Endocrine",
    "Polycystic Ovary Syndrome": "Endocrine",
    "Thyroid Disorders": "Endocrine",
    "Visual Impairment": "Eye",
    "Chronic Kidney Disease": "Genitourinary",
    "Menopausal and Perimenopausal": "Genitourinary",
    "Dementia": "Mental",
    "Mental Illness": "Mental",
    "Tourette": "Mental",
    "Chronic Arthritis": "Musculoskeletal",
    "Chronic Pain Conditions": "Musculoskeletal",
    "Osteoporosis": "Musculoskeletal",
    "Cancer": "Neoplasms",
    "Cerebral Palsy": "Nervous",
    "Epilepsy": "Nervous",
    "Insomnia": "Nervous",
    "Multiple Sclerosis": "Nervous",
    "Neuropathic Pain": "Nervous",
    "Parkinsons": "Nervous",
    "Bronchiectasis": "Respiratory",
    "Chronic Airway Diseases": "Respiratory",
    "Chronic Pneumonia": "Respiratory",
    "Interstitial Lung Disease": "Respiratory",
    "Psoriasis": "Skin"
}

# System colors for visualization
SYSTEM_COLORS = {
    "Endocrine": "#BA55D3",
    "Blood": "#DC143C",
    "Digestive": "#32CD32",
    "Respiratory": "#48D1CC",
    "Neoplasms": "#800080",
    "Nervous": "#FFD700",
    "Musculoskeletal": "#4682B4",
    "Genitourinary": "#DAA520",
    "Mental": "#8B4513",
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
def perform_sensitivity_analysis(data, top_n=5):
    """Perform sensitivity analysis with configurable number of top trajectories"""
    or_thresholds = [2.0, 3.0, 4.0, 5.0]
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    # Get top n patterns from full dataset first
    top_patterns = data.nlargest(top_n, 'OddsRatio')[
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
    # Legend HTML remains the same
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

    # Network options remain the same
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

    # Node positioning logic remains the same
    system_conditions = {}
    for condition in active_conditions:
        category = condition_categories.get(condition, "Other")
        if category not in system_conditions:
            system_conditions[category] = []
        system_conditions[category].append(condition)

    angle_step = (2 * math.pi) / len(active_categories)
    radius = 500
    system_centers = {}

    for i, category in enumerate(sorted(active_categories)):
        angle = i * angle_step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        system_centers[category] = (x, y)

    # Add nodes (unchanged)
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

    # Modified edge addition to correctly handle precedence
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

                # Determine direction based on the Precedence field
                if "precedes" in row['Precedence']:
                    parts = row['Precedence'].split(" precedes ")
                    source = parts[0]
                    target = parts[1]
                    directional_percentage = row['DirectionalPercentage']
                else:
                    # Fallback if precedence format is unexpected
                    source = condition_a
                    target = condition_b
                    directional_percentage = row['DirectionalPercentage']

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

# Add this function near the other functions, before main():
def create_personalized_analysis(data, patient_conditions, time_horizon=None, time_margin=None, min_or=2.0):
    """Create a personalized analysis of disease trajectories for a patient's conditions"""
    filtered_data = data[data['OddsRatio'] >= min_or].copy()
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    def get_risk_level(odds_ratio):
        if odds_ratio >= 5:
            return "High", "#dc3545"
        elif odds_ratio >= 3:
            return "Moderate", "#ffc107"
        else:
            return "Low", "#28a745"

    html = """
    <style>
        .patient-analysis {
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont;
            margin: 20px 0;
            width: 100%;
            max-width: 100%;
        }
        .condition-section {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #f8f9fa;
            width: 100%;
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
            font-size: 14px;
        }
        .trajectory-table th {
            background-color: #f5f5f5;
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
            white-space: nowrap;
        }
        .trajectory-table td {
            padding: 10px;
            border: 1px solid #ddd;
            vertical-align: top;
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
        .progression-arrow {
            color: #4a5568;
            font-weight: bold;
        }
        .percentage {
            color: #2d3748;
            font-weight: bold;
        }
        @media (max-width: 1200px) {
            .trajectory-table {
                display: block;
                overflow-x: auto;
                white-space: nowrap;
            }
        }
        .analysis-container {
            max-width: 100%;
            margin: 0 auto;
            padding: 20px;
        }
        .summary-section {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid #e2e8f0;
        }
    </style>
    <div class="patient-analysis">
        <div class="analysis-container">
            <h2>Personalized Disease Trajectory Analysis</h2>
            <div class="summary-section">
                <h3>Current Conditions:</h3>
                <p>""" + ", ".join(f"<span class='system-tag'>{condition_categories.get(cond, 'Other')}</span> {cond}" for cond in patient_conditions) + """</p>
            </div>
    """

    for condition_a in patient_conditions:
        time_filtered_data = filtered_data[
            (filtered_data['ConditionA'] == condition_a) |
            (filtered_data['ConditionB'] == condition_a)
        ]

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
                if row['ConditionA'] == condition_a:
                    other_condition = row['ConditionB']
                    direction_percentage = row['DirectionalPercentage']
                else:
                    other_condition = row['ConditionA']
                    direction_percentage = 100 - row['DirectionalPercentage']

                if other_condition not in patient_conditions:
                    system_b = condition_categories.get(other_condition, 'Other')
                    median, q1, q3 = parse_iqr(row['MedianDurationYearsWithIQR'])
                    prevalence = (row['PairFrequency'] / total_patients) * 100
                    risk_level, color = get_risk_level(row['OddsRatio'])

                    # Parse precedence to determine direction
                    if "precedes" in row['Precedence']:
                        parts = row['Precedence'].split(" precedes ")
                        first_condition = parts[0]
                        second_condition = parts[1]
                        direction = f"{first_condition} <span class='progression-arrow'>→</span> {second_condition}"
                        if first_condition == row['ConditionA']:
                            percentage = row['DirectionalPercentage']
                        else:
                            percentage = 100 - row['DirectionalPercentage']

                        progression_text = f"""
                            {direction}<br>
                            <span class='percentage'>{percentage:.1f}%</span> of cases follow this pattern
                        """
                    else:
                        direction = f"{condition_a} <span class='progression-arrow'>→</span> {other_condition}"
                        progression_text = f"""
                            {direction}<br>
                            <span class='percentage'>{direction_percentage:.1f}%</span> of cases follow this pattern
                        """

                    html += f"""
                        <tr>
                            <td><span class="risk-badge" style="background-color: {color}">{risk_level}</span></td>
                            <td>
                                <strong>{other_condition}</strong><br>
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
                                {progression_text}
                            </td>
                        </tr>
                    """

            html += """
                    </tbody>
                </table>
            </div>
            """

    html += """
            <div class="summary-section">
                <h4>Understanding This Analysis:</h4>
                <ul>
                    <li><strong>Risk Level:</strong> Based on odds ratio strength (High: OR≥5, Moderate: OR≥3, Low: OR≥2)</li>
                    <li><strong>Expected Timeline:</strong> Median years and range between which progression typically occurs</li>
                    <li><strong>Statistical Support:</strong> Odds ratio and number of observed cases in the population</li>
                    <li><strong>Progression Details:</strong> Direction of progression and percentage of cases that follow this pattern</li>
                </ul>
            </div>
        </div>
    </div>
    """

    return html

# Add these imports at the top with your other imports
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run or password not correct
    if "password_correct" not in st.session_state:
        # Show input for password
        st.text_input(
            "Please enter the password",
            type="password",
            key="password",
            on_change=password_entered
        )
        return False

    # Password correct
    elif st.session_state["password_correct"]:
        return True
        
def main():
    # Initialize session state for data persistence
    if 'sensitivity_results' not in st.session_state:
        st.session_state.sensitivity_results = None
    if 'network_html' not in st.session_state:
        st.session_state.network_html = None
    if 'combinations_results' not in st.session_state:
        st.session_state.combinations_results = None
    if 'combinations_fig' not in st.session_state:
        st.session_state.combinations_fig = None
    if 'personalized_html' not in st.session_state:
        st.session_state.personalized_html = None
    if 'selected_conditions' not in st.session_state:
        st.session_state.selected_conditions = []
    if 'min_or' not in st.session_state:
        st.session_state.min_or = 2.0
    if 'time_horizon' not in st.session_state:
        st.session_state.time_horizon = 5
    if 'time_margin' not in st.session_state:
        st.session_state.time_margin = 0.1
    if 'min_frequency' not in st.session_state:
        st.session_state.min_frequency = None
    if 'min_percentage' not in st.session_state:
        st.session_state.min_percentage = None
    if 'unique_conditions' not in st.session_state:
        st.session_state.unique_conditions = []
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    if 'top_n_trajectories' not in st.session_state:
        st.session_state.top_n_trajectories = 5

    # Page configuration
    st.set_page_config(
        page_title="Multimorbidity Analysis Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check password before showing any content
    if not check_password():
        st.stop()

    # Custom CSS
    st.markdown("""
    <style>
    /* Base Styles */
    .main {
        padding: 2rem;
    }
    
    /* Control Panel Container */
    .control-panel {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        margin-top: 3.7rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Control Panel Header */
    .control-panel h3 {
        color: #333333;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
        letter-spacing: 0.025em;
    }
    
    /* Slider Customization */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Slider Track */
    .stSlider > div > div > div > div {
        background-color: #f0f0f0;
    }
    
    /* Slider Handle */
    .stSlider > div > div > div > div > div {
        background-color: #666666;
        border: 2px solid #ffffff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Button Styles */
    .stButton > button {
        width: 100%;
        height: 3rem;
        margin: 1rem 0;
        background: #f5f5f5;
        color: #333333;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.025em;
        text-transform: uppercase;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button:hover {
        background: #ebebeb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Select Box Customization */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #666666;
    }
    
    /* Number Input Customization */
    .stNumberInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #666666;
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Help Text */
    .stMarkdown div.stMarkdown {
        color: #333333;
        font-size: 0.875rem;
        line-height: 1.25rem;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding: 1rem 0;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f8f8f8;
        border-radius: 8px;
        transition: all 0.2s ease;
        border: 1px solid #e0e0e0;
        color: #333333;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f0f0;
        border-color: #666666;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #f0f0f0;
        color: #000000;
        border: 1px solid #666666;
    }
    
    /* Slider Labels */
    .stSlider > div > div > div > div[data-testid="stTickBar"] > div {
        color: #333333;
        font-size: 0.875rem;
    }
    
    /* Help Icons */
    .stTooltipIcon {
        color: #666666;
    }
    
    .stTooltipIcon:hover {
        color: #333333;
    }
    
    /* Control Group Spacing */
    .control-group {
        margin-bottom: 1.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .control-group:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
    }
    
    /* Info Messages */
    .element-container:has(div.stAlert) {
        background-color: #f8f8f8;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #666666;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background-color: #f8f8f8;
        border: 2px dashed #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #666666;
        background-color: #f0f0f0;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: #f5f5f5;
        color: #333333;
    }
    
    .stDownloadButton > button:hover {
        background: #ebebeb;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    
    .stMultiSelect > div > div:hover {
        border-color: #666666;
    }
    
    /* Widget Labels */
    .stWidgetLabel {
        color: #333333;
        font-weight: 500;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🏥 Multimorbidity Analysis Tool")
    st.markdown("""
    This tool helps analyse disease trajectories and comorbidity patterns in patient data.
    Upload your data file to begin analysis.
    """)

    try:
        # File uploader in a container
        with st.container():
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                help="Upload a CSV file containing your patient data"
            )

        if uploaded_file is not None:
            try:
                # Calculate hash of uploaded file
                file_contents = uploaded_file.getvalue()
                current_hash = hash(file_contents)
                
                # Check if this is a new file
                if 'data_hash' not in st.session_state or current_hash != st.session_state.data_hash:
                    # Clear all session state variables
                    clear_session_state()
                    # Update the hash
                    st.session_state.data_hash = current_hash
                
                # Reset file pointer after reading
                uploaded_file.seek(0)
                
                # Load and process data
                data, total_patients, gender, age_group = load_and_process_data(uploaded_file)

                if data is None:
                    st.error("Failed to load data. Please check your file format.")
                    st.stop()

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
                    "🔍 Condition Combinations",
                    "👤 Personalised Analysis",
                    "🎯 Custom Trajectory Filter"
                ])

                # Sensitivity Analysis Tab
                with tabs[0]:
                    st.header("Sensitivity Analysis")
                    st.markdown("""
                    Explore how different odds ratio thresholds affect the number of disease
                    trajectories and population coverage.
                    """)

                    main_col, control_col = st.columns([3, 1])

                    with control_col:
                        with st.container():
                            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
                            st.markdown("### Control Panel")
                            top_n = st.slider(
                                "Number of Top Trajectories",
                                min_value=1,
                                max_value=20,
                                value=st.session_state.top_n_trajectories,
                                step=1,
                                help="Select how many top trajectories to display"
                            )
                            st.session_state.top_n_trajectories = top_n
                            
                            analyse_button = st.button(
                                "🚀 Run Analysis",
                                key="run_sensitivity",
                                help="Click to perform sensitivity analysis"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with main_col:
                        try:
                            if analyse_button:
                                with st.spinner("💫 Analysing data..."):
                                    # Clear previous results
                                    st.session_state.sensitivity_results = None
                                    
                                    # Generate new results with selected top_n
                                    results = perform_sensitivity_analysis(data, top_n=top_n)
                                    st.session_state.sensitivity_results = results

                                    display_df = results.drop('Top_Patterns', axis=1)
                                    st.subheader("Analysis Results")
                                    st.dataframe(
                                        display_df.style.background_gradient(cmap='YlOrRd', subset=['Coverage_Percent'])
                                    )

                                    st.subheader(f"Top {top_n} Strongest Trajectories")
                                    patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                                    st.dataframe(
                                        patterns_df.style.background_gradient(cmap='YlOrRd', subset=['OddsRatio'])
                                    )

                                    fig = create_sensitivity_plot(results)
                                    st.pyplot(fig)

                                    csv = display_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results",
                                        data=csv,
                                        file_name="sensitivity_analysis_results.csv",
                                        mime="text/csv"
                                    )
                            
                            # Display existing results if available
                            elif st.session_state.sensitivity_results is not None:
                                results = st.session_state.sensitivity_results
                                display_df = results.drop('Top_Patterns', axis=1)
                                
                                st.subheader("Analysis Results")
                                st.dataframe(
                                    display_df.style.background_gradient(cmap='YlOrRd', subset=['Coverage_Percent'])
                                )

                                num_patterns = len(results.iloc[0]['Top_Patterns'])
                                st.subheader(f"Top {num_patterns} Strongest Trajectories")
                                patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                                st.dataframe(
                                    patterns_df.style.background_gradient(cmap='YlOrRd', subset=['OddsRatio'])
                                )

                                fig = create_sensitivity_plot(results)
                                st.pyplot(fig)

                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name="sensitivity_analysis_results.csv",
                                    mime="text/csv"
                                )

                        except Exception as e:
                            st.error(f"Error in sensitivity analysis: {str(e)}")
                            st.session_state.sensitivity_results = None

                # Condition Combinations Tab
                with tabs[1]:
                    st.header("Condition Combinations Analysis")
                    
                    main_col, control_col = st.columns([3, 1])

                    with control_col:
                        with st.container():
                            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
                            st.markdown("### Control Panel")
                            try:
                                min_freq_range = (data['PairFrequency'].min(), data['PairFrequency'].max())
                                min_frequency = st.slider(
                                    "Minimum Pair Frequency",
                                    int(min_freq_range[0]),
                                    int(min_freq_range[1]),
                                    int(min_freq_range[0]) if st.session_state.min_frequency is None
                                    else st.session_state.min_frequency,
                                    help="Minimum number of occurrences required"
                                )
                                st.session_state.min_frequency = min_frequency

                                min_percentage_range = (data['Percentage'].min(), data['Percentage'].max())
                                min_percentage = st.slider(
                                    "Minimum Prevalence (%)",
                                    float(min_percentage_range[0]),
                                    float(min_percentage_range[1]),
                                    float(min_percentage_range[0]) if st.session_state.min_percentage is None
                                    else st.session_state.min_percentage,
                                    0.1,
                                    help="Minimum percentage of population affected"
                                )
                                st.session_state.min_percentage = min_percentage

                                analyse_combinations_button = st.button(
                                    "🔍 Analyse Combinations",
                                    help="Click to analyse condition combinations"
                                )
                            except Exception as e:
                                st.error(f"Error setting up analysis parameters: {str(e)}")
                            st.markdown('</div>', unsafe_allow_html=True)

                    with main_col:
                        try:
                            if analyse_combinations_button:
                                with st.spinner("🔄 Analysing combinations..."):
                                    # Clear previous results
                                    st.session_state.combinations_results = None
                                    st.session_state.combinations_fig = None
                                    
                                    # Generate new results
                                    results_df = analyze_condition_combinations(
                                        data,
                                        min_percentage,
                                        min_frequency
                                    )

                                    if not results_df.empty:
                                        st.session_state.combinations_results = results_df
                                        
                                        st.subheader(f"Analysis Results ({len(results_df)} combinations)")
                                        st.dataframe(
                                            results_df.style.background_gradient(
                                                cmap='YlOrRd',
                                                subset=['Prevalence of the combination (%)']
                                            )
                                        )

                                        fig = create_combinations_plot(results_df)
                                        st.session_state.combinations_fig = fig
                                        st.pyplot(fig)

                                        csv = results_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Results",
                                            data=csv,
                                            file_name="condition_combinations.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.warning("No combinations found matching the criteria. Try adjusting the parameters.")
                            
                            # Display existing results if available
                            elif st.session_state.combinations_results is not None:
                                results_df = st.session_state.combinations_results
                                st.subheader(f"Analysis Results ({len(results_df)} combinations)")
                                st.dataframe(
                                    results_df.style.background_gradient(
                                        cmap='YlOrRd',
                                        subset=['Prevalence of the combination (%)']
                                    )
                                )

                                if st.session_state.combinations_fig is not None:
                                    st.pyplot(st.session_state.combinations_fig)

                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name="condition_combinations.csv",
                                    mime="text/csv"
                                )

                        except Exception as e:
                            st.error(f"Error in combinations analysis: {str(e)}")
                            st.session_state.combinations_results = None
                            st.session_state.combinations_fig = None

                # Personalised Analysis Tab
                with tabs[2]:
                    st.header("Personalised Trajectory Analysis")
                    st.markdown("""
                    Analyse potential disease progressions based on a patient's current conditions,
                    considering population-level statistics and time-based progression patterns.
                    """)

                    main_col, control_col = st.columns([3, 1])

                    with control_col:
                        with st.container():
                            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
                            st.markdown("### Control Panel")
                            
                            min_or = st.slider(
                                "Minimum Odds Ratio",
                                1.0, 10.0, st.session_state.min_or, 0.5,
                                key="personal_min_or",
                                help="Filter trajectories by minimum odds ratio"
                            )
                            st.session_state.min_or = min_or

                            max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]).max())
                            time_horizon = st.slider(
                                "Time Horizon (years)",
                                1, max_years, st.session_state.time_horizon,
                                key="personal_time_horizon",
                                help="Maximum time period to consider"
                            )
                            st.session_state.time_horizon = time_horizon

                            time_margin = st.slider(
                                "Time Margin",
                                0.0, 0.5, st.session_state.time_margin, 0.05,
                                key="personal_time_margin",
                                help="Allowable variation in time predictions"
                            )
                            st.session_state.time_margin = time_margin

                            analyse_button = st.button(
                                "🔍 Analyse Trajectories",
                                key="personal_analyse",
                                help="Generate personalised analysis"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with main_col:
                        st.markdown("""
                        <h4 style='font-size: 1.2em; font-weight: 600; color: #333; margin-bottom: 10px;'>
                            🔍 Please select all conditions that the patient currently has:
                        </h4>
                        """, unsafe_allow_html=True)
                        
                        unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
                        selected_conditions = st.multiselect(
                            "Select Current Conditions",
                            unique_conditions,
                            default=st.session_state.selected_conditions,
                            key="personal_select",
                            help="Choose all conditions that the patient currently has"
                        )
                        st.session_state.selected_conditions = selected_conditions

                        if selected_conditions and analyse_button:
                            with st.spinner("🔄 Generating personalised analysis..."):
                                # Clear previous results
                                st.session_state.personalized_html = None
                                
                                # Generate new analysis
                                html_content = create_personalized_analysis(
                                    data,
                                    selected_conditions,
                                    time_horizon,
                                    time_margin,
                                    min_or
                                )
                                st.session_state.personalized_html = html_content

                                html_container = f"""
                                <div style="min-height: 800px; width: 100%; padding: 20px;">
                                    {html_content}
                                </div>
                                """
                                st.components.v1.html(html_container, height=1200, scrolling=True)
                                st.download_button(
                                    label="📥 Download Analysis",
                                    data=html_content,
                                    file_name="personalised_trajectory_analysis.html",
                                    mime="text/html"
                                )

                        # Display existing analysis if available
                        elif st.session_state.personalized_html is not None:
                            html_container = f"""
                            <div style="min-height: 800px; width: 100%; padding: 20px;">
                                {st.session_state.personalized_html}
                            </div>
                            """
                            st.components.v1.html(html_container, height=1200, scrolling=True)
                            st.download_button(
                                label="📥 Download Analysis",
                                data=st.session_state.personalized_html,
                                file_name="personalised_trajectory_analysis.html",
                                mime="text/html"
                            )

                # Custom Trajectory Filter Tab
                with tabs[3]:
                    st.header("Custom Trajectory Filter")
                    st.markdown("""
                    Visualise disease trajectories based on custom odds ratio and frequency thresholds.
                    Select conditions and adjust filters to explore different trajectory patterns.
                    """)

                    main_col, control_col = st.columns([3, 1])

                    with control_col:
                        with st.container():
                            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
                            st.markdown("### Control Panel")
                            try:
                                min_or = st.slider(
                                    "Minimum Odds Ratio",
                                    1.0, 10.0, st.session_state.min_or, 0.5,
                                    key="custom_min_or",
                                    help="Filter trajectories by minimum odds ratio"
                                )

                                min_freq = st.slider(
                                    "Minimum Frequency",
                                    int(data['PairFrequency'].min()),
                                    int(data['PairFrequency'].max()),
                                    int(data['PairFrequency'].min()),
                                    help="Minimum number of occurrences required"
                                )

                                # Filter data based on both OR and frequency
                                filtered_data = data[
                                    (data['OddsRatio'] >= min_or) &
                                    (data['PairFrequency'] >= min_freq)
                                ]

                                # Get conditions from filtered data
                                unique_conditions = sorted(set(
                                    filtered_data['ConditionA'].unique()) |
                                    set(filtered_data['ConditionB'].unique())
                                )

                                selected_conditions = st.multiselect(
                                    "Select Initial Conditions",
                                    unique_conditions,
                                    default=st.session_state.selected_conditions,
                                    key="custom_select",
                                    help="Choose the starting conditions for trajectory analysis"
                                )
                                st.session_state.selected_conditions = selected_conditions

                                if selected_conditions:
                                    max_years = math.ceil(filtered_data['MedianDurationYearsWithIQR']
                                                        .apply(lambda x: parse_iqr(x)[0]).max())
                                    time_horizon = st.slider(
                                        "Time Horizon (years)",
                                        1, max_years, st.session_state.time_horizon,
                                        key="custom_time_horizon",
                                        help="Maximum time period to consider"
                                    )

                                    time_margin = st.slider(
                                        "Time Margin",
                                        0.0, 0.5, st.session_state.time_margin, 0.05,
                                        key="custom_time_margin",
                                        help="Allowable variation in time predictions"
                                    )

                                    generate_button = st.button(
                                        "🔄 Generate Network",
                                        key="custom_generate",
                                        help="Click to generate trajectory network"
                                    )

                            except Exception as e:
                                st.error(f"Error in custom trajectory analysis: {str(e)}")
                            st.markdown('</div>', unsafe_allow_html=True)

                    with main_col:
                        if selected_conditions and generate_button:
                            with st.spinner("🌐 Generating network..."):
                                try:
                                    # Clear previous network
                                    st.session_state.network_html = None
                                    
                                    # Generate new network
                                    html_content = create_network_graph(
                                        filtered_data,
                                        selected_conditions,
                                        min_or,
                                        time_horizon,
                                        time_margin
                                    )
                                    st.session_state.network_html = html_content
                                    st.components.v1.html(html_content, height=800)

                                    st.download_button(
                                        label="📥 Download Network",
                                        data=html_content,
                                        file_name="custom_trajectory_network.html",
                                        mime="text/html"
                                    )
                                except Exception as viz_error:
                                    st.error(f"Failed to generate network visualisation: {str(viz_error)}")
                                    st.session_state.network_html = None
                        
                        # Display existing network if available
                        elif st.session_state.network_html is not None:
                            st.components.v1.html(st.session_state.network_html, height=800)
                            st.download_button(
                                label="📥 Download Network",
                                data=st.session_state.network_html,
                                file_name="custom_trajectory_network.html",
                                mime="text/html"
                            )

            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.stop()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
                                
