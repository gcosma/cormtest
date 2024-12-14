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

Select conditions and adjust filters to explore different trajectory patterns.
                """)

                viz_col, param_col = st.columns([3, 1])

                with param_col:
                    st.markdown("### Parameters")
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

                with viz_col:
                    if selected_conditions and generate_button:
                        with st.spinner("🌐 Generating network..."):
                            try:
                                # Use filtered data instead of original data
                                html_content = create_network_graph(
                                    filtered_data,  # Use filtered data here
                                    selected_conditions,
                                    min_or,
                                    time_horizon,
                                    time_margin
                                )
                                st.components.v1.html(html_content, height=800)

                                st.download_button(
                                    label="📥 Download Network",
                                    data=html_content,
                                    file_name="custom_trajectory_network.html",
                                    mime="text/html"
                                )
                            except Exception as e:
                                st.error(f"Failed to generate network: {str(e)}")


if __name__ == "__main__":
    main()
