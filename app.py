# Save this code as app.py and run with `streamlit run app.py`

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Model Performance Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --- Title and Introduction ---
st.title("📊 AI Model Performance Dashboard")
st.markdown("""
This dashboard provides an interactive analysis of six AI models based on the `ComparIA_Benchmark` dataset.
Explore the trade-offs between answer quality, cost, energy consumption (CO2), and latency to inform strategic model selection.
""")

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the benchmark data."""
    try:
        # Try to load as Excel file first
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Standardize column names
        df.columns = [
            'Prompt', 'Model', 'Quality', 'CO2_Emission_kg', 
            'Latency_s', 'Cost'
        ]
        
        # Clean data: convert to numeric, coercing errors
        # Handle comma decimal separator (European format)
        for col in ['Quality', 'CO2_Emission_kg', 'Latency_s', 'Cost']:
            if df[col].dtype == 'object':
                # Replace comma with dot for decimal separator
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where critical data is missing
        df.dropna(subset=['Quality', 'CO2_Emission_kg', 'Latency_s', 'Cost'], inplace=True)
        
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file '{file_path}' was not found. Please make sure it's in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load the data
DATA_FILE = 'FComparIA_Benchmark_LongFormat.xlsx'
df = load_data(DATA_FILE)

if df.empty:
    st.stop()

# --- Calculate Model Statistics ---
@st.cache_data
def calculate_model_stats(df):
    """Calculate aggregate statistics for each model."""
    stats = df.groupby('Model').agg({
        'Quality': ['mean', 'std', 'min', 'max'],
        'CO2_Emission_kg': ['mean', 'std'],
        'Latency_s': ['mean', 'std', 'median'],
        'Cost': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in stats.columns.values]
    
    # Calculate efficiency scores (normalized inverse for "lower is better" metrics)
    stats['Quality_Score'] = stats['Quality_mean']
    stats['CO2_Score'] = 1 / (stats['CO2_Emission_kg_mean'] + 0.01)  # Add small constant to avoid division by zero
    stats['Cost_Score'] = 1 / (stats['Cost_mean'] + 0.01)
    stats['Speed_Score'] = 1 / (stats['Latency_s_mean'] + 0.01)
    
    # Overall efficiency score (weighted average)
    stats['Overall_Score'] = (
        stats['Quality_Score'] * 0.35 + 
        stats['CO2_Score'] * 0.25 + 
        stats['Cost_Score'] * 0.25 + 
        stats['Speed_Score'] * 0.15
    )
    
    return stats

model_stats = calculate_model_stats(df)

# --- Sidebar Filters ---
st.sidebar.header("Filters")
selected_models = st.sidebar.multiselect(
    "Select Models to Display:",
    options=df['Model'].unique(),
    default=df['Model'].unique()
)

# Filter dataframe based on selection
df_filtered = df[df['Model'].isin(selected_models)]
model_stats_filtered = model_stats[model_stats['Model'].isin(selected_models)]

if df_filtered.empty:
    st.warning("No models selected or no data available for the selected models. Please select at least one model from the sidebar.")
    st.stop()

# --- Key Metrics Overview ---
st.header("📈 Key Performance Metrics")
metric_cols = st.columns(4)

with metric_cols[0]:
    best_quality_model = model_stats_filtered.loc[model_stats_filtered['Quality_mean'].idxmax(), 'Model']
    best_quality_value = model_stats_filtered['Quality_mean'].max()
    st.metric("🏆 Highest Quality", best_quality_model, f"{best_quality_value:.2f}/5")

with metric_cols[1]:
    best_eco_model = model_stats_filtered.loc[model_stats_filtered['CO2_Emission_kg_mean'].idxmin(), 'Model']
    best_eco_value = model_stats_filtered['CO2_Emission_kg_mean'].min()
    st.metric("🌱 Lowest CO2", best_eco_model, f"{best_eco_value:.2f} kg")

with metric_cols[2]:
    best_cost_model = model_stats_filtered.loc[model_stats_filtered['Cost_mean'].idxmin(), 'Model']
    best_cost_value = model_stats_filtered['Cost_mean'].min()
    st.metric("💰 Lowest Cost", best_cost_model, f"{best_cost_value:.2f}")

with metric_cols[3]:
    best_speed_model = model_stats_filtered.loc[model_stats_filtered['Latency_s_mean'].idxmin(), 'Model']
    best_speed_value = model_stats_filtered['Latency_s_mean'].min()
    st.metric("⚡ Fastest", best_speed_model, f"{best_speed_value:.2f}s")

# --- Main Dashboard Layout ---
st.header("Comparative Analysis Visualizations")

# Create columns for the charts
col1, col2 = st.columns(2)

# --- Chart 1: Quality vs. Energy (CO2 Emission) ---
with col1:
    st.subheader("Quality vs. Energy Efficiency")
    fig_energy = px.scatter(
        df_filtered,
        x='CO2_Emission_kg',
        y='Quality',
        color='Model',
        size='Cost',
        hover_name='Model',
        hover_data={
            'Prompt': True,
            'Latency_s': ':.2f',
            'CO2_Emission_kg': ':.2f',
            'Cost': ':.2f',
            'Quality': True
        },
        title="Answer Quality vs. CO2 Emission",
        labels={
            'CO2_Emission_kg': 'CO2 Emission (kg eq.)',
            'Quality': 'Answer Quality (1-5)'
        }
    )
    fig_energy.update_layout(
        legend_title_text='Model',
        xaxis_title="CO2 Emission (kg eq.) → Lower is Better",
        yaxis_title="Answer Quality (1-5) → Higher is Better",
        hovermode='closest'
    )
    fig_energy.update_traces(marker=dict(sizemin=5, line=dict(width=1, color='white')))
    st.plotly_chart(fig_energy, use_container_width=True)
    st.info("💡 **Analysis:** The ideal models are in the **top-left corner** (high quality, low CO2). Bubble size represents cost.")

# --- Chart 2: Quality vs. Cost ---
with col2:
    st.subheader("Quality vs. Cost-Effectiveness")
    fig_cost = px.scatter(
        df_filtered,
        x='Cost',
        y='Quality',
        color='Model',
        size='CO2_Emission_kg',
        hover_name='Model',
        hover_data={
            'Prompt': True,
            'Latency_s': ':.2f',
            'CO2_Emission_kg': ':.2f',
            'Cost': ':.2f',
            'Quality': True
        },
        title="Answer Quality vs. Cost per Task",
        labels={
            'Cost': 'Cost per Task (Wh or €)',
            'Quality': 'Answer Quality (1-5)'
        }
    )
    fig_cost.update_layout(
        legend_title_text='Model',
        xaxis_title="Cost per Task → Lower is Better",
        yaxis_title="Answer Quality (1-5) → Higher is Better",
        hovermode='closest'
    )
    fig_cost.update_traces(marker=dict(sizemin=5, line=dict(width=1, color='white')))
    st.plotly_chart(fig_cost, use_container_width=True)
    st.info("💡 **Analysis:** The ideal models are in the **top-left corner** (high quality, low cost). Bubble size represents CO2 impact.")

# --- Chart 3: Multi-dimensional Radar Chart ---
st.subheader("Multi-Dimensional Performance Comparison")

# Normalize metrics for radar chart (0-1 scale)
radar_stats = model_stats_filtered.copy()
radar_stats['Quality_Norm'] = radar_stats['Quality_mean'] / 5
radar_stats['CO2_Norm'] = 1 - (radar_stats['CO2_Emission_kg_mean'] - radar_stats['CO2_Emission_kg_mean'].min()) / (radar_stats['CO2_Emission_kg_mean'].max() - radar_stats['CO2_Emission_kg_mean'].min() + 0.01)
radar_stats['Cost_Norm'] = 1 - (radar_stats['Cost_mean'] - radar_stats['Cost_mean'].min()) / (radar_stats['Cost_mean'].max() - radar_stats['Cost_mean'].min() + 0.01)
radar_stats['Speed_Norm'] = 1 - (radar_stats['Latency_s_mean'] - radar_stats['Latency_s_mean'].min()) / (radar_stats['Latency_s_mean'].max() - radar_stats['Latency_s_mean'].min() + 0.01)

fig_radar = go.Figure()

categories = ['Quality', 'Eco-Friendly<br>(Low CO2)', 'Cost-Effective', 'Speed']

for _, row in radar_stats.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row['Quality_Norm'], row['CO2_Norm'], row['Cost_Norm'], row['Speed_Norm']],
        theta=categories,
        fill='toself',
        name=row['Model']
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title="Normalized Performance Across All Dimensions (1 = Best)"
)

st.plotly_chart(fig_radar, use_container_width=True)
st.info("💡 **Analysis:** Models with larger, more circular shapes perform well across all dimensions. Look for the model that best matches your priority areas.")

# --- Chart 4: Latency Comparison ---
st.subheader("Latency (Inference Timing) Comparison")
latency_stats = model_stats_filtered[['Model', 'Latency_s_mean', 'Latency_s_std', 'Latency_s_median']].copy()
latency_stats = latency_stats.sort_values('Latency_s_median', ascending=True)

fig_latency = px.bar(
    latency_stats,
    x='Model',
    y='Latency_s_mean',
    error_y='Latency_s_std',
    color='Model',
    labels={'Latency_s_mean': 'Average Latency (s)', 'Model': 'Model'},
    title='Average Latency with Standard Deviation',
    hover_data={'Latency_s_median': ':.2f', 'Latency_s_std': ':.2f'}
)
fig_latency.update_layout(
    yaxis_title="Average Latency (s) → Lower is Better",
    xaxis_title="Model",
    showlegend=False
)
st.plotly_chart(fig_latency, use_container_width=True)
st.info("💡 **Analysis:** Lower bars are better. Error bars show variability—smaller bars indicate more consistent performance.")

# --- Model Rankings Table ---
st.header("🏆 Model Rankings by Category")

ranking_data = []

# Best Overall
best_overall = model_stats_filtered.nlargest(2, 'Overall_Score')
ranking_data.append({
    'Category': 'Best Overall Trade-Off',
    'Winner': best_overall.iloc[0]['Model'],
    'Runner-Up': best_overall.iloc[1]['Model'] if len(best_overall) > 1 else 'N/A',
    'Key Metric': f"Quality: {best_overall.iloc[0]['Quality_mean']:.2f}, CO2: {best_overall.iloc[0]['CO2_Emission_kg_mean']:.2f}kg"
})

# Lowest CO2
best_co2 = model_stats_filtered.nsmallest(2, 'CO2_Emission_kg_mean')
ranking_data.append({
    'Category': 'Lowest CO2 Emission',
    'Winner': best_co2.iloc[0]['Model'],
    'Runner-Up': best_co2.iloc[1]['Model'] if len(best_co2) > 1 else 'N/A',
    'Key Metric': f"{best_co2.iloc[0]['CO2_Emission_kg_mean']:.2f} kg eq. (Quality: {best_co2.iloc[0]['Quality_mean']:.2f})"
})

# Lowest Cost
best_cost = model_stats_filtered.nsmallest(2, 'Cost_mean')
ranking_data.append({
    'Category': 'Lowest Cost',
    'Winner': best_cost.iloc[0]['Model'],
    'Runner-Up': best_cost.iloc[1]['Model'] if len(best_cost) > 1 else 'N/A',
    'Key Metric': f"{best_cost.iloc[0]['Cost_mean']:.2f} (Quality: {best_cost.iloc[0]['Quality_mean']:.2f})"
})

# Fastest
best_speed = model_stats_filtered.nsmallest(2, 'Latency_s_mean')
ranking_data.append({
    'Category': 'Fastest Latency',
    'Winner': best_speed.iloc[0]['Model'],
    'Runner-Up': best_speed.iloc[1]['Model'] if len(best_speed) > 1 else 'N/A',
    'Key Metric': f"{best_speed.iloc[0]['Latency_s_mean']:.2f}s (Quality: {best_speed.iloc[0]['Quality_mean']:.2f})"
})

# Highest Quality
best_quality = model_stats_filtered.nlargest(2, 'Quality_mean')
ranking_data.append({
    'Category': 'Highest Quality',
    'Winner': best_quality.iloc[0]['Model'],
    'Runner-Up': best_quality.iloc[1]['Model'] if len(best_quality) > 1 else 'N/A',
    'Key Metric': f"{best_quality.iloc[0]['Quality_mean']:.2f}/5 (Cost: {best_quality.iloc[0]['Cost_mean']:.2f})"
})

ranking_df = pd.DataFrame(ranking_data)
st.table(ranking_df)

# --- Detailed Model Statistics ---
st.header("📊 Detailed Model Statistics")
display_stats = model_stats_filtered[['Model', 'Quality_mean', 'CO2_Emission_kg_mean', 'Cost_mean', 'Latency_s_mean']].copy()
display_stats.columns = ['Model', 'Avg Quality', 'Avg CO2 (kg)', 'Avg Cost', 'Avg Latency (s)']
display_stats = display_stats.round(2)
st.dataframe(display_stats, use_container_width=True)

# --- Raw Data Viewer ---
with st.expander("View Raw Filtered Data"):
    st.dataframe(df_filtered)