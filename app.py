# # Save this code as app.py and run with `streamlit run app.py`

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import numpy as np

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="AI Model Performance Dashboard",
#     page_icon="🤖",
#     layout="wide"
# )

# # --- Title and Introduction ---
# st.title("📊 AI Model Performance Dashboard")
# st.markdown("""
# This dashboard provides an interactive analysis of six AI models based on the `ComparIA_Benchmark` dataset.
# Explore the trade-offs between answer quality, cost, energy consumption (CO2), and latency to inform strategic model selection.
# """)

# # --- Data Loading and Caching ---
# @st.cache_data
# def load_data(file_path):
#     """Loads and preprocesses the benchmark data."""
#     try:
#         # Try to load as Excel file first
#         if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
#             df = pd.read_excel(file_path)
#         else:
#             df = pd.read_csv(file_path)
        
#         # Standardize column names
#         df.columns = [
#             'Prompt', 'Model', 'Quality', 'CO2_Emission_kg', 
#             'Latency_s', 'Cost'
#         ]
        
#         # Clean data: convert to numeric, coercing errors
#         # Handle comma decimal separator (European format)
#         for col in ['Quality', 'CO2_Emission_kg', 'Latency_s', 'Cost']:
#             if df[col].dtype == 'object':
#                 # Replace comma with dot for decimal separator
#                 df[col] = df[col].astype(str).str.replace(',', '.')
#             df[col] = pd.to_numeric(df[col], errors='coerce')
        
#         # Drop rows where critical data is missing
#         df.dropna(subset=['Quality', 'CO2_Emission_kg', 'Latency_s', 'Cost'], inplace=True)
        
#         return df
#     except FileNotFoundError:
#         st.error(f"Error: The data file '{file_path}' was not found. Please make sure it's in the same directory.")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         return pd.DataFrame()

# # Load the data
# DATA_FILE = 'FinalComparIA_Benchmark_LongFormat.xlsx'
# df = load_data(DATA_FILE)

# if df.empty:
#     st.stop()

# # --- Calculate Model Statistics ---
# @st.cache_data
# def calculate_model_stats(df):
#     """Calculate aggregate statistics for each model."""
#     stats = df.groupby('Model').agg({
#         'Quality': ['mean', 'std', 'min', 'max'],
#         'CO2_Emission_kg': ['mean', 'std'],
#         'Latency_s': ['mean', 'std', 'median'],
#         'Cost': ['mean', 'std']
#     }).reset_index()
    
#     # Flatten column names
#     stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in stats.columns.values]
    
#     # Calculate efficiency scores (normalized inverse for "lower is better" metrics)
#     stats['Quality_Score'] = stats['Quality_mean']
#     stats['CO2_Score'] = 1 / (stats['CO2_Emission_kg_mean'] + 0.01)  # Add small constant to avoid division by zero
#     stats['Cost_Score'] = 1 / (stats['Cost_mean'] + 0.01)
#     stats['Speed_Score'] = 1 / (stats['Latency_s_mean'] + 0.01)
    
#     # Overall efficiency score (weighted average)
#     stats['Overall_Score'] = (
#         stats['Quality_Score'] * 0.35 + 
#         stats['CO2_Score'] * 0.25 + 
#         stats['Cost_Score'] * 0.25 + 
#         stats['Speed_Score'] * 0.15
#     )
    
#     return stats

# model_stats = calculate_model_stats(df)

# # --- Sidebar Filters ---
# st.sidebar.header("Filters")
# selected_models = st.sidebar.multiselect(
#     "Select Models to Display:",
#     options=df['Model'].unique(),
#     default=df['Model'].unique()
# )

# # Filter dataframe based on selection
# df_filtered = df[df['Model'].isin(selected_models)]
# model_stats_filtered = model_stats[model_stats['Model'].isin(selected_models)]

# if df_filtered.empty:
#     st.warning("No models selected or no data available for the selected models. Please select at least one model from the sidebar.")
#     st.stop()

# # --- Key Metrics Overview ---
# st.header("📈 Key Performance Metrics")
# metric_cols = st.columns(4)

# with metric_cols[0]:
#     best_quality_model = model_stats_filtered.loc[model_stats_filtered['Quality_mean'].idxmax(), 'Model']
#     best_quality_value = model_stats_filtered['Quality_mean'].max()
#     st.metric("🏆 Highest Quality", best_quality_model, f"{best_quality_value:.2f}/5")

# with metric_cols[1]:
#     best_eco_model = model_stats_filtered.loc[model_stats_filtered['CO2_Emission_kg_mean'].idxmin(), 'Model']
#     best_eco_value = model_stats_filtered['CO2_Emission_kg_mean'].min()
#     st.metric("🌱 Lowest CO2", best_eco_model, f"{best_eco_value:.2f} kg")

# with metric_cols[2]:
#     best_cost_model = model_stats_filtered.loc[model_stats_filtered['Cost_mean'].idxmin(), 'Model']
#     best_cost_value = model_stats_filtered['Cost_mean'].min()
#     st.metric("💰 Lowest Cost", best_cost_model, f"{best_cost_value:.2f}")

# with metric_cols[3]:
#     best_speed_model = model_stats_filtered.loc[model_stats_filtered['Latency_s_mean'].idxmin(), 'Model']
#     best_speed_value = model_stats_filtered['Latency_s_mean'].min()
#     st.metric("⚡ Fastest", best_speed_model, f"{best_speed_value:.2f}s")

# # --- Main Dashboard Layout ---
# st.header("Comparative Analysis Visualizations")

# # Create columns for the charts
# col1, col2 = st.columns(2)

# # --- Chart 1: Quality vs. Energy (CO2 Emission) ---
# with col1:
#     st.subheader("Quality vs. Energy Efficiency")
#     fig_energy = px.scatter(
#         df_filtered,
#         x='CO2_Emission_kg',
#         y='Quality',
#         color='Model',
#         size='Cost',
#         hover_name='Model',
#         hover_data={
#             'Prompt': True,
#             'Latency_s': ':.2f',
#             'CO2_Emission_kg': ':.2f',
#             'Cost': ':.2f',
#             'Quality': True
#         },
#         title="Answer Quality vs. CO2 Emission",
#         labels={
#             'CO2_Emission_kg': 'CO2 Emission (kg eq.)',
#             'Quality': 'Answer Quality (1-5)'
#         }
#     )
#     fig_energy.update_layout(
#         legend_title_text='Model',
#         xaxis_title="CO2 Emission (kg eq.) → Lower is Better",
#         yaxis_title="Answer Quality (1-5) → Higher is Better",
#         hovermode='closest'
#     )
#     fig_energy.update_traces(marker=dict(sizemin=5, line=dict(width=1, color='white')))
#     st.plotly_chart(fig_energy, use_container_width=True)
#     st.info("💡 **Analysis:** The ideal models are in the **top-left corner** (high quality, low CO2). Bubble size represents cost.")

# # --- Chart 2: Quality vs. Cost ---
# with col2:
#     st.subheader("Quality vs. Cost-Effectiveness")
#     fig_cost = px.scatter(
#         df_filtered,
#         x='Cost',
#         y='Quality',
#         color='Model',
#         size='CO2_Emission_kg',
#         hover_name='Model',
#         hover_data={
#             'Prompt': True,
#             'Latency_s': ':.2f',
#             'CO2_Emission_kg': ':.2f',
#             'Cost': ':.2f',
#             'Quality': True
#         },
#         title="Answer Quality vs. Cost per Task",
#         labels={
#             'Cost': 'Cost per Task (Wh or €)',
#             'Quality': 'Answer Quality (1-5)'
#         }
#     )
#     fig_cost.update_layout(
#         legend_title_text='Model',
#         xaxis_title="Cost per Task → Lower is Better",
#         yaxis_title="Answer Quality (1-5) → Higher is Better",
#         hovermode='closest'
#     )
#     fig_cost.update_traces(marker=dict(sizemin=5, line=dict(width=1, color='white')))
#     st.plotly_chart(fig_cost, use_container_width=True)
#     st.info("💡 **Analysis:** The ideal models are in the **top-left corner** (high quality, low cost). Bubble size represents CO2 impact.")

# # --- Chart 3: Multi-dimensional Radar Chart ---
# st.subheader("Multi-Dimensional Performance Comparison")

# # Normalize metrics for radar chart (0-1 scale)
# radar_stats = model_stats_filtered.copy()
# radar_stats['Quality_Norm'] = radar_stats['Quality_mean'] / 5
# radar_stats['CO2_Norm'] = 1 - (radar_stats['CO2_Emission_kg_mean'] - radar_stats['CO2_Emission_kg_mean'].min()) / (radar_stats['CO2_Emission_kg_mean'].max() - radar_stats['CO2_Emission_kg_mean'].min() + 0.01)
# radar_stats['Cost_Norm'] = 1 - (radar_stats['Cost_mean'] - radar_stats['Cost_mean'].min()) / (radar_stats['Cost_mean'].max() - radar_stats['Cost_mean'].min() + 0.01)
# radar_stats['Speed_Norm'] = 1 - (radar_stats['Latency_s_mean'] - radar_stats['Latency_s_mean'].min()) / (radar_stats['Latency_s_mean'].max() - radar_stats['Latency_s_mean'].min() + 0.01)

# fig_radar = go.Figure()

# categories = ['Quality', 'Eco-Friendly<br>(Low CO2)', 'Cost-Effective', 'Speed']

# for _, row in radar_stats.iterrows():
#     fig_radar.add_trace(go.Scatterpolar(
#         r=[row['Quality_Norm'], row['CO2_Norm'], row['Cost_Norm'], row['Speed_Norm']],
#         theta=categories,
#         fill='toself',
#         name=row['Model']
#     ))

# fig_radar.update_layout(
#     polar=dict(
#         radialaxis=dict(
#             visible=True,
#             range=[0, 1]
#         )),
#     showlegend=True,
#     title="Normalized Performance Across All Dimensions (1 = Best)"
# )

# st.plotly_chart(fig_radar, use_container_width=True)
# st.info("💡 **Analysis:** Models with larger, more circular shapes perform well across all dimensions. Look for the model that best matches your priority areas.")

# # --- Chart 4: Latency Comparison ---
# st.subheader("Latency (Inference Timing) Comparison")
# latency_stats = model_stats_filtered[['Model', 'Latency_s_mean', 'Latency_s_std', 'Latency_s_median']].copy()
# latency_stats = latency_stats.sort_values('Latency_s_median', ascending=True)

# fig_latency = px.bar(
#     latency_stats,
#     x='Model',
#     y='Latency_s_mean',
#     error_y='Latency_s_std',
#     color='Model',
#     labels={'Latency_s_mean': 'Average Latency (s)', 'Model': 'Model'},
#     title='Average Latency with Standard Deviation',
#     hover_data={'Latency_s_median': ':.2f', 'Latency_s_std': ':.2f'}
# )
# fig_latency.update_layout(
#     yaxis_title="Average Latency (s) → Lower is Better",
#     xaxis_title="Model",
#     showlegend=False
# )
# st.plotly_chart(fig_latency, use_container_width=True)
# st.info("💡 **Analysis:** Lower bars are better. Error bars show variability—smaller bars indicate more consistent performance.")

# # --- Model Rankings Table ---
# st.header("🏆 Model Rankings by Category")

# ranking_data = []

# # Best Overall
# best_overall = model_stats_filtered.nlargest(2, 'Overall_Score')
# ranking_data.append({
#     'Category': 'Best Overall Trade-Off',
#     'Winner': best_overall.iloc[0]['Model'],
#     'Runner-Up': best_overall.iloc[1]['Model'] if len(best_overall) > 1 else 'N/A',
#     'Key Metric': f"Quality: {best_overall.iloc[0]['Quality_mean']:.2f}, CO2: {best_overall.iloc[0]['CO2_Emission_kg_mean']:.2f}kg"
# })

# # Lowest CO2
# best_co2 = model_stats_filtered.nsmallest(2, 'CO2_Emission_kg_mean')
# ranking_data.append({
#     'Category': 'Lowest CO2 Emission',
#     'Winner': best_co2.iloc[0]['Model'],
#     'Runner-Up': best_co2.iloc[1]['Model'] if len(best_co2) > 1 else 'N/A',
#     'Key Metric': f"{best_co2.iloc[0]['CO2_Emission_kg_mean']:.2f} kg eq. (Quality: {best_co2.iloc[0]['Quality_mean']:.2f})"
# })

# # Lowest Cost
# best_cost = model_stats_filtered.nsmallest(2, 'Cost_mean')
# ranking_data.append({
#     'Category': 'Lowest Cost',
#     'Winner': best_cost.iloc[0]['Model'],
#     'Runner-Up': best_cost.iloc[1]['Model'] if len(best_cost) > 1 else 'N/A',
#     'Key Metric': f"{best_cost.iloc[0]['Cost_mean']:.2f} (Quality: {best_cost.iloc[0]['Quality_mean']:.2f})"
# })

# # Fastest
# best_speed = model_stats_filtered.nsmallest(2, 'Latency_s_mean')
# ranking_data.append({
#     'Category': 'Fastest Latency',
#     'Winner': best_speed.iloc[0]['Model'],
#     'Runner-Up': best_speed.iloc[1]['Model'] if len(best_speed) > 1 else 'N/A',
#     'Key Metric': f"{best_speed.iloc[0]['Latency_s_mean']:.2f}s (Quality: {best_speed.iloc[0]['Quality_mean']:.2f})"
# })

# # Highest Quality
# best_quality = model_stats_filtered.nlargest(2, 'Quality_mean')
# ranking_data.append({
#     'Category': 'Highest Quality',
#     'Winner': best_quality.iloc[0]['Model'],
#     'Runner-Up': best_quality.iloc[1]['Model'] if len(best_quality) > 1 else 'N/A',
#     'Key Metric': f"{best_quality.iloc[0]['Quality_mean']:.2f}/5 (Cost: {best_quality.iloc[0]['Cost_mean']:.2f})"
# })

# ranking_df = pd.DataFrame(ranking_data)
# st.table(ranking_df)

# # --- Detailed Model Statistics ---
# st.header("📊 Detailed Model Statistics")
# display_stats = model_stats_filtered[['Model', 'Quality_mean', 'CO2_Emission_kg_mean', 'Cost_mean', 'Latency_s_mean']].copy()
# display_stats.columns = ['Model', 'Avg Quality', 'Avg CO2 (kg)', 'Avg Cost', 'Avg Latency (s)']
# display_stats = display_stats.round(2)
# st.dataframe(display_stats, use_container_width=True)

# # --- Raw Data Viewer ---
# with st.expander("View Raw Filtered Data"):
#     st.dataframe(df_filtered)



# Save this code as app.py and run with `streamlit run app.py`

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Model Performance Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --- Title and Introduction ---
st.title("📊 AI Model Performance Dashboard")
st.markdown("""
An expert dashboard for the interactive analysis of AI models from the `ComparIA_Benchmark` dataset.
Evaluate trade-offs between quality, cost, CO2 emissions, and latency. Use the sidebar filters to refine the analysis.
""")

# --- Data Loading and Transformation ---
@st.cache_data
def load_data(file_path):
    """
    Loads, cleans, transforms, and categorizes the benchmark data.
    - Converts CO2 from kg to g if necessary.
    - Assigns a 'Type' (Small, Medium, Large) to each model.
    - Categorizes each prompt.
    """
    try:
        df = pd.read_excel(file_path)
        
        # Standardize column names based on the provided file structure
        df.columns = ['Prompt', 'Model', 'Quality', 'CO2_Emission_kg', 'Latency_s', 'Cost']
            
        # --- Data Cleaning and Type Conversion ---
        for col in ['Quality', 'CO2_Emission_kg', 'Latency_s', 'Cost']:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # --- Unit Conversion: CO2 from kg to g ---
        df['CO2_Emission_g'] = df['CO2_Emission_kg'] * 1000
        df.drop(columns=['CO2_Emission_kg'], inplace=True)

        df.dropna(subset=['Quality', 'CO2_Emission_g', 'Latency_s', 'Cost'], inplace=True)

        # --- Assign Model 'Type' (Size) ---
        # Correctly mapping model names found in the Excel file
        model_type_map = {
            'Gemma 3nb': 'Small', 'Llama 3.1 8B': 'Small',
            'Mistral Small': 'Medium', 'GPT-OSS 20B': 'Medium',
            'GPT-5': 'Large', 'DeepSeek R1': 'Large'
        }
        df['Type'] = df['Model'].map(model_type_map)
        
        # --- Categorize Prompts ---
        def categorize_prompt(prompt_id):
            if 1 <= prompt_id <= 10 or 21 <= prompt_id <= 30:
                return "Literary/General"
            elif 11 <= prompt_id <= 15:
                return "Mathematical/Logical"
            elif 16 <= prompt_id <= 20:
                return "Coding"
            return "Other"
        df['Prompt_Type'] = df['Prompt'].apply(categorize_prompt)
        
        return df

    except FileNotFoundError:
        st.error(f"Error: The data file '{file_path}' was not found. Please ensure it's in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

# Load the data
df = load_data('FComparIA_Benchmark_LongFormat.xlsx')

if df.empty:
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")

selected_types = st.sidebar.multiselect(
    "Filter by Model Size (Type):",
    options=df['Type'].unique(),
    default=df['Type'].unique()
)

prompt_type_filter = st.sidebar.radio(
    "Filter by Prompt Category:",
    options=['All', 'Literary/General', 'Mathematical/Logical', 'Coding'],
    index=0
)

# Filter data based on selections
df_filtered = df[df['Type'].isin(selected_types)]
if prompt_type_filter != 'All':
    df_filtered = df_filtered[df_filtered['Prompt_Type'] == prompt_type_filter]

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# --- Aggregated Statistics ---
model_stats = df_filtered.groupby(['Model', 'Type']).agg(
    Quality_mean=('Quality', 'mean'),
    CO2_mean=('CO2_Emission_g', 'mean'),
    Cost_mean=('Cost', 'mean'),
    Latency_mean=('Latency_s', 'mean')
).reset_index()

# --- SECTION 1: OVERALL PERFORMANCE ANALYSIS ---
st.header("I. Overall Performance Analysis")

# Key Metrics Overview
st.subheader("Key Performance Indicators (KPIs)")
kpi_cols = st.columns(4)
best_quality = model_stats.loc[model_stats['Quality_mean'].idxmax()]
best_eco = model_stats.loc[model_stats['CO2_mean'].idxmin()]
best_cost = model_stats.loc[model_stats['Cost_mean'].idxmin()]
best_speed = model_stats.loc[model_stats['Latency_mean'].idxmin()]

kpi_cols[0].metric("🏆 Highest Quality", best_quality['Model'], f"{best_quality['Quality_mean']:.2f}/5")
kpi_cols[1].metric("🌱 Lowest CO2", best_eco['Model'], f"{best_eco['CO2_mean']:.1f} g")
kpi_cols[2].metric("💰 Lowest Cost", best_cost['Model'], f"{best_cost['Cost_mean']:.2f} Wh")
kpi_cols[3].metric("⚡ Fastest Latency", best_speed['Model'], f"{best_speed['Latency_mean']:.2f} s")

st.markdown("---")

# --- NEW: VISUALIZATIONS FOR AGGREGATE STATISTICS ---
st.subheader("Comparative Metric Analysis")
vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    # Quality Chart
    quality_sorted = model_stats.sort_values('Quality_mean', ascending=False)
    fig_qual = px.bar(quality_sorted, x='Model', y='Quality_mean', color='Type', title='Average Quality Score', labels={'Quality_mean': 'Avg Quality (1-5)'})
    fig_qual.update_layout(xaxis_title=None)
    st.plotly_chart(fig_qual, use_container_width=True)
    st.info("💡 **Quality:** Higher scores are better.")

    # Cost Chart
    cost_sorted = model_stats.sort_values('Cost_mean', ascending=True)
    fig_cost_bar = px.bar(cost_sorted, x='Model', y='Cost_mean', color='Type', title='Average Cost per Task', labels={'Cost_mean': 'Avg Cost (Wh)'})
    fig_cost_bar.update_layout(xaxis_title=None)
    st.plotly_chart(fig_cost_bar, use_container_width=True)
    st.info("💡 **Cost:** Lower values are better.")

with vis_col2:
    # CO2 Chart
    co2_sorted = model_stats.sort_values('CO2_mean', ascending=True)
    fig_co2 = px.bar(co2_sorted, x='Model', y='CO2_mean', color='Type', title='Average CO2 Emission', labels={'CO2_mean': 'Avg CO2 (g)'})
    fig_co2.update_layout(xaxis_title=None)
    st.plotly_chart(fig_co2, use_container_width=True)
    st.info("💡 **CO2 Emission:** Lower emissions are better.")

    # Latency Chart
    latency_sorted = model_stats.sort_values('Latency_mean', ascending=True)
    fig_lat = px.bar(latency_sorted, x='Model', y='Latency_mean', color='Type', title='Average Latency', labels={'Latency_mean': 'Avg Latency (s)'})
    fig_lat.update_layout(xaxis_title=None)
    st.plotly_chart(fig_lat, use_container_width=True)
    st.info("💡 **Latency:** Lower (faster) is better.")

st.markdown("---")

# Main Visualization Grid
st.subheader("Core Performance Trade-offs")
col1, col2 = st.columns(2)

with col1:
    fig_quality_co2 = px.scatter(
        df_filtered, x="CO2_Emission_g", y="Quality", color="Model", symbol="Type", size="Cost",
        hover_name="Model", hover_data={'Prompt': True, 'Cost': ':.2f', 'Latency_s': ':.2f'},
        title="Quality vs. CO2 Emission"
    )
    fig_quality_co2.update_layout(xaxis_title="CO2 Emission (g) → Lower is Better", yaxis_title="Answer Quality (1-5) → Higher is Better", legend_title_text='Model')
    fig_quality_co2.update_traces(marker=dict(sizemin=5))
    st.plotly_chart(fig_quality_co2, use_container_width=True)
    st.info("💡 **Analysis:** Ideal models are in the **top-left corner**. Shape indicates size; bubble size represents cost.")

with col2:
    fig_quality_cost = px.scatter(
        df_filtered, x="Cost", y="Quality", color="Model", symbol="Type", size="CO2_Emission_g",
        hover_name="Model", hover_data={'Prompt': True, 'CO2_Emission_g': ':.1f', 'Latency_s': ':.2f'},
        title="Quality vs. Cost"
    )
    fig_quality_cost.update_layout(xaxis_title="Cost per Task (Wh) → Lower is Better", yaxis_title="Answer Quality (1-5) → Higher is Better", legend_title_text='Model')
    fig_quality_cost.update_traces(marker=dict(sizemin=5))
    st.plotly_chart(fig_quality_cost, use_container_width=True)
    st.info("💡 **Analysis:** Ideal models are in the **top-left corner**. Shape indicates size; bubble size represents CO2.")

st.subheader("Multi-Dimensional Performance Radar")
radar_data = model_stats.copy()
metric_to_original_col = {'Quality_mean': 'Quality', 'CO2_mean': 'CO2_Emission_g', 'Cost_mean': 'Cost', 'Latency_mean': 'Latency_s'}
for metric, original_col in metric_to_original_col.items():
    min_val, max_val = df[original_col].min(), df[original_col].max()
    norm_name = f"{metric}_norm"
    if 'CO2' in metric or 'Cost' in metric or 'Latency' in metric:
        radar_data[norm_name] = 1 - (radar_data[metric] - min_val) / (max_val - min_val)
    else:
        radar_data[norm_name] = (radar_data[metric] - min_val) / (max_val - min_val)
fig_radar = go.Figure()
categories = ['Quality', 'Eco-Friendly', 'Cost-Effective', 'Speed']
for _, row in radar_data.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row['Quality_mean_norm'], row['CO2_mean_norm'], row['Cost_mean_norm'], row['Latency_mean_norm']],
        theta=categories, fill='toself', name=row['Model']
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
st.plotly_chart(fig_radar, use_container_width=True)
st.info("💡 **Analysis:** A larger and more uniform shape indicates better all-around performance.")

st.markdown("<br>", unsafe_allow_html=True)

# --- SECTION 2: DEEP DIVE ANALYSIS BY PROMPT CATEGORY ---
st.header("II. Deep Dive: Performance by Prompt Category")
if prompt_type_filter == 'All':
    prompt_stats = df_filtered.groupby(['Model', 'Type', 'Prompt_Type']).agg(Quality_mean=('Quality', 'mean'), Cost_mean=('Cost', 'mean')).reset_index()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Average Quality by Category")
        fig_prompt_qual = px.bar(prompt_stats, x='Model', y='Quality_mean', color='Prompt_Type', barmode='group', title="Quality: Literary vs. Math vs. Coding", labels={'Quality_mean': 'Average Quality (1-5)'}, facet_col='Type', category_orders={"Type": ["Small", "Medium", "Large"]})
        fig_prompt_qual.update_xaxes(title="")
        st.plotly_chart(fig_prompt_qual, use_container_width=True)
        st.info("💡 **Analysis:** Compare how models perform on different task types.")
    with col4:
        st.subheader("Average Cost (Wh) by Category")
        fig_prompt_cost = px.bar(prompt_stats, x='Model', y='Cost_mean', color='Prompt_Type', barmode='group', title="Cost: Literary vs. Math vs. Coding", labels={'Cost_mean': 'Average Cost (Wh)'}, facet_col='Type', category_orders={"Type": ["Small", "Medium", "Large"]})
        fig_prompt_cost.update_xaxes(title="")
        st.plotly_chart(fig_prompt_cost, use_container_width=True)
        st.info("💡 **Analysis:** Identify which models are most cost-effective for specific tasks.")
else:
    st.info(f"Displaying data only for the **'{prompt_type_filter}'** category. Select 'All' in the sidebar to view the comparative deep dive.")

st.markdown("---")

# --- SECTION 3: DATA TABLES ---
st.header("III. Data Tables")
tab1, tab2 = st.tabs(["Detailed Statistics", "Raw Data Viewer"])
with tab1:
    st.subheader("Aggregated Model Statistics")
    display_stats = model_stats[['Model', 'Type', 'Quality_mean', 'CO2_mean', 'Cost_mean', 'Latency_mean']].copy()
    display_stats.columns = ['Model', 'Type', 'Avg Quality', 'Avg CO2 (g)', 'Avg Cost (Wh)', 'Avg Latency (s)']
    st.dataframe(display_stats.set_index('Model').style.format({'Avg Quality': '{:.2f}', 'Avg CO2 (g)': '{:.1f}', 'Avg Cost (Wh)': '{:.2f}', 'Avg Latency (s)': '{:.2f}'}), use_container_width=True)
with tab2:
    st.subheader("Filtered Raw Data")
    st.dataframe(df_filtered)

