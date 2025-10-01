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
    Loads and processes the benchmark data directly from the user's file structure.
    This version is specifically adapted for the 7-column Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        
        # --- CRITICAL FIX: Assign names for all 7 columns from the Excel file ---
        # The column order is: Prompt, Type, Model, Quality, CO2 (g), Latency, Cost
        df.columns = ['Prompt', 'Type', 'Model', 'Quality', 'CO2_Emission_g', 'Latency_s', 'Cost']
            
        # --- Data Cleaning and Type Conversion ---
        for col in ['Quality', 'CO2_Emission_g', 'Latency_s', 'Cost']:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['Quality', 'CO2_Emission_g', 'Latency_s', 'Cost'], inplace=True)
        
        # --- Categorize Prompts ---
        def categorize_prompt(prompt_id):
            # Ensure prompt_id is an integer for comparison
            prompt_id = int(prompt_id)
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

# --- Load Data ---
# The name of the file in your GitHub repository
DATA_FILE = 'FinalComparIA_Benchmark_LongFormat.xlsx' 
df = load_data(DATA_FILE)

if df.empty:
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")

# Handle potential missing 'Type' column if data loading fails partially
if 'Type' in df.columns:
    selected_types = st.sidebar.multiselect(
        "Filter by Model Size (Type):",
        options=df['Type'].unique(),
        default=df['Type'].unique()
    )
    df_filtered = df[df['Type'].isin(selected_types)]
else:
    st.sidebar.warning("Model 'Type' column not found.")
    df_filtered = df.copy()


prompt_type_filter = st.sidebar.radio(
    "Filter by Prompt Category:",
    options=['All', 'Literary/General', 'Mathematical/Logical', 'Coding'],
    index=0
)

# Filter data based on selections
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

# --- VISUALIZATIONS FOR AGGREGATE STATISTICS ---
st.subheader("Comparative Metric Analysis")
vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    quality_sorted = model_stats.sort_values('Quality_mean', ascending=False)
    fig_qual = px.bar(quality_sorted, x='Model', y='Quality_mean', color='Type', title='Average Quality Score', labels={'Quality_mean': 'Avg Quality (1-5)'})
    fig_qual.update_layout(xaxis_title=None)
    st.plotly_chart(fig_qual, use_container_width=True)
    st.info("💡 **Quality:** Higher scores are better.")

    cost_sorted = model_stats.sort_values('Cost_mean', ascending=True)
    fig_cost_bar = px.bar(cost_sorted, x='Model', y='Cost_mean', color='Type', title='Average Cost per Task', labels={'Cost_mean': 'Avg Cost (Wh)'})
    fig_cost_bar.update_layout(xaxis_title=None)
    st.plotly_chart(fig_cost_bar, use_container_width=True)
    st.info("💡 **Cost:** Lower values are better.")

with vis_col2:
    co2_sorted = model_stats.sort_values('CO2_mean', ascending=True)
    fig_co2 = px.bar(co2_sorted, x='Model', y='CO2_mean', color='Type', title='Average CO2 Emission', labels={'CO2_mean': 'Avg CO2 (g)'})
    fig_co2.update_layout(xaxis_title=None)
    st.plotly_chart(fig_co2, use_container_width=True)
    st.info("💡 **CO2 Emission:** Lower emissions are better.")

    latency_sorted = model_stats.sort_values('Latency_mean', ascending=True)
    fig_lat = px.bar(latency_sorted, x='Model', y='Latency_mean', color='Type', title='Average Latency', labels={'Latency_mean': 'Avg Latency (s)'})
    fig_lat.update_layout(xaxis_title=None)
    st.plotly_chart(fig_lat, use_container_width=True)
    st.info("💡 **Latency:** Lower (faster) is better.")

st.markdown("---")

# Core Performance Trade-offs
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

# --- SECTION 2: DEEP DIVE & DATA TABLES ---
st.header("II. Deep Dive & Data")
tab1, tab2, tab3 = st.tabs(["Performance by Category", "Detailed Statistics", "Raw Data Viewer"])

with tab1:
    if prompt_type_filter == 'All':
        prompt_stats = df_filtered.groupby(['Model', 'Type', 'Prompt_Type']).agg(Quality_mean=('Quality', 'mean'), Cost_mean=('Cost', 'mean')).reset_index()
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Average Quality by Category")
            fig_prompt_qual = px.bar(prompt_stats, x='Model', y='Quality_mean', color='Prompt_Type', barmode='group', title="Quality: Literary vs. Math vs. Coding", labels={'Quality_mean': 'Average Quality (1-5)'}, facet_col='Type', category_orders={"Type": ["Small", "Medium", "Large"]})
            fig_prompt_qual.update_xaxes(title="")
            st.plotly_chart(fig_prompt_qual, use_container_width=True)
        with col4:
            st.subheader("Average Cost (Wh) by Category")
            fig_prompt_cost = px.bar(prompt_stats, x='Model', y='Cost_mean', color='Prompt_Type', barmode='group', title="Cost: Literary vs. Math vs. Coding", labels={'Cost_mean': 'Average Cost (Wh)'}, facet_col='Type', category_orders={"Type": ["Small", "Medium", "Large"]})
            fig_prompt_cost.update_xaxes(title="")
            st.plotly_chart(fig_prompt_cost, use_container_width=True)
    else:
        st.info(f"Displaying data only for the **'{prompt_type_filter}'** category. Select 'All' in the sidebar to view the comparative deep dive.")

with tab2:
    st.subheader("Aggregated Model Statistics")
    display_stats = model_stats[['Model', 'Type', 'Quality_mean', 'CO2_mean', 'Cost_mean', 'Latency_mean']].copy()
    display_stats.columns = ['Model', 'Type', 'Avg Quality', 'Avg CO2 (g)', 'Avg Cost (Wh)', 'Avg Latency (s)']
    st.dataframe(display_stats.set_index('Model').style.format({'Avg Quality': '{:.2f}', 'Avg CO2 (g)': '{:.1f}', 'Avg Cost (Wh)': '{:.2f}', 'Avg Latency (s)': '{:.2f}'}), use_container_width=True)

with tab3:
    st.subheader("Filtered Raw Data")
    st.dataframe(df_filtered)

