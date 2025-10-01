import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Model Performance Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --- Initialize Session State ---
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

# --- Title and Introduction ---
st.title("📊 AI Model Performance Dashboard")
st.markdown("""
An expert dashboard for the interactive analysis of AI models from the `ComparIA_Benchmark` dataset.
Evaluate trade-offs between quality, cost, CO2 emissions, and latency. Use the sidebar filters to refine the analysis.
""")

# --- Data Loading and Transformation ---
@st.cache_data
def load_data(file_source):
    """
    Loads and processes the benchmark data from file upload or default path.
    """
    try:
        if isinstance(file_source, str):
            df = pd.read_excel(file_source) if file_source.endswith(('.xlsx', '.xls')) else pd.read_csv(file_source)
        else:
            if file_source.name.endswith('.csv'):
                df = pd.read_csv(file_source)
            else:
                df = pd.read_excel(file_source)
        
        # Assign column names
        df.columns = ['Prompt', 'Type', 'Model', 'Quality', 'CO2_Emission_g', 'Latency_s', 'Cost']
            
        # Data Cleaning and Type Conversion
        for col in ['Quality', 'CO2_Emission_g', 'Latency_s', 'Cost']:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['Quality', 'CO2_Emission_g', 'Latency_s', 'Cost'], inplace=True)
        
        # Categorize Prompts
        def categorize_prompt(prompt_id):
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

    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

def get_dataset_summary(df):
    """Generate dataset summary statistics"""
    return {
        'n_models': df['Model'].nunique(),
        'n_prompts': df['Prompt'].nunique(),
        'n_records': len(df),
        'quality_range': (df['Quality'].min(), df['Quality'].max()),
        'co2_range': (df['CO2_Emission_g'].min(), df['CO2_Emission_g'].max()),
        'cost_range': (df['Cost'].min(), df['Cost'].max()),
        'latency_range': (df['Latency_s'].min(), df['Latency_s'].max())
    }

def pareto_front(df, x='CO2_mean', y='Quality_mean'):
    """Calculate Pareto front for multi-objective optimization"""
    pts = df[[x, y, 'Model']].copy()
    pts = pts.sort_values(x)
    pareto = []
    current_best = -1e9
    for _, row in pts.iterrows():
        if row[y] > current_best:
            pareto.append(row)
            current_best = row[y]
    return pd.DataFrame(pareto)

# --- Sidebar: File Upload ---
st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your benchmark file (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

# Load data
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("✅ Custom file loaded!")
else:
    DATA_FILE = 'FinalComparIA_Benchmark_LongFormat.xlsx'
    df = load_data(DATA_FILE)

if df.empty:
    st.error("⚠️ No data loaded. Please check your file.")
    st.stop()

# --- Dataset Summary ---
with st.expander("📋 Dataset Summary", expanded=False):
    summary = get_dataset_summary(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", summary['n_models'])
    col2.metric("Prompts", summary['n_prompts'])
    col3.metric("Total Records", summary['n_records'])
    col4.metric("Model Types", df['Type'].nunique())
    
    st.markdown("**Metric Ranges:**")
    st.write(f"- Quality: {summary['quality_range'][0]:.1f} - {summary['quality_range'][1]:.1f}")
    st.write(f"- CO2: {summary['co2_range'][0]:.1f}g - {summary['co2_range'][1]:.1f}g")
    st.write(f"- Cost: {summary['cost_range'][0]:.2f} - {summary['cost_range'][1]:.2f} Wh")
    st.write(f"- Latency: {summary['latency_range'][0]:.2f}s - {summary['latency_range'][1]:.2f}s")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Dashboard Filters")

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

# --- Composite Score Weights ---
st.sidebar.header("⚖️ Composite Score - Weights")
st.sidebar.markdown("*Adjust weights for custom model ranking*")
w_quality = st.sidebar.slider("Quality Weight", 0.0, 1.0, 0.40, 0.05)
w_co2 = st.sidebar.slider("CO2 Weight (lower is better)", 0.0, 1.0, 0.20, 0.05)
w_cost = st.sidebar.slider("Cost Weight (lower is better)", 0.0, 1.0, 0.20, 0.05)
w_latency = st.sidebar.slider("Latency Weight (lower is better)", 0.0, 1.0, 0.20, 0.05)

total_w = max(1e-9, w_quality + w_co2 + w_cost + w_latency)
weights = {
    'Quality': w_quality/total_w,
    'CO2_Emission_g': w_co2/total_w,
    'Cost': w_cost/total_w,
    'Latency_s': w_latency/total_w
}

# --- Advanced Options ---
with st.sidebar.expander("⚙️ Advanced Options"):
    use_log_scale = st.checkbox("Use log scale for CO2/Cost", value=False)
    show_outliers = st.checkbox("Show outliers", value=True)

# Filter data
df_filtered = df[df['Type'].isin(selected_types)]
if prompt_type_filter != 'All':
    df_filtered = df_filtered[df_filtered['Prompt_Type'] == prompt_type_filter]

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# --- Aggregated Statistics ---
model_stats = df_filtered.groupby(['Model', 'Type']).agg(
    Quality_mean=('Quality', 'mean'),
    Quality_std=('Quality', 'std'),
    CO2_mean=('CO2_Emission_g', 'mean'),
    CO2_std=('CO2_Emission_g', 'std'),
    Cost_mean=('Cost', 'mean'),
    Cost_std=('Cost', 'std'),
    Latency_mean=('Latency_s', 'mean'),
    Latency_std=('Latency_s', 'std'),
    Count=('Quality', 'count')
).reset_index()

# --- Calculate Composite Score ---
ms = model_stats.copy()
# Normalize and invert where necessary
for metric, col_mean in [('CO2_Emission_g', 'CO2_mean'), ('Cost', 'Cost_mean'), ('Latency_s', 'Latency_mean')]:
    min_val, max_val = ms[col_mean].min(), ms[col_mean].max()
    ms[f'{metric}_norm'] = 1 - (ms[col_mean] - min_val) / (max_val - min_val + 1e-9)

min_val, max_val = ms['Quality_mean'].min(), ms['Quality_mean'].max()
ms['Quality_norm'] = (ms['Quality_mean'] - min_val) / (max_val - min_val + 1e-9)

ms['Composite'] = (
    weights['Quality'] * ms['Quality_norm'] +
    weights['CO2_Emission_g'] * ms['CO2_Emission_g_norm'] +
    weights['Cost'] * ms['Cost_norm'] +
    weights['Latency_s'] * ms['Latency_s_norm']
)
ms = ms.sort_values('Composite', ascending=False)

# --- Export Functionality ---
st.sidebar.header("💾 Export Data")
if not df_filtered.empty:
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="benchmark_filtered.csv",
        mime="text/csv"
    )
    
    # Export model statistics
    stats_csv = ms.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Statistics (CSV)",
        data=stats_csv,
        file_name="model_statistics.csv",
        mime="text/csv"
    )

# --- SECTION 1: OVERALL PERFORMANCE ANALYSIS ---
st.header("I. Overall Performance Analysis")

# Key Metrics Overview
st.subheader("🎯 Key Performance Indicators (KPIs)")
kpi_cols = st.columns(5)
best_quality = model_stats.loc[model_stats['Quality_mean'].idxmax()]
best_eco = model_stats.loc[model_stats['CO2_mean'].idxmin()]
best_cost = model_stats.loc[model_stats['Cost_mean'].idxmin()]
best_speed = model_stats.loc[model_stats['Latency_mean'].idxmin()]
best_composite = ms.iloc[0]

kpi_cols[0].metric("🏆 Highest Quality", best_quality['Model'], f"{best_quality['Quality_mean']:.2f}/5")
kpi_cols[1].metric("🌱 Lowest CO2", best_eco['Model'], f"{best_eco['CO2_mean']:.1f} g")
kpi_cols[2].metric("💰 Lowest Cost", best_cost['Model'], f"{best_cost['Cost_mean']:.2f} Wh")
kpi_cols[3].metric("⚡ Fastest", best_speed['Model'], f"{best_speed['Latency_mean']:.2f} s")
kpi_cols[4].metric("🎖️ Best Composite", best_composite['Model'], f"{best_composite['Composite']:.3f}")

st.markdown("---")

# Composite Score Ranking
st.subheader("🏅 Model Ranking by Composite Score")
with st.expander("ℹ️ How to interpret", expanded=False):
    st.markdown("""
    **Composite Score** combines all metrics using your custom weights:
    - Higher scores indicate better overall performance
    - Adjust weights in the sidebar to match your priorities
    - Models are normalized (0-1) for fair comparison
    """)

display_composite = ms[['Model', 'Type', 'Quality_mean', 'CO2_mean', 'Cost_mean', 'Latency_mean', 'Composite']].head(10)
display_composite.columns = ['Model', 'Type', 'Quality', 'CO2 (g)', 'Cost (Wh)', 'Latency (s)', 'Score']
st.dataframe(
    display_composite.style.format({
        'Quality': '{:.2f}',
        'CO2 (g)': '{:.1f}',
        'Cost (Wh)': '{:.3f}',
        'Latency (s)': '{:.2f}',
        'Score': '{:.3f}'
    }).background_gradient(subset=['Score'], cmap='RdYlGn'),
    use_container_width=True
)

st.markdown("---")

# Core Performance Trade-offs
st.subheader("📈 Core Performance Trade-offs")

with st.expander("ℹ️ How to interpret scatter plots", expanded=False):
    st.markdown("""
    - **Top-left corner** = optimal zone (high quality, low environmental/cost impact)
    - **Point shape** indicates model size (Small/Medium/Large)
    - **Bubble size** represents the third dimension
    - **Hover** for detailed information about each point
    """)

col1, col2 = st.columns(2)

with col1:
    fig_quality_co2 = px.scatter(
        df_filtered, x="CO2_Emission_g", y="Quality", color="Model", symbol="Type", size="Cost",
        hover_name="Model", hover_data={'Prompt': True, 'Cost': ':.2f', 'Latency_s': ':.2f'},
        title="Quality vs. CO2 Emission",
        log_x=use_log_scale
    )
    fig_quality_co2.update_layout(
        xaxis_title="CO2 Emission (g) → Lower is Better",
        yaxis_title="Answer Quality (1-5) → Higher is Better"
    )
    fig_quality_co2.update_traces(marker=dict(sizemin=5))
    st.plotly_chart(fig_quality_co2, use_container_width=True)

with col2:
    fig_quality_cost = px.scatter(
        df_filtered, x="Cost", y="Quality", color="Model", symbol="Type", size="CO2_Emission_g",
        hover_name="Model", hover_data={'Prompt': True, 'CO2_Emission_g': ':.1f', 'Latency_s': ':.2f'},
        title="Quality vs. Cost",
        log_x=use_log_scale
    )
    fig_quality_cost.update_layout(
        xaxis_title="Cost per Task (Wh) → Lower is Better",
        yaxis_title="Answer Quality (1-5) → Higher is Better"
    )
    fig_quality_cost.update_traces(marker=dict(sizemin=5))
    st.plotly_chart(fig_quality_cost, use_container_width=True)

# Pareto Front Analysis
st.subheader("🎯 Pareto Frontier Analysis")
with st.expander("ℹ️ Understanding Pareto fronts", expanded=False):
    st.markdown("""
    The **Pareto front** shows non-dominated solutions:
    - Models on the front cannot be improved in one metric without sacrificing another
    - Models **below** the front are suboptimal
    - Use this to identify the best trade-offs for your use case
    """)

pareto_col1, pareto_col2 = st.columns(2)

with pareto_col1:
    pf_co2 = pareto_front(model_stats, x='CO2_mean', y='Quality_mean')
    fig_pareto_co2 = px.scatter(
        model_stats, x='CO2_mean', y='Quality_mean', text='Model',
        color='Type', size='Cost_mean', title='Pareto: Quality vs CO2'
    )
    fig_pareto_co2.add_trace(go.Scatter(
        x=pf_co2['CO2_mean'], y=pf_co2['Quality_mean'],
        mode='lines+markers', name='Pareto Front',
        line=dict(width=3, dash='dash', color='red'),
        marker=dict(size=10, color='red')
    ))
    fig_pareto_co2.update_traces(textposition='top center')
    fig_pareto_co2.update_layout(xaxis_title="CO2 (g)", yaxis_title="Quality (1-5)")
    st.plotly_chart(fig_pareto_co2, use_container_width=True)

with pareto_col2:
    pf_cost = pareto_front(model_stats, x='Cost_mean', y='Quality_mean')
    fig_pareto_cost = px.scatter(
        model_stats, x='Cost_mean', y='Quality_mean', text='Model',
        color='Type', size='CO2_mean', title='Pareto: Quality vs Cost'
    )
    fig_pareto_cost.add_trace(go.Scatter(
        x=pf_cost['Cost_mean'], y=pf_cost['Quality_mean'],
        mode='lines+markers', name='Pareto Front',
        line=dict(width=3, dash='dash', color='red'),
        marker=dict(size=10, color='red')
    ))
    fig_pareto_cost.update_traces(textposition='top center')
    fig_pareto_cost.update_layout(xaxis_title="Cost (Wh)", yaxis_title="Quality (1-5)")
    st.plotly_chart(fig_pareto_cost, use_container_width=True)

st.markdown("---")

# --- SECTION 2: STATISTICAL ANALYSIS ---
st.header("II. Statistical Analysis & Distributions")

tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📉 Performance by Category"])

with tab1:
    st.subheader("Metric Distributions by Model Type")
    with st.expander("ℹ️ How to interpret box plots"):
        st.markdown("""
        - **Box** shows the interquartile range (25th-75th percentile)
        - **Line inside box** is the median
        - **Whiskers** extend to 1.5× IQR
        - **Dots** are outliers (if shown)
        """)
    
    box_col1, box_col2 = st.columns(2)
    with box_col1:
        fig_box_quality = px.box(df_filtered, x='Type', y='Quality', color='Type',
                                  title='Quality Distribution by Model Size')
        st.plotly_chart(fig_box_quality, use_container_width=True)
        
        fig_box_co2 = px.box(df_filtered, x='Type', y='CO2_Emission_g', color='Type',
                             title='CO2 Distribution by Model Size', log_y=use_log_scale)
        st.plotly_chart(fig_box_co2, use_container_width=True)
    
    with box_col2:
        fig_box_cost = px.box(df_filtered, x='Type', y='Cost', color='Type',
                              title='Cost Distribution by Model Size', log_y=use_log_scale)
        st.plotly_chart(fig_box_cost, use_container_width=True)
        
        fig_box_latency = px.box(df_filtered, x='Type', y='Latency_s', color='Type',
                                 title='Latency Distribution by Model Size')
        st.plotly_chart(fig_box_latency, use_container_width=True)

with tab2:
    st.subheader("Metric Correlation Analysis")
    with st.expander("ℹ️ Understanding correlations"):
        st.markdown("""
        - **Spearman correlation** measures monotonic relationships
        - Values range from -1 (perfect negative) to +1 (perfect positive)
        - **Close to 0** = no correlation
        - Helps identify trade-offs between metrics
        """)
    
    corr = df_filtered[['Quality', 'CO2_Emission_g', 'Cost', 'Latency_s']].corr(method='spearman')
    fig_corr = px.imshow(
        corr, text_auto='.2f', aspect='auto',
        title="Spearman Correlation Matrix",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("**Key Insights:**")
    st.write(f"- Quality ↔ CO2: {corr.loc['Quality', 'CO2_Emission_g']:.2f}")
    st.write(f"- Quality ↔ Cost: {corr.loc['Quality', 'Cost']:.2f}")
    st.write(f"- CO2 ↔ Cost: {corr.loc['CO2_Emission_g', 'Cost']:.2f}")

with tab3:
    if prompt_type_filter == 'All':
        st.subheader("Performance Across Task Categories")
        with st.expander("ℹ️ Category analysis explanation"):
            st.markdown("""
            Compare how models perform on different types of tasks:
            - **Literary/General**: Creative writing, summarization
            - **Mathematical/Logical**: Problem solving, reasoning
            - **Coding**: Programming tasks
            """)
        
        prompt_stats = df_filtered.groupby(['Model', 'Type', 'Prompt_Type']).agg(
            Quality_mean=('Quality', 'mean'),
            CO2_mean=('CO2_Emission_g', 'mean'),
            Cost_mean=('Cost', 'mean')
        ).reset_index()
        
        cat_col1, cat_col2 = st.columns(2)
        with cat_col1:
            fig_cat_qual = px.bar(
                prompt_stats, x='Model', y='Quality_mean', color='Prompt_Type',
                barmode='group', title="Quality by Task Category",
                labels={'Quality_mean': 'Avg Quality'},
                facet_col='Type', category_orders={"Type": ["Small", "Medium", "Large"]}
            )
            fig_cat_qual.update_xaxes(title="")
            st.plotly_chart(fig_cat_qual, use_container_width=True)
        
        with cat_col2:
            fig_cat_cost = px.bar(
                prompt_stats, x='Model', y='Cost_mean', color='Prompt_Type',
                barmode='group', title="Cost by Task Category",
                labels={'Cost_mean': 'Avg Cost (Wh)'},
                facet_col='Type', category_orders={"Type": ["Small", "Medium", "Large"]}
            )
            fig_cat_cost.update_xaxes(title="")
            st.plotly_chart(fig_cat_cost, use_container_width=True)
    else:
        st.info(f"Currently filtered to **'{prompt_type_filter}'** category. Select 'All' in the sidebar to compare across categories.")

st.markdown("---")

# --- SECTION 3: DATA TABLES ---
st.header("III. Detailed Data & Rankings")

data_tab1, data_tab2, data_tab3 = st.tabs(["📋 Model Statistics", "🔍 Raw Data", "📊 Model Comparison"])

with data_tab1:
    st.subheader("Comprehensive Model Statistics")
    display_stats = model_stats[['Model', 'Type', 'Quality_mean', 'Quality_std', 
                                  'CO2_mean', 'CO2_std', 'Cost_mean', 'Cost_std',
                                  'Latency_mean', 'Latency_std', 'Count']].copy()
    display_stats.columns = ['Model', 'Type', 'Quality (μ)', 'Quality (σ)', 
                             'CO2 (μ)', 'CO2 (σ)', 'Cost (μ)', 'Cost (σ)',
                             'Latency (μ)', 'Latency (σ)', 'N']
    
    st.dataframe(
        display_stats.style.format({
            'Quality (μ)': '{:.2f}', 'Quality (σ)': '{:.2f}',
            'CO2 (μ)': '{:.1f}', 'CO2 (σ)': '{:.1f}',
            'Cost (μ)': '{:.3f}', 'Cost (σ)': '{:.3f}',
            'Latency (μ)': '{:.2f}', 'Latency (σ)': '{:.2f}'
        }),
        use_container_width=True
    )

with data_tab2:
    st.subheader("Filtered Raw Data")
    st.write(f"Showing {len(df_filtered)} records")
    st.dataframe(df_filtered, use_container_width=True)

with data_tab3:
    st.subheader("Side-by-Side Model Comparison")
    selected_models = st.multiselect(
        "Select models to compare:",
        options=df_filtered['Model'].unique(),
        default=list(df_filtered['Model'].unique()[:3])
    )
    
    if selected_models:
        comparison_data = model_stats[model_stats['Model'].isin(selected_models)]
        
        # Create comparison chart
        fig_comparison = go.Figure()
        metrics = ['Quality_mean', 'CO2_mean', 'Cost_mean', 'Latency_mean']
        metric_names = ['Quality', 'CO2', 'Cost', 'Latency']
        
        for model in selected_models:
            model_data = comparison_data[comparison_data['Model'] == model]
            normalized_values = []
            for metric in metrics:
                val = model_data[metric].values[0]
                min_val = comparison_data[metric].min()
                max_val = comparison_data[metric].max()
                if 'CO2' in metric or 'Cost' in metric or 'Latency' in metric:
                    norm = 1 - (val - min_val) / (max_val - min_val + 1e-9)
                else:
                    norm = (val - min_val) / (max_val - min_val + 1e-9)
                normalized_values.append(norm)
            
            fig_comparison.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=metric_names,
                fill='toself',
                name=model
            ))
        
        fig_comparison.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalized Model Comparison (1 = Best)"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("**Detailed Metrics:**")
        comp_table = comparison_data[['Model', 'Type', 'Quality_mean', 'CO2_mean', 
                                      'Cost_mean', 'Latency_mean']].copy()
        comp_table.columns = ['Model', 'Type', 'Quality', 'CO2 (g)', 'Cost (Wh)', 'Latency (s)']
        st.dataframe(
            comp_table.style.format({
                'Quality': '{:.2f}',
                'CO2 (g)': '{:.1f}',
                'Cost (Wh)': '{:.3f}',
                'Latency (s)': '{:.2f}'
            }),
            use_container_width=True
        )

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>📊 AI Model Performance Dashboard | Built with Streamlit & Plotly</p>
    <p>💡 Tip: Adjust weights in the sidebar to customize the composite score for your priorities</p>
</div>
""", unsafe_allow_html=True)
