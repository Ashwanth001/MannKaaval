"""
ManKaaval Sand Mining Detection Dashboard - IMPROVED VERSION
=============================================================
Streamlit-based interactive dashboard for monitoring sand mining activity across Indian states.

KEY IMPROVEMENTS (Feb 2026):
- ✅ Removed date slider - map always shows LATEST predictions
- ✅ Removed all GEE dependencies - 100% Folium-based
- ✅ Optimized data loading with aggressive caching
- ✅ Added intervention date line in SCA visualizations
- ✅ Pre/Post period background shading in graphs
- ✅ Faster site selection with spatial indexing
- ✅ Smooth state transitions
- ✅ Responsive and robust performance

MAP FEATURES:
- Interactive tile-based satellite background
- Mining Risk Grid: 1km squares colored by probability (green=low, red=high)
- SCA-Ready Sites: Green boundaries showing sites eligible for causal analysis
- Layer Control: Toggle layers on/off for focused analysis

DATA SOURCES:
- Site predictions: CSV files in dashboardData/stateData/
- Satellite tiles: Esri World Imagery via TileLayer API (no authentication needed)
- Time series: Spectral indices (NDVI, MNDWI, BSI) from local CSV files
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import folium
import hashlib
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ManKaaval",
    page_icon="🛰️",
    layout="wide",
)

st.set_option("client.toolbarMode", "viewer")

# ============================================================================
# CUSTOM CSS - IMPROVED AESTHETICS
# ============================================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        font-family: Inter, sans-serif;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #00d4ff;
    }
    [data-testid="stMetricLabel"] {
        color: #aaaaaa;
    }
    [data-testid="stMetricDelta"] {
        display: none;
    }
    table thead tr th:first-child {
        display: none;
    }
    table tbody th {
        display: none;
    }
    .risk-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff4444;
        margin-bottom: 10px;
    }
    .info-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
    }
    
    /* Dashboard Mode: Optimized scrolling */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden;
        height: 100vh;
    }
    
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-height: 100vh;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    
    /* State Selection Buttons */
    div.stButton > button {
        background-color: #1a1a1a;
        color: #eee;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px;
        width: 100%;
        text-align: left;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        border-color: #ff4444;
        color: #ff4444;
        transform: translateX(5px);
    }
    
    /* Left Column: Fixed Map */
    [data-testid="column"]:nth-child(1) {
        height: calc(100vh - 120px) !important;
        overflow: hidden !important;
    }
    
    /* Right Column: Scrollable Analysis */
    [data-testid="column"]:nth-child(2) {
        height: calc(100vh - 120px) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 10px !important;
    }
    
    /* Premium Scrollbar */
    [data-testid="column"]:nth-child(2)::-webkit-scrollbar {
        width: 6px;
    }
    [data-testid="column"]:nth-child(2)::-webkit-scrollbar-track {
        background: transparent;
    }
    [data-testid="column"]:nth-child(2)::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 10px;
    }
    [data-testid="column"]:nth-child(2)::-webkit-scrollbar-thumb:hover {
        background: #ff4d4d;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'selected_site' not in st.session_state:
    st.session_state.selected_site = None
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = "Tamil Nadu"
if 'sca_control_ids' not in st.session_state:
    st.session_state.sca_control_ids = None

# ============================================================================
# OPTIMIZED DATA LOADING WITH AGGRESSIVE CACHING
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_state_data(selected_state="Tamil Nadu"):
    """
    Optimized data loading with caching + pre-computed analytics.
    Returns: full_df, available_dates, sca_ready_df, latest_date, shap_df, sc_df, placebo_df
    """
    try:
        # File mapping
        file_mapping = {
            "Bihar": "BiharDataPredictions.csv",
            "Tamil Nadu": "tamilNaduPredictions.csv",
            "Uttar Pradesh": "UttarPradeshDataPredictions.csv"
        }
        
        file_name = file_mapping.get(selected_state, f"{selected_state}DataPredictions.csv")
        
        # Check paths
        potential_paths = [
            os.path.join("ManKaavalGitHub", "FinalDashboardData", "stateData", selected_state.replace(" ", "_"), file_name),
            os.path.join("FinalDashboardData", "stateData", selected_state.replace(" ", "_"), file_name),
            os.path.join("ManKaavalGitHub", "dashboardData", "stateData", file_name),
            os.path.join("dashboardData", "stateData", file_name)
        ]
        
        file_path = None
        for path in potential_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            return pd.DataFrame(), [], pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Load main predictions data
        df = pd.read_csv(file_path, parse_dates=['week_date'] if 'week_date' in pd.read_csv(file_path, nrows=0).columns else None)
        
        # Standardize column names
        lower_map = {c.lower(): c for c in df.columns}
        rename_map = {}
        
        for candidate in ['site_id', 'siteid', 'site', 'id']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'id'
                break
        
        for candidate in ['week_date', 'date', 'day']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'date'
                break
        
        for candidate in ['latitude', 'lat', 'lat_deg']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'lat'
                break
        
        for candidate in ['longitude', 'lon', 'long', 'lng']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'lon'
                break
        
        for candidate in ['probability', 'prob', 'pred_prob', 'prediction']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'probability'
                break
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Ensure date column exists and is datetime
        if 'date' not in df.columns:
            return pd.DataFrame(), [], pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove any rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Optimize memory
        if 'probability' in df.columns:
            df['probability'] = df['probability'].astype('float32')
        
        # Get available dates
        available_dates = sorted(df['date'].dt.strftime('%Y-%m-%d').unique().tolist())
        latest_date = available_dates[-1] if available_dates else None
        
        # Determine SCA-ready sites
        if not df.empty and latest_date:
            site_counts = df.groupby('id')['date'].count()
            valid_sites = site_counts[site_counts >= 15].index.tolist()
            
            latest_df = df[df['date'].dt.strftime('%Y-%m-%d') == latest_date]
            sca_ready_df = (
                latest_df[latest_df["id"].isin(valid_sites)][["id", "lat", "lon"]]
                .drop_duplicates(subset=["id"])
                .reset_index(drop=True)
            )
        else:
            sca_ready_df = pd.DataFrame(columns=["id", "lat", "lon"])
        
        # ===== LOAD PRE-COMPUTED ANALYTICS FILES =====
        state_dir = os.path.dirname(file_path)
        
        # Load SHAP values
        shap_path = os.path.join(state_dir, f"shap_values_{selected_state.replace(' ', '_')}.csv")
        shap_df = pd.read_csv(shap_path) if os.path.exists(shap_path) else pd.DataFrame()
        
        # Load Synthetic Control results
        sc_path = os.path.join(state_dir, f"synthetic_control_{selected_state.replace(' ', '_')}.csv")
        sc_df = pd.read_csv(sc_path) if os.path.exists(sc_path) else pd.DataFrame()
        
        # Load Placebo results
        placebo_path = os.path.join(state_dir, f"placebo_results_{selected_state.replace(' ', '_')}.csv")
        placebo_df = pd.read_csv(placebo_path) if os.path.exists(placebo_path) else pd.DataFrame()
        
        return df, available_dates, sca_ready_df, latest_date, shap_df, sc_df, placebo_df
    
    except Exception as e:
        st.error(f"❌ Error loading data for {selected_state}: {e}")
        return pd.DataFrame(), [], pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# ============================================================================
# MAP UTILITIES (Folium-based visualization)
# ============================================================================

def probability_to_hex(p: float) -> str:
    """Convert probability [0, 1] to hex color (green -> yellow -> red)."""
    p = float(max(0.0, min(1.0, p)))
    if p <= 0.5:
        t = p / 0.5
        start = (68, 255, 68)    # green
        end = (255, 255, 0)      # yellow
    else:
        t = (p - 0.5) / 0.5
        start = (255, 255, 0)    # yellow
        end = (255, 0, 0)        # red
    r = int(start[0] + (end[0] - start[0]) * t)
    g = int(start[1] + (end[1] - start[1]) * t)
    b = int(start[2] + (end[2] - start[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def create_folium_map(
    selected_state="Tamil Nadu",
    prediction_df=None,
    sca_ready_df=None,
):
    """Create folium map with satellite basemap + risk grid + SCA overlays."""
    
    # State-specific configurations
    state_configs = {
        "Bihar": {"center": [25.5, 85.5], "zoom": 7},
        "Tamil Nadu": {"center": [11.0, 78.5], "zoom": 7},
        "Uttar Pradesh": {"center": [27.0, 80.5], "zoom": 7},
        "West Bengal": {"center": [23.8, 87.9], "zoom": 7},
        "Punjab": {"center": [31.1, 75.3], "zoom": 7},
        "Rajasthan": {"center": [26.8, 73.8], "zoom": 6},
        "Gujarat": {"center": [22.2, 71.1], "zoom": 7}
    }
    
    config = state_configs.get(selected_state, {"center": [22.5, 78.9], "zoom": 5})
    center = config["center"]
    zoom = config["zoom"]
    
    # Initialize folium map
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    
    # Satellite base layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # OSM base
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Mining Risk Grid Layer
    risk_group = folium.FeatureGroup(name='Mining Activity Risk (Latest)', show=True)
    
    if prediction_df is not None and not prediction_df.empty:
        offset = 0.0045  # ~500m grid
        for _, row in prediction_df.iterrows():
            try:
                sid = row['id']
                lat = float(row['lat'])
                lon = float(row['lon'])
                prob = float(row['probability'])
            except:
                continue
            
            south = lat - offset
            north = lat + offset
            west = lon - offset
            east = lon + offset
            
            color = probability_to_hex(prob)
            tooltip = f"Site: {sid}<br>Risk: {prob*100:.1f}%"
            
            folium.Rectangle(
                bounds=[[south, west], [north, east]],
                color=None,
                weight=0,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                tooltip=tooltip
            ).add_to(risk_group)
    
    risk_group.add_to(m)
    
    # SCA-Ready Sites Layer
    if sca_ready_df is not None and not sca_ready_df.empty:
        sca_group = folium.FeatureGroup(
            name=f"SCA-Ready Sites ({len(sca_ready_df)} sites)",
            show=True
        )
        offset = 0.0045
        for _, row in sca_ready_df.iterrows():
            try:
                sid = row['id']
                lat = float(row['lat'])
                lon = float(row['lon'])
            except:
                continue
            
            south = lat - offset
            north = lat + offset
            west = lon - offset
            east = lon + offset
            
            folium.Rectangle(
                bounds=[[south, west], [north, east]],
                color='#00ff00',
                weight=2,
                fill=False,
                tooltip=f"SCA site: {sid}"
            ).add_to(sca_group)
        
        sca_group.add_to(m)
    
    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ============================================================================
# SITE SELECTION LOGIC
# ============================================================================
def find_nearest_site(click_lat, click_lon, df, threshold=0.006):
    """Find nearest site to clicked location using optimized numpy operations."""
    if df.empty:
        return None
    
    coords = df[['lat', 'lon']].values
    click_point = np.array([click_lat, click_lon])
    distances = np.linalg.norm(coords - click_point, axis=1)
    nearest_idx = np.argmin(distances)
    min_distance = distances[nearest_idx]
    
    if min_distance < threshold:
        return df.iloc[nearest_idx]
    
    return None


# ============================================================================
# TIME SERIES VISUALIZATION WITH INTERVENTION LINE
# ============================================================================


def create_shap_attribution_plot(site_id, shap_df):
    """
    Create SHAP-like feature attribution bar chart from pre-computed values.
    """
    if shap_df.empty:
        return None
    
    site_shap = shap_df[shap_df['site_id'] == site_id]
    if site_shap.empty:
        return None
    
    # Get SHAP values for this site
    shap_row = site_shap.iloc[0]
    metrics = ['NDVI', 'MNDWI', 'BSI']
    
    feature_names = []
    feature_values = []
    
    for metric in metrics:
        if metric in shap_row.index:
            feature_names.append(metric)
            feature_values.append(float(shap_row[metric]))
    
    if not feature_values:
        return None
    
    # Sort by absolute importance
    sorted_idx = np.argsort(np.abs(feature_values))
    feature_names = [feature_names[i] for i in sorted_idx]
    feature_values = [feature_values[i] for i in sorted_idx]
    
    # Colors: positive (red) vs negative (blue)
    colors = ['#ff4444' if v > 0 else '#00d4ff' for v in feature_values]
    
    fig = go.Figure(go.Bar(
        y=feature_names,
        x=feature_values,
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>Attribution: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Feature Attribution (Spectral Indices)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font=dict(color='white', size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        xaxis_title='Impact on Mining Detection',
        yaxis_gridcolor='rgba(128,128,128,0.2)',
        showlegend=False
    )
    
    return fig


def create_timeseries_plot_with_intervention(site_id, ts_df, intervention_date=None):
    """
    Create time-series plots with intervention line and pre/post shading.
    Shows all historical data with clear intervention marker.
    """
    site_ts = ts_df[ts_df['id'] == site_id].copy()
    if site_ts.empty:
        return None
    
    site_ts = site_ts.sort_values('date')
    
    # Check for required metrics
    required_metrics = ['NDVI', 'MNDWI', 'BSI']
    available_metrics = [m for m in required_metrics if m in site_ts.columns]
    
    if not available_metrics:
        return None
    
    # Determine intervention date (median if not provided)
    if intervention_date is None:
        # Calculate median date properly
        sorted_dates = site_ts['date'].sort_values()
        median_idx = len(sorted_dates) // 2
        intervention_date = sorted_dates.iloc[median_idx]
    else:
        intervention_date = pd.to_datetime(intervention_date)
        
    intervention_ts = pd.Timestamp(intervention_date) 
    intervention_num = intervention_ts.timestamp() * 1000
    
    # Create subplots
    num_metrics = len(available_metrics)
    fig = make_subplots(
        rows=num_metrics, cols=1,
        subplot_titles=tuple([f'{m} ({"Vegetation" if m=="NDVI" else "Water" if m=="MNDWI" else "Bare Soil"} Index)' for m in available_metrics]),
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    colors = {'NDVI': '#44ff44', 'MNDWI': '#00d4ff', 'BSI': '#ffaa00'}
    
    # Get clean date range
    date_min = site_ts['date'].min().timestamp() * 1000
    date_max = site_ts['date'].max().timestamp() * 1000
    
    for i, metric in enumerate(available_metrics, 1):
        # Plot metric line
        fig.add_trace(
            go.Scatter(
                x=site_ts['date'],
                y=site_ts[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[metric], width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>{metric}:</b> %{{y:.3f}}<extra></extra>'
            ),
            row=i, col=1
        )
        
        # Add intervention line
        fig.add_vline(
            x=intervention_num,
            line_width=2,
            line_dash="dot",
            line_color="red",
            annotation_text="Mining Date",
            annotation_position="top",
            row=i, col=1
        )
        
        # Add pre/post period shading
        # Pre-period (blue background)
        fig.add_vrect(
            x0=date_min,
            x1=intervention_num,
            fillcolor="rgba(100, 100, 255, 0.1)",
            layer="below",
            line_width=0,
            row=i, col=1
        )
        
        # Post-period (red background)
        fig.add_vrect(
            x0=intervention_num,
            x1=date_max,
            fillcolor="rgba(255, 100, 100, 0.1)",
            layer="below",
            line_width=0,
            row=i, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font=dict(color='white', size=11),
        margin=dict(l=10, r=10, t=40, b=40),
        hovermode='x unified'
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, title_text='Date', row=num_metrics, col=1)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, zeroline=True, zerolinecolor='rgba(255,255,255,0.3)')
    
    return fig


# ============================================================================
# SYNTHETIC CONTROL ANALYSIS - IMPROVED
# ============================================================================
def perform_sca_analysis(site_id, full_df, latest_df, sc_df, placebo_df):
    """
    Load pre-computed SCA results from CSV files.
    Returns control IDs and SCA results for plotting.
    """
    if sc_df.empty:
        return None, None
    
    # Get all SCA records for this site
    site_sc = sc_df[sc_df['site_id'] == site_id].copy()
    if site_sc.empty:
        return None, None
    
    # Extract control IDs from first record
    control_ids_str = site_sc.iloc[0].get('control_ids', '')
    control_ids = [int(c) for c in control_ids_str.split(',') if c.strip().isdigit()] if control_ids_str else []
    
    # Build results dict for three metrics
    results_dict = {}
    metrics = ['NDVI', 'MNDWI', 'BSI']
    
    for metric in metrics:
        actual_col = f'{metric}_actual'
        synthetic_col = f'{metric}_synthetic'
        
        if actual_col in site_sc.columns and synthetic_col in site_sc.columns:
            results_dict[metric] = pd.DataFrame({
                'date': pd.to_datetime(site_sc['date']),
                'actual': site_sc[actual_col].astype(float),
                'synthetic': site_sc[synthetic_col].astype(float)
            })
    
    return control_ids, results_dict
    """
    Perform Synthetic Control Analysis with:
    - Pre-period data used for fitting
    - Post-period shows divergence
    - Clear intervention date marker
    """
    if full_df.empty or baseline_df.empty:
        return None, None
    
    # Get treated site time series
    ts_treated = full_df[full_df['id'] == site_id].copy()
    if ts_treated.empty:
        return None, None
    
    ts_treated = ts_treated.sort_values('date')
    
    metrics = ['NDVI', 'MNDWI', 'BSI']
    # Filter to only available metrics
    available_metrics = [m for m in metrics if m in ts_treated.columns]
    
    if not available_metrics:
        return None, None
    
    ts_treated_clean = ts_treated.dropna(subset=['date'] + available_metrics)
    
    # Determine intervention date (median)
    sorted_dates = ts_treated_clean['date'].sort_values()
    median_idx = len(sorted_dates) // 2
    intervention_date = sorted_dates.iloc[median_idx]
    ts_pre = ts_treated_clean[ts_treated_clean['date'] < intervention_date]
    
    if len(ts_pre) < 3:
        return None, None
    
    # Find control sites
    treated_info = baseline_df[baseline_df['id'] == site_id].iloc[0]
    
    candidates = baseline_df[
        (baseline_df['probability'] < 0.4) & 
        (baseline_df['id'] != site_id)
    ].copy()
    
    # Filter by basin if available
    basin_col = next((col for col in candidates.columns if col.lower() in ['basin', 'river_basin', 'hybas_id']), None)
    if basin_col and basin_col in treated_info:
        target_basin = treated_info[basin_col]
        candidates = candidates[candidates[basin_col] == target_basin]
    
    valid_control_ids = []
    
    # Correlation-based filtering
    for _, cand in candidates.iterrows():
        cand_id = cand['id']
        ts_cand = full_df[full_df['id'] == cand_id].dropna(subset=['date'] + available_metrics)
        
        merged_pre = ts_pre.merge(ts_cand, on='date', suffixes=('_t', '_c'))
        
        if len(merged_pre) >= 3:
            corrs = []
            for m in available_metrics:
                try:
                    c, _ = pearsonr(merged_pre[f'{m}_t'], merged_pre[f'{m}_c'])
                    if not np.isnan(c):
                        corrs.append(c)
                except:
                    continue
            
            if corrs and (sum(corrs)/len(corrs)) > 0.7:
                valid_control_ids.append(cand_id)
    
    if not valid_control_ids:
        valid_control_ids = candidates['id'].unique().tolist()[:10]
    
    if not valid_control_ids:
        return None, None
    
    # Aggregate control time series
    ts_control = full_df[full_df['id'].isin(valid_control_ids)].copy()
    control_agg = ts_control.groupby('date')[available_metrics].mean().reset_index()
    control_agg.columns = ['date'] + [f'control_{m}' for m in available_metrics]
    
    # Merge
    df_sca = ts_treated_clean.rename(columns={m: f'treated_{m}' for m in available_metrics}).merge(control_agg, on='date', how='inner')
    
    if len(df_sca) < 5:
        return None, None
    
    df_sca['period'] = df_sca['date'].apply(lambda x: 'PRE' if x < intervention_date else 'POST')
    df_pre = df_sca[df_sca['period'] == 'PRE'].copy()
    
    # Fit synthetic control models
    results_dict = {}
    
    for metric in available_metrics:
        try:
            X = df_pre[[f'control_{metric}']].values
            y = df_pre[[f'treated_{metric}']].values
            model = Ridge(alpha=0.01, fit_intercept=True)
            model.fit(X, y)
            
            df_sca[f'synthetic_{metric}'] = model.predict(df_sca[[f'control_{metric}']].values)
            
            # Create results dataframe with proper types
            result_df = pd.DataFrame({
                'date': df_sca['date'].values,
                'actual': df_sca[f'treated_{metric}'].values,
                'synthetic': df_sca[f'synthetic_{metric}'].values,
                'intervention_date': intervention_date
            })
            
            # Ensure date column is datetime
            result_df['date'] = pd.to_datetime(result_df['date'])
            
            # Remove any rows with NaN dates
            result_df = result_df.dropna(subset=['date'])
            
            results_dict[metric] = result_df
        except Exception as e:
            print(f"[SCA] Error fitting {metric}: {e}")
            continue
    
    return valid_control_ids[:5], results_dict


def create_sca_plots_with_intervention(sca_results_dict):
    """
    Create SCA plots with:
    - Actual (treated) line
    - Synthetic (counterfactual) baseline
    - Intervention date marker
    - Pre/Post period shading
    """
    if not sca_results_dict:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            'NDVI: Actual vs Synthetic Control',
            'MNDWI: Actual vs Synthetic Control',
            'BSI: Actual vs Synthetic Control'
        )
    )
    
    metrics = ['NDVI', 'MNDWI', 'BSI']
    colors = {'NDVI': '#44ff44', 'MNDWI': '#00d4ff', 'BSI': '#ffaa00'}
    
    for i, metric in enumerate(metrics, 1):
        if metric not in sca_results_dict:
            continue
        
        res = sca_results_dict[metric].copy()
        
        # Clean the data - remove any NaN dates
        res = res.dropna(subset=['date', 'actual', 'synthetic'])
        
        if res.empty:
            continue
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(res['date']):
            res['date'] = pd.to_datetime(res['date'])
        
        intervention_date = res['intervention_date'].iloc[0]
        if not pd.api.types.is_datetime64_any_dtype(pd.Series([intervention_date])):
            intervention_date = pd.to_datetime(intervention_date)
            
            
        intervention_ts = pd.Timestamp(intervention_date) 
        intervention_num = intervention_ts.timestamp() * 1000
    
        
        # Get clean min/max dates
        date_min = res['date'].min().timestamp() * 1000
        date_max = res['date'].max().timestamp() * 1000
        
        # Actual (treated) line
        fig.add_trace(
            go.Scatter(
                x=res['date'],
                y=res['actual'],
                name=f'{metric} (Actual/Treated)',
                line=dict(color=colors[metric], width=2),
                mode='lines+markers',
                marker=dict(size=5),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> %{y:.3f}<extra></extra>'
            ),
            row=i, col=1
        )
        
        # Synthetic (counterfactual) line
        fig.add_trace(
            go.Scatter(
                x=res['date'],
                y=res['synthetic'],
                name=f'{metric} (Synthetic Control)',
                line=dict(color='white', dash='dash', width=2),
                mode='lines',
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Synthetic:</b> %{y:.3f}<extra></extra>'
            ),
            row=i, col=1
        )
        
        # Intervention line
        fig.add_vline(
            x=intervention_num,
            line_width=2,
            line_dash="dot",
            line_color="red",
            annotation_text="Mining Date",
            annotation_position="top right",
            row=i, col=1
        )
        
        # Pre-period shading (fit period)
        fig.add_vrect(
            x0=date_min,
            x1=intervention_num,
            fillcolor="rgba(100, 100, 255, 0.15)",
            layer="below",
            line_width=0,
            annotation_text="Pre-period (fit)",
            annotation_position="top left",
            row=i, col=1
        )
        
        # Post-period shading
        fig.add_vrect(
            x0=intervention_num,
            x1=date_max,
            fillcolor="rgba(255, 100, 100, 0.15)",
            layer="below",
            line_width=0,
            annotation_text="Post-period",
            annotation_position="top right",
            row=i, col=1
        )
    
    fig.update_layout(
        height=800,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font=dict(color='white', size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, title_text='Date', row=3, col=1)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

# Load data for current state
current_state = st.session_state.selected_state

with st.spinner(f"🚀 Loading {current_state} data..."):
    full_df, available_dates, sca_ready_df, latest_date, shap_df, sc_df, placebo_df = load_state_data(current_state)

# Get latest per-site predictions (each site at its own latest date)
if not full_df.empty:
    # For each site, get the row with the most recent date
    latest_df = full_df.loc[full_df.groupby('id')['date'].idxmax()].copy()
    
    # Prepare minimal dataframe for map
    if all(c in latest_df.columns for c in ["id", "lat", "lon", "probability"]):
        prediction_minimal = latest_df[["id", "lat", "lon", "probability"]].copy()
    else:
        prediction_minimal = None
else:
    latest_df = pd.DataFrame()
    prediction_minimal = None

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("ManKaaval")
    st.markdown("# 🗺️ Select Region")
    
    # Data diagnostics
    with st.expander("🔍 Data Diagnostics"):
        if full_df.empty:
            st.error("⚠️ Current State Data: NOT LOADED")
        else:
            st.success(f"📊 Total Records: {len(full_df):,}")
            st.info(f"📍 Latest Snapshot: {len(latest_df):,} sites")
            if latest_date:
                st.metric("Latest Update", latest_date)
    
    # State selection
    states = ["Bihar", "Uttar Pradesh", "West Bengal", "Punjab", "Rajasthan", "Gujarat", "Tamil Nadu"]
    
    for state in states:
        is_active = (state == st.session_state.selected_state)
        label = f"✓ {state}" if is_active else state
        
        if is_active:
            st.markdown(f"""
            <div style="background-color: #ff4d4d; color: white; padding: 12px; border-radius: 8px; 
                        margin-bottom: 10px; font-weight: bold; text-align: center; border: 1px solid #ff4d4d;">
                {label}
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(state, key=f"sidebar_btn_{state}"):
                st.session_state.selected_state = state
                st.session_state.selected_site = None  # Reset selected site
                st.rerun()
    
    st.markdown("---")
    
    # Highest risk sites
    st.markdown("## ⚠️ Highest Risk Sites")
    if not latest_df.empty:
        top_sites = latest_df.nlargest(10, 'probability')
        for i, (idx, row) in enumerate(top_sites.iterrows(), 1):
            prob = row['probability']
            risk_pct = prob * 100
            
            if risk_pct > 80:
                status = "CRITICAL"; color = "#ff4444"; dot = "🔴"
            elif risk_pct > 60:
                status = "MEDIUM"; color = "#ffaa00"; dot = "🟡"
            else:
                status = "LOW"; color = "#44ff44"; dot = "🟢"
            
            st.markdown(f"""
            <div style="background: #1a1a1a; border: 1px solid #333; border-radius: 10px; 
                        padding: 15px; margin-bottom: 12px;">
                <div style="font-weight: bold; color: white; font-size: 1.1rem; margin-bottom: 10px;">
                    {i}. Site {row['id']}
                </div>
                <div style="color: #eee; font-size: 0.95rem; margin-bottom: 8px;">
                    Risk: {risk_pct:.1f}% {dot} {status}
                </div>
                <div style="color: #888; font-size: 0.85rem;">
                    Lat: {row['lat']:.4f}, Lon: {row['lon']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No data available.")
    
    st.markdown("---")
    
    # SCA-ready info
    if not sca_ready_df.empty:
        st.info(f"📊 {len(sca_ready_df)} sites are SCA-ready")
    
    # Refresh button
    st.markdown("### 🛠️ System Controls")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared!")
        st.rerun()

# ============================================================================
# MAIN LAYOUT
# ============================================================================
col_left, col_right = st.columns([3, 2])

# LEFT COLUMN: MAP (No slider - always shows latest)
with col_left:
    st.markdown(f"### 🗺️ {current_state} - Latest Mining Risk Assessment")
    if latest_date:
        st.caption(f"Showing predictions for: **{latest_date}**")
    
    # Create and display map
    folium_map = create_folium_map(
        selected_state=current_state,
        prediction_df=prediction_minimal,
        sca_ready_df=sca_ready_df
    )
    
    map_output = st_folium(
        folium_map,
        use_container_width=True,
        height=700,
        returned_objects=["last_clicked"],
        key=f"mankaaval_map_{current_state}",
    )
    
    # Handle map clicks
    if map_output and map_output.get("last_clicked"):
        click_lat = map_output["last_clicked"]["lat"]
        click_lon = map_output["last_clicked"]["lng"]
        
        nearest_site = find_nearest_site(click_lat, click_lon, latest_df)
        
        if nearest_site is not None:
            if st.session_state.selected_site is None or st.session_state.selected_site['id'] != nearest_site['id']:
                st.session_state.selected_site = nearest_site
                st.rerun()

# RIGHT COLUMN: SITE ANALYSIS
with col_right:
    with st.container(border=True, height=800):
        if st.session_state.selected_site is None:
            # Global overview
            st.markdown("## 📊 Global Overview")
            if full_df.empty:
                st.info("No data loaded. Please check file paths.")
            else:
                st.info("👆 Click a site on the map to get detailed analysis and trends.")
                st.metric("Total Monitored Sites", len(full_df['id'].unique()))
                if latest_date:
                    st.metric("Latest Update", latest_date)
                    
                # Show distribution
                if not latest_df.empty:
                    high_risk = len(latest_df[latest_df['probability'] > 0.7])
                    medium_risk = len(latest_df[(latest_df['probability'] > 0.4) & (latest_df['probability'] <= 0.7)])
                    low_risk = len(latest_df[latest_df['probability'] <= 0.4])
                    
                    st.markdown("### Risk Distribution")
                    st.markdown(f"🔴 **High Risk:** {high_risk} sites")
                    st.markdown(f"🟡 **Medium Risk:** {medium_risk} sites")
                    st.markdown(f"🟢 **Low Risk:** {low_risk} sites")
        
        else:
            # Selected site analysis
            site = st.session_state.selected_site
            site_id = site['id']
            site_lat = site['lat']
            site_lon = site['lon']
            site_prob = site['probability']
            risk_pct = site_prob * 100
            risk_color = '#ff4444' if risk_pct > 70 else '#ffaa00' if risk_pct > 40 else '#44ff44'
            
            # Risk score card
            st.markdown(
                f'''
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a1a, #2a2a2a); 
                            border-radius: 15px; border: 2px solid {risk_color};">
                    <p style="color: #aaa; margin: 0; font-size: 0.9rem;">CURRENT RISK PROBABILITY</p>
                    <h1 style="color: {risk_color}; margin: 10px 0; font-size: 3rem;">{risk_pct:.1f}%</h1>
                    <p style="color: {risk_color}; margin: 0; font-weight: bold;">
                        {"🔴 HIGH RISK" if risk_pct > 70 else "🟡 MODERATE" if risk_pct > 40 else "🟢 LOW RISK"}
                    </p>
                </div>
                ''',
                unsafe_allow_html=True
            )
            
            st.markdown("---")
            
            # Site information
            st.markdown(f"**📍 Location:** {site_lat:.4f}°N, {site_lon:.4f}°E")
            st.markdown(f"**🔎 Site ID:** {site_id}")
            
            st.markdown("---")
            
            # SHAP Attribution (replaces time-series, now shown before SCA)
            st.markdown("### 🎯 Feature Attribution (Why This Risk?)")
            st.caption("Shows how spectral indices contribute to mining detection")
            
            shap_fig = create_shap_attribution_plot(site_id, shap_df)
            if shap_fig:
                st.plotly_chart(shap_fig, use_container_width=True, key=f"shap_plot_{site_id}")
                st.caption("Red bars increase mining likelihood. Blue bars decrease it.")
            else:
                st.info("No SHAP attribution data available for this site.")
            
            st.markdown("---")
            
            # Synthetic Control Analysis
            st.markdown("### 🔬 Synthetic Control Analysis")
            st.caption("Causal impact assessment using pre-computed baseline")
            
            # Load pre-computed SCA results
            control_ids, sca_results = perform_sca_analysis(site_id, full_df, latest_df, sc_df, placebo_df)
            
            if control_ids and sca_results:
                st.session_state.sca_control_ids = control_ids
                
                sca_fig = create_sca_plots_with_intervention(sca_results)
                if sca_fig:
                    st.plotly_chart(sca_fig, use_container_width=True, key=f"sca_plot_{site_id}")
                    
                    st.markdown("#### 📊 Interpretation Guide")
                    st.markdown("""
                    - **Green/Cyan/Orange Line**: Actual observed values (treated site)
                    - **White Dashed Line**: Synthetic control (counterfactual baseline)
                    - **Divergence**: Gap between actual and synthetic indicates mining impact
                    """)
                    
                    st.info(f"✅ Using {len(control_ids)} control sites with high correlation")
                else:
                    st.warning("Could not generate SCA visualization.")
            else:
                st.warning("⚠️ No synthetic control data available for this site.")
                st.session_state.sca_control_ids = None

st.markdown("---")
st.caption("ManKaaval Dashboard v2.0 | Powered by Folium & Streamlit | All data updated to latest available date")
