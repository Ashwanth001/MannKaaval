import streamlit as st
import pandas as pd
import numpy as np
import ee
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import xgboost as xgb
import time
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ManKaaval: Illegal Sand Mining Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
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
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'selected_site' not in st.session_state:
    st.session_state.selected_site = None
if 'aoi_coords' not in st.session_state:
    st.session_state.aoi_coords = None
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'inference_running' not in st.session_state:
    st.session_state.inference_running = False
if 'sca_control_ids' not in st.session_state:
    st.session_state.sca_control_ids = None
if 'last_click_coords' not in st.session_state:
    st.session_state.last_click_coords = None

# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================

def load_data():
    """Load predictions and time-series data from CSV files"""
    try:
        # Load baseline features with predictions
        baseline_path = "dashboardData/baseline_features_predictions.csv"
        if not os.path.exists(baseline_path):
            st.error(f" Baseline features file not found at: {baseline_path}")
            return pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = pd.read_csv(baseline_path)

        # Validate required columns
        required_cols = ['id', 'lat', 'lon', 'probability']
        if not all(col in df.columns for col in required_cols):
            st.error(f" Dataset missing required columns: {required_cols}")
            st.stop()

        # Clean data - remove NaN values
        df = df.dropna(subset=['lat', 'lon', 'probability'])

        # Load time-series data
        ts_path = "dashboardData/timeseries_features.csv"
        ts_df = pd.DataFrame()
        available_dates = []

        if os.path.exists(ts_path):
            ts_df = pd.read_csv(ts_path)
            ts_df['date'] = pd.to_datetime(ts_df['date'])
            available_dates = sorted(ts_df['date'].dt.strftime('%Y-%m-%d').unique().tolist(), reverse=True)
        else:
            st.warning(f" Time-series file not found at: {ts_path}")

        # Load SHAP values
        shap_path = "dashboardData/shap_values_baseline_features.csv"
        shap_df = pd.DataFrame()
        if os.path.exists(shap_path):
            shap_df = pd.read_csv(shap_path)
        else:
            st.warning(f" SHAP values file not found at: {shap_path}")

        # Load ground truth data
        gt_path = "dataset/cleanedGroundTruth2_indexed.csv"
        gt_df = pd.DataFrame()
        if os.path.exists(gt_path):
            gt_df = pd.read_csv(gt_path)
        else:
            st.warning(f" Ground Truth file not found at: {gt_path}")

        #  LOAD SCA-READY SITES CSV
        sca_ready_path = "dashboardData/sca_ready_sites.csv"
        sca_ready_df = pd.DataFrame()
        if os.path.exists(sca_ready_path):
            sca_ready_df = pd.read_csv(sca_ready_path)
            st.success(f" Loaded {len(sca_ready_df):,} SCA-ready sites")
        else:
            st.warning(f" SCA-ready sites file not found. Run generate_sca_ready_sites.py first.")

        return df, ts_df, available_dates, shap_df, gt_df, sca_ready_df

    except Exception as e:
        st.error(f" Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# ============================================================================
# GOOGLE EARTH ENGINE INITIALIZATION
# ============================================================================

def initialize_gee():
    """Initialize Google Earth Engine with service account for cloud deployment"""
    try:
        # Check if running on Streamlit Cloud (secrets available)
        if "gee" in st.secrets:
            # Use service account authentication (for deployment)
            credentials = ee.ServiceAccountCredentials(
                email=st.secrets["gee"]["service_account"],
                key_data=st.secrets["gee"]["private_key"]
            )
            ee.Initialize(
                credentials=credentials,
                project=st.secrets["gee"]["project"]
            )
            return True, " GEE initialized with service account"
        else:
            # Local development - use regular auth
            ee.Initialize(project="sandminingproject")
            return True, " GEE initialized (local)"
            
    except Exception as e:
        error_msg = f" GEE initialization failed: {str(e)}"
        st.error(error_msg)
        return False, error_msg


# ============================================================================
# MAP CREATION WITH 1KM GRID (SIMPLIFIED - REMOVED REDUNDANT LAYERS)
# ============================================================================

def get_grid_feature_collection(_df):
    """Cache the conversion of the large DataFrame to GEE FeatureCollection."""
    if _df is None or _df.empty:
        return None

    features = []
    for row in _df.itertuples():
        # Create centered 1km x 1km square around each point
        lon = float(row.lon)
        lat = float(row.lat)
        # Calculate approximate 1km offset in degrees
        offset = 0.0045
        square = ee.Geometry.Polygon([[
            [lon - offset, lat - offset],
            [lon + offset, lat - offset],
            [lon + offset, lat + offset],
            [lon - offset, lat + offset],
            [lon - offset, lat - offset]
        ]])
        features.append(ee.Feature(square, {
            'probability': float(row.probability),
            'site_id': str(row.id),
            'lon': lon,
            'lat': lat
        }))

    return ee.FeatureCollection(features)


def get_sca_ready_layer(_sca_ready_df):
    """
    Create a GEE layer highlighting sites with sufficient time-series data.
    Returns a visualization layer showing SCA-ready sites with green outlines.
    """
    if _sca_ready_df is None or _sca_ready_df.empty:
        return None
    
    features = []
    for row in _sca_ready_df.itertuples():
        lon = float(row.lon)
        lat = float(row.lat)
        
        # Create 1km square
        offset = 0.0045
        square = ee.Geometry.Polygon([[
            [lon - offset, lat - offset],
            [lon + offset, lat - offset],
            [lon + offset, lat + offset],
            [lon - offset, lat + offset],
            [lon - offset, lat - offset]
        ]])
        
        features.append(ee.Feature(square, {
            'site_id': str(row.id),
            'timepoints': int(row.timepoint_count) if hasattr(row, 'timepoint_count') else 0
        }))
    
    fc = ee.FeatureCollection(features)
    
    # Create outline visualization (green borders)
    outline = ee.Image().byte().paint(
        featureCollection=fc,
        color=1,
        width=3  # Border width in pixels
    )
    
    # Visualize as bright green outlines
    return outline.visualize(**{
        'palette': ['00ff00'],  # Bright green
        'opacity': 1.0
    })



def create_gee_river_grid(_df, gee_ready, selected_date=None, show_coords=False, aoi_coords=None, 
                          _prediction_df=None, _ground_truth_df=None, _sca_ready_df=None):
    """Create a Google Earth Engine map with 1km grid - CLEAN VERSION (no redundant layers)"""

    # Base map initialization
    center = [25.5, 85.5]
    zoom = 7
    if aoi_coords:
        lats = [c[1] for c in aoi_coords]
        lons = [c[0] for c in aoi_coords]
        center = [sum(lats)/len(lats), sum(lons)/len(lons)]
        zoom = 12

    m = geemap.Map(
        center=center,
        zoom=zoom,
        draw_control=True,
        measure_control=False,
        fullscreen_control=True,
        attribution_control=True
    )

    if not gee_ready:
        m.add_basemap("SATELLITE")
        return m

    try:
        #  ADD ONLY SATELLITE BASEMAP - NO OPENSTREETMAP
        m.add_basemap("SATELLITE")

        if aoi_coords:
            roi = ee.Geometry.Polygon([list(aoi_coords)])
        else:
            roi = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq('ADM1_NAME', 'Bihar'))

        # ===== RIVER MASKING =====
        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        occurrence = gsw.select('occurrence')
        river_mask = occurrence.gt(10)
        connected_pixels = river_mask.selfMask().connectedPixelCount(1024)
        river_mask_clean = river_mask.updateMask(connected_pixels.gte(500))
        river_mask_final = river_mask_clean.clip(roi)
        buffered_river_mask = river_mask_final.focal_max(radius=100, units='meters', kernelType='circle')

        # ===== 1KM GRID GENERATION WITH ACTUAL PREDICTIONS =====
        if _prediction_df is not None and not _prediction_df.empty:
            # RETRIEVE CACHED FEATURE COLLECTION
            sites_fc = get_grid_feature_collection(_prediction_df)
            # Create image from polygons
            prob_image = ee.Image().float().paint(sites_fc, 'probability')
            prob_image = prob_image.clip(roi)
        else:
            # Fallback to random for preview
            proj = ee.Projection('EPSG:3857').atScale(1000)
            prob_image = ee.Image.random(seed=99).reproject(proj).clip(roi)

        # Apply river mask
        grid_masked = prob_image.updateMask(buffered_river_mask)

        # Visualization - ONLY MINING RISK LAYER (no grid boundaries)
        fill_vis = grid_masked.visualize(**{
            'min': 0,
            'max': 1,
            'palette': ['00ff00', 'ffff00', 'ff0000'],
            'opacity': 0.6
        })

        m.addLayer(fill_vis, {}, "Mining Activity Risk (1km Grid)", True)

        #  ADD SCA-READY SITES LAYER (green outlines)
        if _sca_ready_df is not None and not _sca_ready_df.empty:
            sca_layer = get_sca_ready_layer(_sca_ready_df)
            if sca_layer:
                m.addLayer(
                    sca_layer, 
                    {}, 
                    f"SCA-Ready Sites ({len(_sca_ready_df)} sites with 5+ timepoints)", 
                    True  # Visible by default
                )

        #  ADD AOI BOUNDARY IF PROVIDED
        if aoi_coords:
            m.addLayer(roi.style(fillColor='00000000', color='yellow', width=2), {}, "Analyzed AOI Bound")

        m.add_layer_control()
        return m

    except Exception as e:
        import traceback
        st.error(f" Error creating GEE layers: {e}")
        st.code(traceback.format_exc())
        m.add_basemap("SATELLITE")
        return m


# ============================================================================
# SITE SELECTION LOGIC
# ============================================================================
def find_nearest_site(click_lat, click_lon, df, threshold=0.006):
    """Find nearest site to clicked location"""
    if df.empty:
        return None

    coords = df[['lat', 'lon']].values
    click_point = np.array([click_lat, click_lon])
    distances = np.linalg.norm(coords - click_point, axis=1)
    nearest_idx = np.argmin(distances)
    min_distance = distances[nearest_idx]

    nearest_site = df.iloc[nearest_idx]
    print(f"Click: ({click_lat:.4f}, {click_lon:.4f})")
    print(f"Nearest: {nearest_site['id']} at ({nearest_site['lat']:.4f}, {nearest_site['lon']:.4f})")
    print(f"Distance: {min_distance:.6f}° ({min_distance*111:.1f}km)")
    print(f"Probability: {nearest_site['probability']:.3f}")

    if min_distance < threshold:
        return nearest_site

    return None

# ============================================================================
# TIME SERIES VISUALIZATION
# ============================================================================
def create_timeseries_plot(site_id, ts_df):
    """Create interactive time-series plots for NDVI, MNDWI, and BSI"""
    # Filter data for the selected site
    site_ts = ts_df[ts_df['id'] == site_id].copy()
    if site_ts.empty:
        return None

    # Sort by date
    site_ts = site_ts.sort_values('date')

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('NDVI (Vegetation Index)', 'MNDWI (Water Index)', 'BSI (Bare Soil Index)'),
        vertical_spacing=0.12,
        shared_xaxes=True
    )

    # NDVI trace
    fig.add_trace(
        go.Scatter(
            x=site_ts['date'],
            y=site_ts['NDVI'],
            mode='lines+markers',
            name='NDVI',
            line=dict(color='#44ff44', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>NDVI:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # MNDWI trace
    fig.add_trace(
        go.Scatter(
            x=site_ts['date'],
            y=site_ts['MNDWI'],
            mode='lines+markers',
            name='MNDWI',
            line=dict(color='#00d4ff', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>MNDWI:</b> %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # BSI trace
    fig.add_trace(
        go.Scatter(
            x=site_ts['date'],
            y=site_ts['BSI'],
            mode='lines+markers',
            name='BSI',
            line=dict(color='#ffaa00', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>BSI:</b> %{y:.3f}<extra></extra>'
        ),
        row=3, col=1
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

    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, title_text='Date', row=3, col=1)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, zeroline=True, zerolinecolor='rgba(255,255,255,0.3)')

    return fig

# ============================================================================
# SHAP VISUALIZATION
# ============================================================================
def create_shap_plot(site_id, shap_df):
    """Create a bar chart showing feature importance/attribution for a specific site"""
    # Filter for the specific site
    site_shap = shap_df[shap_df['id'] == site_id].copy()
    if site_shap.empty:
        return None

    # Columns to drop
    cols_to_drop = ['id', 'lat', 'lon', 'probability', 'prediction']
    features_shap = site_shap.drop(columns=[c for c in cols_to_drop if c in site_shap.columns])

    # Transpose and sort
    plot_data = features_shap.melt(var_name='Feature', value_name='SHAP Value')
    plot_data = plot_data.sort_values(by='SHAP Value', ascending=True)

    # Define colors: Red for positive, Blue for negative
    plot_data['color'] = plot_data['SHAP Value'].apply(lambda x: '#ff4444' if x > 0 else '#00d4ff')

    fig = go.Figure(go.Bar(
        x=plot_data['SHAP Value'],
        y=plot_data['Feature'],
        orientation='h',
        marker_color=plot_data['color'],
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title='Feature Attribution (SHAP)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font=dict(color='white', size=10),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
        xaxis_title='Impact on Prediction',
        yaxis_gridcolor='rgba(128,128,128,0.2)'
    )

    return fig

def perform_sca_analysis(site_id, baseline_df, ts_df):
    """
    Perform SITE-SPECIFIC Synthetic Control Analysis:
    - Treat the CLICKED SITE as the treated unit
    - Find similar control sites (low-risk)
    - Build synthetic control for THIS specific site
    """
    
    # 🔍 DEBUG BLOCK
    if "sca_debug_counter" not in st.session_state:
        st.session_state.sca_debug_counter = 0
        st.session_state.sca_last_site = None
        st.session_state.sca_last_time = None

    st.session_state.sca_debug_counter += 1
    st.session_state.sca_last_site = site_id
    st.session_state.sca_last_time = time.strftime("%H:%M:%S")

    print(f"[SCA DEBUG] Run #{st.session_state.sca_debug_counter} for site {site_id}")
    
    if ts_df.empty or baseline_df.empty:
        return None, None, None
    
    # Ensure date is datetime
    ts_df = ts_df.copy()
    ts_df['date'] = pd.to_datetime(ts_df['date'])
    
    # ===== STEP 1: Get time series for THIS SPECIFIC SITE (treated) =====
    ts_treated = ts_df[ts_df['id'] == site_id].copy()  # ONLY the clicked site
    
    if ts_treated.empty:
        print(f"[SCA] No time series data for site {site_id}")
        return None, None, None
    
    print(f"[SCA] Treated site {site_id}: {len(ts_treated)} time points")
    
    # ===== STEP 2: Find control sites (low probability, NOT the clicked site) =====
    control_site_ids = baseline_df[
        (baseline_df['probability'] < 0.4) & 
        (baseline_df['id'] != site_id)
    ]['id'].unique()
    
    if len(control_site_ids) == 0:
        print("[SCA] No control sites found")
        return None, None, None
    
    print(f"[SCA] Found {len(control_site_ids)} control sites")
    
    # ===== STEP 3: Get control time series and aggregate =====
    ts_control = ts_df[ts_df['id'].isin(control_site_ids)].copy()
    
    if ts_control.empty:
        print("[SCA] No control time series")
        return None, None, None
    
    # Clean data
    ts_treated_clean = ts_treated[['date', 'NDVI', 'MNDWI', 'BSI']].dropna()
    ts_control_clean = ts_control[['date', 'NDVI', 'MNDWI', 'BSI']].dropna()
    
    # Aggregate ONLY controls (treated is already a single site)
    control_agg = ts_control_clean.groupby('date')[['NDVI', 'MNDWI', 'BSI']].mean().reset_index()
    control_agg = control_agg.rename(columns={
        'NDVI': 'control_NDVI', 
        'MNDWI': 'control_MNDWI', 
        'BSI': 'control_BSI'
    })
    
    # Rename treated columns
    treated_agg = ts_treated_clean[['date', 'NDVI', 'MNDWI', 'BSI']].rename(columns={
        'NDVI': 'treated_NDVI',
        'MNDWI': 'treated_MNDWI',
        'BSI': 'treated_BSI'
    })
    
    # Merge
    df_sca = treated_agg.merge(control_agg, on='date', how='inner')
    
    if len(df_sca) < 5:
        print(f"[SCA] Insufficient data points: {len(df_sca)}")
        return None, None, None
    
    print(f"[SCA] Merged data: {len(df_sca)} dates")
    
    # ===== STEP 4: Define pre/post treatment period =====
    median_date = df_sca['date'].median()
    df_sca['period'] = df_sca['date'].apply(lambda x: 'PRE' if x < median_date else 'POST')
    
    df_pre = df_sca[df_sca['period'] == 'PRE'].copy()
    df_post = df_sca[df_sca['period'] == 'POST'].copy()
    
    if len(df_pre) < 3 or len(df_post) < 3:
        print(f"[SCA] Insufficient pre/post data: {len(df_pre)}/{len(df_post)}")
        return None, None, None
    
    # ===== STEP 5: Fit synthetic control =====
    metrics = ['NDVI', 'MNDWI', 'BSI']
    weights = {}
    
    for metric in metrics:
        try:
            X = df_pre[[f'control_{metric}']].values
            y = df_pre[[f'treated_{metric}']].values
            
            model = Ridge(alpha=0.01, fit_intercept=True)
            model.fit(X, y)
            
            weight = model.coef_.flatten()[0]
            weights[metric] = weight
            
            print(f"[SCA] {metric}: weight={weight:.4f}")
            
        except Exception as e:
            print(f"[SCA] Error fitting {metric}: {e}")
            weights[metric] = 1.0
    
    # ===== STEP 6: Create synthetic counterfactual =====
    for metric in metrics:
        df_sca[f'synthetic_{metric}'] = weights[metric] * df_sca[f'control_{metric}']
        df_sca[f'effect_{metric}'] = df_sca[f'treated_{metric}'] - df_sca[f'synthetic_{metric}']
    
    # ===== STEP 7: Prepare results =====
    results_dict = {}
    for metric in metrics:
        metric_res = pd.DataFrame({
            'date': df_sca['date'],
            'actual': df_sca[f'treated_{metric}'],
            'synthetic': df_sca[f'synthetic_{metric}']
        })
        results_dict[metric] = metric_res
    
    # ===== STEP 8: Calculate Attributable Damage (Impact) =====
    damage_dict = {}
    for metric in metrics:
        # Damage is the cumulative sum of (Actual - Synthetic) in the POST period
        post_data = df_sca[df_sca['period'] == 'POST']
        damage_dict[metric] = post_data[f'effect_{metric}'].sum()
    
    return control_site_ids, results_dict, damage_dict


def create_sca_plots(sca_results_dict, damage_dict=None):
    """Create three subplots for SCA (NDVI, MNDWI, BSI) - Actual vs Synthetic"""
    if not sca_results_dict:
        return None

    metrics = ['NDVI', 'MNDWI', 'BSI']
    
    # Generate dynamic titles with Attributable Damage
    subplot_titles = []
    for m in metrics:
        title = f'{m} - Actual vs Synthetic'
        if damage_dict and m in damage_dict:
            damage = damage_dict[m]
            title += f' (Attributable Damage: {damage:+.3f})'
        subplot_titles.append(title)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles
    )

    metrics = ['NDVI', 'MNDWI', 'BSI']
    colors = {'NDVI': '#44ff44', 'MNDWI': '#00d4ff', 'BSI': '#ffaa00'}

    for i, metric in enumerate(metrics, 1):
        if metric not in sca_results_dict:
            continue
        
        res = sca_results_dict[metric]
        
        # Actual line
        fig.add_trace(
            go.Scatter(
                x=res['date'],
                y=res['actual'],
                name=f'{metric} (Actual)',
                line=dict(color=colors[metric], width=2),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> %{y:.3f}<extra></extra>'
            ),
            row=i, col=1
        )
        
        # Synthetic line
        fig.add_trace(
            go.Scatter(
                x=res['date'],
                y=res['synthetic'],
                name=f'{metric} (Synthetic)',
                line=dict(color='white', dash='dash', width=1.5),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Synthetic:</b> %{y:.3f}<extra></extra>'
            ),
            row=i, col=1
        )

    fig.update_layout(
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font=dict(color='white', size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified'
    )

    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, title_text='Date', row=3, col=1)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True)

    return fig

# ============================================================================
# MAIN APP
# ============================================================================

# Title and header
st.markdown("# ManKaaval: Illegal Sand Mining Detection")
st.markdown("Real-Time Satellite Surveillance System")

# Load data
df, ts_df, available_dates, shap_df, gt_df, sca_ready_df = load_data()



# Initialize GEE
gee_ready, gee_msg = initialize_gee()
if not gee_ready:
    st.warning(gee_msg)

# Create layout
col_left, col_right = st.columns([3, 2])

# Display SCA statistics
if not sca_ready_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 SCA Analysis Status")
    st.sidebar.metric("SCA-Ready Sites", f"{len(sca_ready_df):,}")
    st.sidebar.metric("Total Sites", f"{len(df):,}")
    if len(df) > 0:
        st.sidebar.caption(f"{(len(sca_ready_df)/len(df)*100):.1f}% have sufficient time-series data")

# ============================================================================
# LEFT COLUMN: INTERACTIVE MAP
# ============================================================================

with col_left:
    st.markdown("## 🛰️ Real-Time Satellite Surveillance (1km Grid)")
    st.markdown("*Click on any grid cell to view detailed analysis and trends*")
    
    if not sca_ready_df.empty:
        st.caption("**Green outlined sites** have sufficient time-series data for SCA analysis")

    #  ONLY CREATE MAP ONCE - Cache the map object itself
    if 'gee_map' not in st.session_state or st.session_state.get('force_map_refresh', False):
        with st.spinner("Loading satellite imagery and predictions..."):
            st.session_state.gee_map = create_gee_river_grid(
                df, gee_ready, None, False,
                aoi_coords=None,
                _prediction_df=df if not df.empty else None,
                _ground_truth_df=None,
                _sca_ready_df=sca_ready_df if not sca_ready_df.empty else None
            )
        st.session_state.force_map_refresh = False

    #  RENDER CACHED MAP (fast!)
    map_output = st_folium(
        st.session_state.gee_map,  # ← Use cached map
        width="100%",
        height=600,
        returned_objects=["last_clicked"],
        key="mankaaval_map"
    )

    # Handle click events
    if map_output and map_output.get("last_clicked"):
        click_lat = map_output["last_clicked"]["lat"]
        click_lon = map_output["last_clicked"]["lng"]

        last_click = st.session_state.get('last_click_coords')
        current_click = (click_lat, click_lon)

        if last_click != current_click:
            st.session_state.last_click_coords = current_click

            selected = find_nearest_site(click_lat, click_lon, df)

            if selected is not None:
                st.session_state.selected_site = selected
            else:
                st.session_state.selected_site = None
                st.warning(" No mining site found at this location. Click closer to a grid cell.")


# ============================================================================
# RIGHT COLUMN: SITE ANALYSIS
# ============================================================================
with col_right:
    
    if st.session_state.selected_site is None:
        # Global Overview
        st.markdown("## Global Overview")
        if df.empty:
            st.info("No data loaded. Please check file paths.")
        else:
            total_sites = len(df)
            high_risk = df[df['probability'] > 0.7]
            high_risk_count = len(high_risk)
            avg_risk = df['probability'].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sites", f"{total_sites:,}")
            with col2:
                st.metric("High Risk", f"{high_risk_count:,}", delta=f"{(high_risk_count/total_sites)*100:.1f}%", delta_color="inverse")

            st.metric("Average Risk", f"{avg_risk*100:.1f}%")

            st.markdown("---")
            st.markdown("### Risk Distribution")
            fig_dist = px.histogram(df, x='probability', nbins=30, color_discrete_sequence=['#00d4ff'])
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,0.5)',
                font_color='white',
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("---")
            st.markdown("### Top Risk Sites")
            top_5 = df.nlargest(5, 'probability')[['id', 'probability']]
            for idx, row in top_5.iterrows():
                risk_pct = row['probability'] * 100
                color = '#ff4444' if risk_pct > 80 else '#ffaa00' if risk_pct > 60 else '#ffff00'
                st.markdown(
                    f'<div class="risk-card"><b>{row["id"]}</b>: <span style="color:{color}">{risk_pct:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

    else:
        # Selected Site Analysis
        site = st.session_state.selected_site
        site_id = site['id']
        site_lat = site['lat']
        site_lon = site['lon']
        site_prob = site['probability']
        risk_pct = site_prob * 100
        risk_color = '#ff4444' if risk_pct > 70 else '#ffaa00' if risk_pct > 40 else '#44ff44'

        # Risk Score Card
        st.markdown(
            f'''
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a1a, #2a2a2a); border-radius: 15px; border: 2px solid {risk_color};">
                <p style="color: #aaa; margin: 0; font-size: 0.9rem;">RISK PROBABILITY</p>
                <h1 style="color: {risk_color}; margin: 10px 0; font-size: 3rem;">{risk_pct:.1f}%</h1>
                <p style="color: {risk_color}; margin: 0; font-weight: bold;">
                    {"🔴 HIGH RISK" if risk_pct > 70 else "🟡 MODERATE" if risk_pct > 40 else "🟢 LOW RISK"}
                </p>
            </div>
            ''',
            unsafe_allow_html=True
        )


        st.markdown("---")

        # SHAP Attribution
        st.markdown("### 🎯 Why This Risk?")
        if not shap_df.empty:
            shap_fig = create_shap_plot(site_id, shap_df)
            if shap_fig:
                st.plotly_chart(shap_fig, use_container_width=True, key=f"shap_plot_{site_id}")
                st.caption("Positive values (red) increase mining detection likelihood. Negative values (blue) decrease it.")
            else:
                st.info("No SHAP attribution data for this site.")
        else:
            st.info("SHAP attribution data not loaded.")

        st.markdown("---")

        # Synthetic Control Analysis
        st.markdown("### Causal Verification (Synthetic Control Analysis)")
        
        with st.spinner("Running Synthetic Control Analysis..."):
            control_ids, sca_results, damage_results = perform_sca_analysis(site_id, df, ts_df)
                # 🔍 DEBUG INFO
            if "sca_debug_counter" in st.session_state:
                st.caption(
                    f"SCA recomputed **{st.session_state.sca_debug_counter}** times. "
                    f"Last run for site **{st.session_state.sca_last_site}** "
                    f"at **{st.session_state.sca_last_time}**."
                )
            if control_ids is not None and sca_results is not None:
                st.session_state.sca_control_ids = control_ids
                st.success(f" Comparing treated sites against control sites")
                

                #  CREATE AND DISPLAY SCA PLOTS WITH DAMAGE ATTRIBUTION
                sca_fig = create_sca_plots(sca_results, damage_results)
                if sca_fig:
                    st.plotly_chart(sca_fig, use_container_width=True, key=f"sca_plot_{site_id}")
                    st.caption("Blue dashed line shows the synthetic counterfactual. Divergence indicates mining-related changes.")
                else:
                    st.warning("Could not generate SCA visualization.")
            else:
                st.warning(" Insufficient data for synthetic control analysis.")
                st.session_state.sca_control_ids = None