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
from scipy.stats import pearsonr
import folium
import hashlib

def get_dataframe_hash(df):
    """Create a hash of DataFrame content for cache invalidation"""
    if df is None or df.empty:
        return "empty"
    # Hash the first/last rows and shape to detect changes
    try:
        content = f"{df.shape}_{df.iloc[0].to_dict()}_{df.iloc[-1].to_dict()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except:
        return "error"


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

def to_camel_case(text):
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    if len(s) == 1: 
        return s[0].lower()
    return s[0].lower() + "".join(i.capitalize() for i in s[1:])


def load_data(selected_state="Tamil Nadu"):
    """
    Load data for the selected state.
    Efficiently loads only necessary columns for the initial view where possible,
    but currently loads full dataset to support client-side filtering and analysis.
    """
    try:
        # 1. DEFINE FILE MAPPING
        # Explicit mapping based on user provided filenames
        file_mapping = {
            "Bihar": "BiharDataPredictions.csv",
            "Tamil Nadu": "tamilNaduPredictions.csv",
            "Uttar Pradesh": "UttarPradeshDataPredictions.csv"
        }
        
        # Get filename or fallback
        file_name = file_mapping.get(selected_state, f"{selected_state}DataPredictions.csv")
        
        # 2. CONSTRUCT PATHS
        # Check standard locations
        potential_paths = [
            os.path.join("ManKaavalGitHub", "dashboardData", "stateData", file_name),
            os.path.join("dashboardData", "stateData", file_name)
        ]
        
        file_path = None
        for path in potential_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            # st.error(f"❌ Data for {selected_state} not found (Expected: {file_name})")
            return pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 3. LOAD DATA EFFICIENTLY
        # Use low_memory=False to avoid mixed type warnings on large files
        # Parse dates immediately to avoid slow post-processing
        # Note: If file is huge, consider loading only subset. But for <100MB, full load is okay.
        
        # Ensure 'week_date' is parsed as date
        try:
            df = pd.read_csv(file_path, parse_dates=['week_date'])
        except ValueError:
             # Fallback if column name differs or date format is weird
            df = pd.read_csv(file_path)
        
        # 4. STANDARDIZE / NORMALIZE COLUMNS (handle common variants)
        # Build a case-insensitive mapping from known variants to standard names
        lower_map = {c.lower(): c for c in df.columns}

        rename_map = {}
        # site id variants
        for candidate in ['site_id', 'siteid', 'site', 'id']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'id'
                break

        # date variants
        for candidate in ['week_date', 'date', 'day']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'date'
                break

        # latitude variants
        for candidate in ['latitude', 'lat', 'lat_deg', 'latitude_deg']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'lat'
                break

        # longitude variants
        for candidate in ['longitude', 'lon', 'long', 'lng', 'longitude_deg']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'lon'
                break

        # probability / prediction variants
        for candidate in ['probability', 'prob', 'pred_prob', 'prediction', 'probability_score']:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = 'probability'
                break

        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure date format if not parsed above
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception:
                # leave as-is; downstream checks will catch invalid dates
                pass
            
        if 'date' not in df.columns:
            st.error(f"❌ '{file_name}' is missing a 'week_date' or 'date' column.")
            return pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 5. OPTIMIZATION: Reduce memory usage (optional but good practice)
        # Cast probability to float32
        if 'probability' in df.columns:
            df['probability'] = df['probability'].astype('float32')

        # 6. EXTRACT AVAILABLE DATES
        # Use numpy for faster unique/sort if needed, but pandas is fine here
        available_dates = sorted(df['date'].dt.strftime('%Y-%m-%d').unique().tolist())
        
        # 7. CREATE BASELINE DF (Latest date snapshot)
        if available_dates:
            latest_date = available_dates[-1]
            # Fast boolean indexing
            baseline_df = df[df['date'] == pd.Timestamp(latest_date)].copy()
        else:
            baseline_df = df.copy()

        # 8. PREPARE TIME SERIES DF
        # Ideally we only keep what we need: id, date, metrics...
        # But we keep all for now to be safe with SHAP etc.
        ts_df = df 

        # 9. DETERMINE SCA READY SITES (with lat/lon)
        if not df.empty:
            site_counts = df.groupby('id')['date'].count()
            valid_sites = site_counts[site_counts >= 15].index.tolist()

            # Attach coordinates from the latest snapshot (baseline_df)
            sca_ready_df = (
                baseline_df[baseline_df["id"].isin(valid_sites)][["id", "lat", "lon"]]
                .drop_duplicates(subset=["id"])
                .reset_index(drop=True)
            )
        else:
            sca_ready_df = pd.DataFrame(columns=["id", "lat", "lon"])

        # 10. EMPTY PLACEHOLDERS for deprecated flows
        shap_df = pd.DataFrame() 
        gt_df = pd.DataFrame()

        return baseline_df, ts_df, available_dates, shap_df, gt_df, sca_ready_df

    except Exception as e:
        st.error(f"❌ Error loading data for {selected_state}: {e}")
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
            return True, "✅ GEE initialized with service account"
        else:
            # Local development - use regular auth
            ee.Initialize(project="sandminingproject")
            return True, "✅ GEE initialized (local)"
            
    except Exception as e:
        error_msg = f"❌ GEE initialization failed: {str(e)}"
        st.error(error_msg)
        return False, error_msg


# ============================================================================
# MAP UTILITIES (Optimized for Speed)
# ============================================================================

def get_grid_feature_collection(_df: pd.DataFrame):
    """Convert points to GEE FeatureCollection (Optimized Vectorized Path)."""
    if _df is None or _df.empty:
        return None

    # Use to_dict('records') - significantly faster than iterrows/itertuples
    records = _df[['id', 'lat', 'lon', 'probability']].to_dict('records')
    offset = 0.0045
    
    features = []
    for rec in records:
        lon, lat, prob = rec['lon'], rec['lat'], rec['probability']
        geom = ee.Geometry.Polygon([[
            [lon - offset, lat - offset],
            [lon + offset, lat - offset],
            [lon + offset, lat + offset],
            [lon - offset, lat + offset],
            [lon - offset, lat - offset]
        ]], None, False) # 'False' for geodesic avoids extra calculations
        
        features.append(ee.Feature(geom, {
            'probability': float(prob),
            'site_id': str(rec['id'])
        }))
    
    return ee.FeatureCollection(features)

def get_sca_ready_layer(_sca_ready_df: pd.DataFrame):
    """Create GEE layer for SCA-ready sites (Optimized Vectorized Path)."""
    if _sca_ready_df is None or _sca_ready_df.empty:
        return None

    # Use to_dict('records') for speed
    records = _sca_ready_df[['id', 'lat', 'lon']].to_dict('records')
    offset = 0.0045
    
    features = []
    for rec in records:
        lon, lat = rec['lon'], rec['lat']
        geom = ee.Geometry.Polygon([[
            [lon - offset, lat - offset],
            [lon + offset, lat - offset],
            [lon + offset, lat + offset],
            [lon - offset, lat + offset],
            [lon - offset, lat - offset]
        ]], None, False)
        
        features.append(ee.Feature(geom, {
            'site_id': str(rec['id'])
        }))
    
    fc = ee.FeatureCollection(features)

    outline = ee.Image().byte().paint(
        featureCollection=fc,
        color=1,
        width=3,
    )
    return outline.visualize(**{
        'palette': ['00ff00'],
        'opacity': 1.0
    })






def create_gee_river_grid(
    gee_ready: bool,
    selected_state="Tamil Nadu",
    selected_date=None,
    show_coords: bool = False,
    aoi_coords=None,
    prediction_df_minimal: pd.DataFrame = None,
    sca_ready_df: pd.DataFrame = None,
):
    """Create a Google Earth Engine map with 1km grid - Following User Clean Version"""

    # State-specific center and zoom
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

    if aoi_coords:
        lats = [c[1] for c in aoi_coords]
        lons = [c[0] for c in aoi_coords]
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]
        zoom = 12

    m = geemap.Map(
        center=center,
        zoom=zoom,
        draw_control=True,
        measure_control=False,
        fullscreen_control=True,
        attribution_control=True,
    )

    if not gee_ready:
        m.add_basemap("SATELLITE")
        return m

    try:
        m.add_basemap("SATELLITE")

        if aoi_coords:
            roi = ee.Geometry.Polygon([list(aoi_coords)])
        else:
            # ROI filtered dynamically by state name
            roi = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
                ee.Filter.eq("ADM1_NAME", selected_state)
            )

        # ===== RIVER MASKING (Clean Logic) =====
        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        occurrence = gsw.select("occurrence")
        river_mask = occurrence.gt(10)
        connected_pixels = river_mask.selfMask().connectedPixelCount(1024)
        river_mask_clean = river_mask.updateMask(connected_pixels.gte(500))
        river_mask_final = river_mask_clean.clip(roi)
        buffered_river_mask = river_mask_final.focal_max(
            radius=100, units="meters", kernelType="circle"
        )

        # ===== 1KM GRID GENERATION WITH ACTUAL PREDICTIONS =====
        if prediction_df_minimal is not None and not prediction_df_minimal.empty:
            sites_fc = get_grid_feature_collection(prediction_df_minimal)
            if sites_fc is None:
                m.add_basemap("SATELLITE")
                return m
            prob_image = ee.Image().float().paint(sites_fc, "probability")
            prob_image = prob_image.clip(roi)
        else:
            # NO RANDOM FALLBACK - User requested removal
            m.add_basemap("SATELLITE")
            return m

        # Apply river mask
        grid_masked = prob_image.updateMask(buffered_river_mask)

        # Visualization
        fill_vis = grid_masked.visualize(**{
            "min": 0,
            "max": 1,
            "palette": ["00ff00", "ffff00", "ff0000"],
            "opacity": 0.6,
        })

        m.addLayer(fill_vis, {}, "Mining Activity Risk (1km Grid)", True)

        # SCA-ready outline layer
        if sca_ready_df is not None and not sca_ready_df.empty:
            sca_layer = get_sca_ready_layer(sca_ready_df)
            if sca_layer:
                m.addLayer(
                    sca_layer,
                    {},
                    f"SCA-Ready Sites ({len(sca_ready_df)} sites with 5+ timepoints)",
                    True,
                )

        if aoi_coords:
            m.addLayer(
                roi.style(fillColor="00000000", color="yellow", width=2),
                {},
                "Analyzed AOI Bound",
            )

        m.add_layer_control()
        return m

    except Exception as e:
        import traceback
        st.error(f"⚠️ Error creating GEE layers: {e}")
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
    - Find similar control sites based on:
        1. Probability < 0.4
        2. Same river basin
        3. Pearson correlation > 0.7 (pre-disturbance)
    - Build synthetic control for THIS specific site
    - Save analysis plots to syntheticcontroloutput/
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
        return None, None
    
    # Ensure date is datetime
    ts_df = ts_df.copy()
    ts_df['date'] = pd.to_datetime(ts_df['date'])
    
    # ===== STEP 1: Get time series for THIS SPECIFIC SITE (treated) =====
    ts_treated = ts_df[ts_df['id'] == site_id].copy()
    if ts_treated.empty:
        return None, None
    
    # Metrics to analyze
    metrics = ['NDVI', 'MNDWI', 'BSI']
    ts_treated_clean = ts_treated.dropna(subset=['date'] + metrics).sort_values('date')
    
    # Determine pre/post median date
    median_date = ts_treated_clean['date'].median()
    ts_treated_pre = ts_treated_clean[ts_treated_clean['date'] < median_date]
    
    if len(ts_treated_pre) < 3:
        print(f"[SCA] Insufficient pre-period data for site {site_id}")
        return None, None

    # ===== STEP 2: Find control sites with refined criteria =====
    # 1. Basic filter: low risk and not the same site
    treated_info = baseline_df[baseline_df['id'] == site_id].iloc[0]
    
    candidates = baseline_df[
        (baseline_df['probability'] < 0.4) & 
        (baseline_df['id'] != site_id)
    ].copy()
    
    # 2. Same river basin filter (check common column names)
    basin_col = next((col for col in candidates.columns if col.lower() in ['basin', 'river_basin', 'hybas_id']), None)
    if basin_col and basin_col in treated_info:
        target_basin = treated_info[basin_col]
        candidates = candidates[candidates[basin_col] == target_basin]
        print(f"[SCA] Filtered by basin {target_basin}: {len(candidates)} candidates")

    valid_control_ids = []
    
    # 3 & 4. Similarity and Pearson Correlation filter
    for _, cand in candidates.iterrows():
        cand_id = cand['id']
        ts_cand = ts_df[ts_df['id'] == cand_id].dropna(subset=['date'] + metrics)
        
        # Merge treated and candidate on dates in pre-period
        merged_pre = ts_treated_pre.merge(ts_cand, on='date', suffixes=('_t', '_c'))
        
        if len(merged_pre) >= 3:
            # Check correlations for metrics
            corrs = []
            for m in metrics:
                try:
                    c, _ = pearsonr(merged_pre[f'{m}_t'], merged_pre[f'{m}_c'])
                    if not np.isnan(c):
                        corrs.append(c)
                except:
                    continue
            
            # Criteria: Average Pearson correlation > 0.7
            if corrs and (sum(corrs)/len(corrs)) > 0.7:
                valid_control_ids.append(cand_id)

    if not valid_control_ids:
        print("[SCA] No control sites matched correlation criteria. Falling back to simple risk filter.")
        valid_control_ids = candidates['id'].unique().tolist()
    
    print(f"[SCA] Selected {len(valid_control_ids)} control sites")
    
    # ===== STEP 3: Aggregate control time series =====
    ts_control = ts_df[ts_df['id'].isin(valid_control_ids)].copy()
    control_agg = ts_control.groupby('date')[metrics].mean().reset_index()
    control_agg.columns = ['date'] + [f'control_{m}' for m in metrics]
    
    # Merge for analysis
    df_sca = ts_treated_clean.rename(columns={m: f'treated_{m}' for m in metrics}).merge(control_agg, on='date', how='inner')
    
    if len(df_sca) < 5:
        return None, None
        
    df_sca['period'] = df_sca['date'].apply(lambda x: 'PRE' if x < median_date else 'POST')
    df_pre = df_sca[df_sca['period'] == 'PRE'].copy()
    
    # ===== STEP 4: Fit synthetic control & Prepare Results =====
    results_dict = {}
    weights = {}
    
    for metric in metrics:
        try:
            X = df_pre[[f'control_{metric}']].values
            y = df_pre[[f'treated_{metric}']].values
            model = Ridge(alpha=0.01, fit_intercept=True)
            model.fit(X, y)
            
            df_sca[f'synthetic_{metric}'] = model.predict(df_sca[[f'control_{metric}']].values)
            weights[metric] = model.coef_.flatten()[0]
            
            results_dict[metric] = pd.DataFrame({
                'date': df_sca['date'],
                'actual': df_sca[f'treated_{metric}'],
                'synthetic': df_sca[f'synthetic_{metric}']
            })
        except Exception as e:
            print(f"[SCA] Error fitting {metric}: {e}")
            results_dict[metric] = pd.DataFrame({'date': df_sca['date'], 'actual': df_sca[f'treated_{metric}'], 'synthetic': df_sca[f'treated_{metric}']})

    # ===== STEP 5: Create and Save Graph =====
    try:
        output_dir = "syntheticcontroloutput"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        plt.style.use('dark_background')
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            res = results_dict[metric]
            
            # Plot Actual
            ax.plot(res['date'], res['actual'], label='Actual (Treated)', color='cyan', linewidth=2, marker='o')
            # Plot Synthetic
            ax.plot(res['date'], res['synthetic'], label='Synthetic (Baseline)', color='white', linestyle='--', linewidth=1.5)
            
            # Vertical line for treatment start
            ax.axvline(x=median_date, color='red', linestyle=':', label='Treatment Start')
            
            ax.set_title(f"{metric} Pattern: Actual vs Synthetic (Weight: {weights.get(metric, 0):.2f})")
            ax.legend()
            ax.grid(alpha=0.2)
            
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{site_id}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"[SCA] Graph saved to {plot_path}")
    except Exception as e:
        print(f"[SCA] Error saving graph: {e}")

    return valid_control_ids[:5], results_dict


def create_sca_plots(sca_results_dict):
    """Create three subplots for SCA (NDVI, MNDWI, BSI) - Actual vs Synthetic"""
    if not sca_results_dict:
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            'NDVI (Vegetation) - Actual vs Synthetic',
            'MNDWI (Water) - Actual vs Synthetic',
            'BSI (Soil) - Actual vs Synthetic'
        )
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

# Custom CSS for rich aesthetics and split-scrolling layout
st.markdown("""
<style>
    /* Dashboard Mode: Freeze the main page scroll */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden;
        height: 100vh;
    }

    /* Main content container should occupy full height */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-height: 100vh;
    }

    /* Sidebar Background */
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
    }
    div.stButton > button:hover {
        border-color: #ff4444;
        color: #ff4444;
    }

    /* Left Column: Fixed Map (No Scroll) */
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

    /* Premium Scrollbar for Right Column */
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



# Helper function for temporal filtering
def filter_df_by_date(data, date_str):
    if not date_str or data.empty:
        return data
    # Standard format check
    return data[data['date'].dt.strftime('%Y-%m-%d') == date_str].copy()

# ========== LAZY DATA LOADING (No @st.cache needed) ==========
current_state = st.session_state.get("selected_state", "Tamil Nadu")

if ("full_df" not in st.session_state or 
    st.session_state.get("loaded_state") != current_state):
    
    with st.spinner(f"🚀 Loading {current_state} Data into RAM..."):
        baseline_df, full_df, available_dates, shap_df, gt_df, sca_ready_df = load_data(current_state)
        
        # Persist in session state
        st.session_state.baseline_df = baseline_df
        st.session_state.full_df = full_df
        st.session_state.available_dates = available_dates
        st.session_state.sca_ready_df = sca_ready_df
        st.session_state.loaded_state = current_state
        
        # Reset date if needed
        if available_dates:
            st.session_state.selected_date = available_dates[-1]

# Local aliases for clean code
full_df = st.session_state.full_df
ts_df = full_df # Alias for historical data
available_dates = st.session_state.available_dates
sca_ready_df = st.session_state.sca_ready_df
baseline_df = st.session_state.baseline_df

# ========== INITIALIZE SESSION STATE (Default values) ==========
if 'selected_site' not in st.session_state:
    st.session_state.selected_site = None
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = "Tamil Nadu"

# Ensure selected_date is valid for the current state's available_dates
if available_dates:
    if 'selected_date' not in st.session_state or st.session_state.selected_date not in available_dates:
        st.session_state.selected_date = available_dates[-1] 
else:
    st.session_state.selected_date = None

# ========== CREATE BASELINE_DF AND MINIMAL_DF FOR MAP ==========
if not full_df.empty and available_dates:
    # Filter full_df for the selected date
    selected_dt = pd.Timestamp(st.session_state.selected_date)
    baseline_df = full_df[full_df['date'] == selected_dt].copy()
    
    # Prepare minimal dataframe for map (only needed columns)
    cols_needed = ["id", "lat", "lon", "probability"]
    if all(c in baseline_df.columns for c in cols_needed):
        prediction_minimal = baseline_df[cols_needed].copy()
    else:
        prediction_minimal = None
    
    # Sidebar stats uses the same baseline snapshot
    df_sidebar = baseline_df.copy()
else:
    baseline_df = pd.DataFrame()
    prediction_minimal = None
    df_sidebar = pd.DataFrame()
    
# Use the SCA-ready df from load_data
sca_ready_for_map = sca_ready_df if not sca_ready_df.empty else None
if sca_ready_for_map is not None:
    st.sidebar.info(f"📊 {len(sca_ready_for_map)} sites are SCA-ready")

# Initialize GEE
gee_ready, gee_msg = initialize_gee()
if not gee_ready:
    st.warning(gee_msg)

# ========== DATA DIAGNOSTICS ==========
def get_data_diagnostics():
    states_to_check = ["Bihar", "Tamil Nadu", "Uttar Pradesh"]
    file_mapping = {
        "Bihar": "BiharDataPredictions.csv",
        "Tamil Nadu": "tamilNaduPredictions.csv",
        "Uttar Pradesh": "UttarPradeshDataPredictions.csv"
    }
    stats = []
    for state in states_to_check:
        file_name = file_mapping[state]
        potential_paths = [
            os.path.join("ManKaavalGitHub", "dashboardData", "stateData", file_name),
            os.path.join("dashboardData", "stateData", file_name)
        ]
        path_used = "Not Found"
        rows = 0
        found = False
        for path in potential_paths:
            if os.path.exists(path):
                found = True
                path_used = path
                try:
                    # Quick size estimate or small read for row count
                    df_tmp = pd.read_csv(path, usecols=['site_id']).shape[0]
                    rows = df_tmp
                except:
                    rows = "Error"
                break
        stats.append({"State": state, "Status": "✅ Found" if found else "❌ Missing", "Records": rows})
    return pd.DataFrame(stats)

with st.sidebar:
    st.title("MannKaaval")
    st.markdown("# 🗺️ Select Region")
    
    with st.expander("🔍 Data Diagnostics"):
        diag_df = get_data_diagnostics()
        st.table(diag_df)
        if full_df.empty:
            st.error("⚠️ Current State Data: NOT LOADED")
        else:
            st.success(f"📊 State Total: {len(full_df):,} entries")
            st.info(f"📍 Active Sites: {len(baseline_df):,} (Current date snapshot)")
    
    # States list matching user request + Image 1 style
    # States list matching user request + Image 1 style
    states = ["Bihar", "Uttar Pradesh", "West Bengal", "Punjab", "Rajasthan", "Gujarat", "Tamil Nadu"]
    
    # Initialize state if not present
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = "Tamil Nadu"

    for state in states:
        is_active = (state == st.session_state.selected_state)
        label = f"✓ {state}" if is_active else state
        
        if is_active:
            st.markdown(f"""
            <div style="background-color: #ff4d4d; color: white; padding: 12px; border-radius: 8px; margin-bottom: 10px; font-weight: bold; text-align: center; border: 1px solid #ff4d4d;">
                {label}
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(state, key=f"sidebar_btn_{state}"):
                st.session_state.selected_state = state
                st.rerun()

    st.markdown("---")

    # 2. Highest Risk Sites (Image 1 Style)
    st.markdown("## ⚠️ Highest Risk Sites")
    if not df_sidebar.empty:
        # Use a consistent sample for the "Image 1" feel
        top_sites = df_sidebar.nlargest(10, 'probability')
        for i, (idx, row) in enumerate(top_sites.iterrows(), 1):
            prob = row['probability']
            risk_pct = prob * 100
            
            # Category logic
            if risk_pct > 80:
                status = "CRITICAL"; color = "#ff4444"; dot = "🔴"
            elif risk_pct > 60:
                status = "MEDIUM"; color = "#ffaa00"; dot = "🟡"
            else:
                status = "LOW"; color = "#44ff44"; dot = "🟢"

            # Use a dark card style
            st.markdown(f"""
            <div style="background: #1a1a1a; border: 1px solid #333; border-radius: 10px; padding: 15px; margin-bottom: 12px;">
                <div style="font-weight: bold; color: white; font-size: 1.1rem; margin-bottom: 10px;">{i}. Site {row['id']}</div>
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
    st.markdown("### 🛠️ System Controls")
    if st.button("🔄 Refresh Map Data", width=True, key="refresh_map_final"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared!")
        st.rerun()

# Create main layout
col_left, col_right = st.columns([3, 2])

# ============================================================================
# MAIN CONTENT AREA: MAP & SLIDER (FRAGMENT FOR SEAMLESS UPDATES)
# ============================================================================
with col_left:

    @st.fragment
    def map_and_slider_section():
        # 1. Filter Data
        df_filtered = filter_df_by_date(full_df, st.session_state.selected_date)

        # 2. Render Map
        with st.spinner(f"Updating predictions for {st.session_state.selected_date}..."):
            # Always pass a minimal prediction dataframe to the map
            if not df_filtered.empty:
                cols_needed = ["id", "lat", "lon", "probability"]
                missing = [c for c in cols_needed if c not in df_filtered.columns]
                if missing:
                    prediction_minimal = None
                else:
                    prediction_minimal = df_filtered[cols_needed].copy()
            else:
                prediction_minimal = None

            gee_map = create_gee_river_grid(
                gee_ready=gee_ready,
                selected_state=st.session_state.selected_state,
                selected_date=st.session_state.selected_date,
                show_coords=False,
                aoi_coords=None,
                prediction_df_minimal=prediction_minimal,
                sca_ready_df=sca_ready_for_map,
            )

            map_output = st_folium(
                gee_map,
                use_container_width=True,
                height=700,
                returned_objects=["last_clicked"],
                key="mankaaval_map_temporal",
            )


        # 3. Date Slider (Below the Map as requested)
        if available_dates:
            # Safety check: ensure current value is in options to avoid ValueError
            slider_val = st.session_state.selected_date
            if slider_val not in available_dates:
                slider_val = available_dates[-1]
                
            selected_date = st.select_slider(
                "📅 Select Analysis Date",
                options=available_dates,
                value=slider_val,
                key="date_slider_internal"
            )
            if selected_date != st.session_state.selected_date:
                st.session_state.selected_date = selected_date
                # No st.rerun() here - let fragment rerun seamlessly for the map

        # 4. Handle Clicks
        if map_output and map_output.get("last_clicked"):
            click_lat = map_output["last_clicked"]["lat"]
            click_lon = map_output["last_clicked"]["lng"]
            
            # Use the filtered df for nearest site
            nearest_site = find_nearest_site(click_lat, click_lon, df_filtered)
            
            if nearest_site is not None:
                if st.session_state.selected_site is None or st.session_state.selected_site['id'] != nearest_site['id']:
                    st.session_state.selected_site = nearest_site
                    st.rerun() # Full rerun to update right column statistics

    # Execute fragment
    map_and_slider_section()


# ============================================================================
# RIGHT COLUMN: SITE ANALYSIS
# ============================================================================
with col_right:

    with st.container(border=True, height=800):
    
        if st.session_state.selected_site is None:
            # Global Overview
            st.markdown("## 📊 Global Overview")
            if full_df.empty:
                st.info("No data loaded. Please check file paths.")
            else:
                st.info("Click a site on the map to get detailed analysis and trends.")
                st.metric("Total Monitored Sites", len(full_df['id'].unique()))
                if available_dates:
                    st.metric("Latest Update", available_dates[-1])

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
                    <p style="color: #aaa; margin: 0; font-size: 0.9rem;">RISK PROBABILITY ({st.session_state.selected_date})</p>
                    <h1 style="color: {risk_color}; margin: 10px 0; font-size: 3rem;">{risk_pct:.1f}%</h1>
                    <p style="color: {risk_color}; margin: 0; font-weight: bold;">
                        {"🔴 HIGH RISK" if risk_pct > 70 else "🟡 MODERATE" if risk_pct > 40 else "🟢 LOW RISK"}
                    </p>
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.markdown("---")

            # Site Information
            st.markdown(f"**📍 Location:** {site_lat:.4f}°N, {site_lon:.4f}°E")
            st.markdown(f"**🔎 Site ID:** {site_id}")

            st.markdown("---")

            # Filter time-series from selected date onwards
            start_date = pd.to_datetime(st.session_state.selected_date)
            ts_filtered = ts_df[ts_df['date'] >= start_date].copy()

            # Time Series Plot
            st.markdown("### 📈 Spectral Indices Over Time")
            ts_fig = create_timeseries_plot(site_id, ts_filtered)
            if ts_fig:
                st.plotly_chart(ts_fig, use_container_width=True, key=f"ts_plot_{site_id}")
            else:
                st.info("No time-series data available for this site from the selected start date.")

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
            st.markdown("### 🔬 Causal Verification (SCA)")
            
            with st.spinner("Running Synthetic Control Analysis..."):
                # Use unfiltered data for finding controls if needed, but here using df_sidebar or full_df
                control_ids, sca_results = perform_sca_analysis(site_id, df_sidebar, ts_filtered)
                
                if control_ids and sca_results:
                    st.session_state.sca_control_ids = control_ids
                    
                    sca_fig = create_sca_plots(sca_results)
                    if sca_fig:
                        st.plotly_chart(sca_fig, use_container_width=True, key=f"sca_plot_{site_id}")
                        st.caption("Blue dashed line shows the synthetic counterfactual. Divergence indicates mining-related changes.")
                    else:
                        st.warning("Could not generate SCA visualization.")
                else:
                    st.warning("⚠️ Insufficient data for synthetic control analysis.")
                    st.session_state.sca_control_ids = None