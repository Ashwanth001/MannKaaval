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