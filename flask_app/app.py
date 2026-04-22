from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import pandas as pd
import numpy as np
import pickle
import io
from pathlib import Path

app = Flask(__name__)

BASE    = Path(__file__).parent.parent
MODELS  = BASE / 'models'
REPORTS = BASE / 'reports'
IMAGES  = BASE / 'images'

model        = pickle.load(open(MODELS / 'model.pkl',         'rb'))
feature_cols = pickle.load(open(MODELS / 'columns.pkl',       'rb'))
scaler       = pickle.load(open(MODELS / 'scaler.pkl',        'rb'))
store_map    = pickle.load(open(MODELS / 'store_mapping.pkl', 'rb'))
dept_map     = pickle.load(open(MODELS / 'dept_mapping.pkl',  'rb'))

df_hist = pd.read_csv(REPORTS / 'historique_walmart.csv', parse_dates=['Date'])

try:
    df_prev = pd.read_csv(REPORTS / 'previsions_store.csv', parse_dates=['Date'])
except Exception:
    df_prev = pd.DataFrame()

COLS_SCALE = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
              'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
              'MarkDown_total', 'Size']
COLS_SCALE = [c for c in COLS_SCALE if c in feature_cols]


# ── Pages ─────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(str(IMAGES), filename)


# ── Historique / top / KPIs (legacy) ──────────────────────────

@app.route('/api/historique')
def api_historique():
    store = request.args.get('store', type=int)
    dept  = request.args.get('dept',  type=int)
    df    = df_hist.copy()
    if store: df = df[df['Store'] == store]
    if dept:  df = df[df['Dept']  == dept]
    agg = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    agg['Date'] = agg['Date'].dt.strftime('%Y-%m-%d')
    return jsonify(agg.to_dict(orient='list'))


@app.route('/api/top_stores')
def api_top_stores():
    top = (df_hist.groupby('Store')['Weekly_Sales']
           .sum().sort_values(ascending=False).head(10).reset_index())
    return jsonify(top.to_dict(orient='list'))


@app.route('/api/kpis')
def api_kpis():
    return jsonify({
        'ventes_total': int(df_hist['Weekly_Sales'].sum()),
        'nb_stores':    int(df_hist['Store'].nunique()),
        'nb_depts':     int(df_hist['Dept'].nunique()),
        'ventes_moy':   int(df_hist['Weekly_Sales'].mean()),
        'wmae':         1264,
        'r2':           98.55,
    })


# ── Analytics ─────────────────────────────────────────────────

@app.route('/api/analytics/overview')
def analytics_overview():
    df = df_hist.copy()
    store_filter = request.args.get('store', 'all')
    date_from    = request.args.get('date_from')
    date_to      = request.args.get('date_to')
    # Filtre date seulement (pour le camembert — sans filtre store)
    df_year = df.copy()
    if date_from: df_year = df_year[df_year['Date'] >= pd.to_datetime(date_from)]
    if date_to:
        import calendar
        yr, mo = int(date_to[:4]), int(date_to[5:7])
        _end = pd.Timestamp(year=yr, month=mo, day=calendar.monthrange(yr, mo)[1])
        df_year = df_year[df_year['Date'] <= _end]
    # Filtre complet (store + dates)
    if store_filter != 'all': df = df[df['Store'] == int(store_filter)]
    if date_from: df = df[df['Date'] >= pd.to_datetime(date_from)]
    if date_to:   df = df[df['Date'] <= _end]
    stores_info = df_year.drop_duplicates('Store')[['Store', 'Type']]
    type_counts = stores_info.groupby('Type').size().to_dict()
    type_ca     = (df_year.groupby('Type')['Weekly_Sales'].sum() / 1e9).round(2).to_dict()
    top15 = (df.groupby('Dept')['Weekly_Sales'].sum()
             .sort_values(ascending=False).head(15))
    # Pareto — tous les départements triés par CA décroissant
    all_depts = (df.groupby('Dept')['Weekly_Sales'].sum()
                 .sort_values(ascending=False))
    total_ca = all_depts.sum()
    pareto_pct   = (all_depts / total_ca * 100).round(2).tolist()
    pareto_cumul = []
    s = 0
    for v in pareto_pct:
        s += v
        pareto_cumul.append(round(s, 2))
    # Croissance réelle CA 2010 → 2012
    ca_by_year = df.groupby(df['Date'].dt.year)['Weekly_Sales'].sum()
    yr_first, yr_last = ca_by_year.index.min(), ca_by_year.index.max()
    ca_growth = round((ca_by_year[yr_last] / ca_by_year[yr_first] - 1) * 100, 1)
    # Comparaison annuelle détaillée
    yearly_ca = {str(yr): round(val/1e9, 2) for yr, val in ca_by_year.items()}
    yearly_growth = {}
    years_sorted = sorted(ca_by_year.index)
    for i in range(1, len(years_sorted)):
        y_prev, y_curr = years_sorted[i-1], years_sorted[i]
        g = round((ca_by_year[y_curr]/ca_by_year[y_prev] - 1)*100, 1)
        yearly_growth[str(y_curr)] = g
    return jsonify({
        'type_counts': type_counts,
        'type_ca':     type_ca,
        'ca_total':    round(df['Weekly_Sales'].sum() / 1e9, 2),
        'nb_obs':      len(df),
        'nb_stores':   int(df['Store'].nunique()),
        'nb_depts':    int(df['Dept'].nunique()),
        'ventes_moy':  round(df['Weekly_Sales'].mean()),
        'top_depts':   top15.index.tolist(),
        'top_depts_ca': (top15.values / 1e6).round(1).tolist(),
        'pareto_depts':  all_depts.index.tolist(),
        'pareto_pct':    pareto_pct,
        'pareto_cumul':  pareto_cumul,
        'ca_growth':     ca_growth,
        'yearly_ca':     yearly_ca,
        'yearly_growth': yearly_growth,
    })


_temporal_cache = {}

@app.route('/api/analytics/temporal')
def analytics_temporal():
    store_filter = request.args.get('store', 'all')
    date_to      = request.args.get('date_to')
    cache_key = f"{store_filter}|{date_to or ''}"
    if cache_key in _temporal_cache:
        return jsonify(_temporal_cache[cache_key])
    df = df_hist.copy()
    if store_filter != 'all': df = df[df['Store'] == int(store_filter)]
    if date_to:
        import calendar
        yr, mo = int(date_to[:4]), int(date_to[5:7])
        last_day = calendar.monthrange(yr, mo)[1]
        df = df[df['Date'] <= pd.Timestamp(year=yr, month=mo, day=last_day)]
    df['YM'] = df['Date'].dt.to_period('M')
    monthly = df.groupby('YM')['Weekly_Sales'].mean().reset_index()
    monthly['label'] = monthly['YM'].dt.strftime('%b %y')

    df['Month'] = df['Date'].dt.month
    seasonal = (df.groupby('Month')['Weekly_Sales'].mean()
                .reindex(range(1, 13)).round().astype(int))

    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.strftime('%W').astype(int)
    yearly = {}
    yearly_weekly = {}
    for yr in sorted(df['Year'].unique()):
        ym = (df[df['Year'] == yr].groupby('Month')['Weekly_Sales'].mean()
              .reindex(range(1, 13)).round().fillna(0).astype(int))
        yearly[str(yr)] = ym.tolist()
        yw = (df[df['Year'] == yr].groupby('Week')['Weekly_Sales'].sum()
              .reindex(range(0, 52)).round())
        yearly_weekly[str(yr)] = [None if pd.isna(v) else int(v) for v in yw]

    # Ventes hebdo totales + moyenne mobile + jours fériés
    weekly = df.groupby('Date').agg(
        total=('Weekly_Sales', 'sum'),
        is_holiday=('IsHoliday', 'max')
    ).reset_index().sort_values('Date')
    weekly['ma4'] = weekly['total'].rolling(4, min_periods=1).mean().round().astype(int)
    HOLIDAY_NAMES = {
        '2010-02-12': 'Super Bowl', '2011-02-11': 'Super Bowl', '2012-02-10': 'Super Bowl',
        '2010-09-10': 'Labor Day',  '2011-09-09': 'Labor Day',  '2012-09-07': 'Labor Day',
        '2010-11-26': 'Thanksgiving','2011-11-25': 'Thanksgiving','2012-11-23': 'Thanksgiving',
        '2010-12-31': 'Christmas',   '2011-12-30': 'Christmas',   '2012-12-28': 'Christmas',
    }
    hol_rows = weekly[weekly['is_holiday'] == True]['Date']
    holiday_dates = hol_rows.dt.strftime('%Y-%m-%d').tolist()
    holiday_names = [HOLIDAY_NAMES.get(d, 'Jour férié') for d in holiday_dates]

    MNAMES = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
              'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    result = {
        'monthly_labels':  monthly['label'].tolist(),
        'monthly_values':  monthly['Weekly_Sales'].round().astype(int).tolist(),
        'seasonal_labels': MNAMES,
        'seasonal_values': seasonal.tolist(),
        'yearly':          yearly,
        'weekly_dates':    weekly['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'weekly_totals':   weekly['total'].round().astype(int).tolist(),
        'weekly_ma4':      weekly['ma4'].tolist(),
        'holiday_dates':   holiday_dates,
        'holiday_names':   holiday_names,
        'yearly_weekly':   yearly_weekly,
    }
    _temporal_cache[cache_key] = result
    return jsonify(result)


@app.route('/api/analytics/stores')
def analytics_stores():
    df = df_hist.copy()
    store_param = request.args.get('store')
    date_to     = request.args.get('date_to')
    if store_param:
        df = df[df['Store'] == int(store_param)]
    if date_to:
        df = df[df['Date'] <= pd.to_datetime(date_to + '-28')]
    if df.empty:
        return jsonify({'stores':[], 'store_types':[], 'store_avgs':[], 'type_avg':{}})
    store_avg = (df.groupby('Store')
                 .agg(type=('Type', 'first'), avg=('Weekly_Sales', 'mean'))
                 .reset_index())
    type_avg = (df.groupby('Type')['Weekly_Sales'].mean()
                .round().astype(int).to_dict())
    return jsonify({
        'stores':      store_avg['Store'].tolist(),
        'store_types': store_avg['type'].tolist(),
        'store_avgs':  store_avg['avg'].round().astype(int).tolist(),
        'type_avg':    type_avg,
    })


@app.route('/api/analytics/departments')
def analytics_departments():
    df = df_hist.copy()
    date_to = request.args.get('date_to')
    if date_to:
        df = df[df['Date'] <= pd.to_datetime(date_to + '-28')]
    dept_ca = df.groupby('Dept')['Weekly_Sales'].sum()
    top15   = dept_ca.sort_values(ascending=False).head(15)
    flop10  = dept_ca.sort_values(ascending=True).head(10)

    df['Month'] = df['Date'].dt.month
    top3 = dept_ca.sort_values(ascending=False).head(3).index.tolist()
    MNAMES = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
              'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    seasonal = {}
    for d in top3:
        s = (df[df['Dept'] == d].groupby('Month')['Weekly_Sales'].mean()
             .reindex(range(1, 13), fill_value=0).round().astype(int))
        seasonal[str(d)] = s.tolist()

    return jsonify({
        'top_depts':      top15.index.tolist(),
        'top_ca':         (top15.values / 1e6).round(1).tolist(),
        'flop_depts':     flop10.index.tolist(),
        'flop_ca':        (flop10.values / 1e3).round(1).tolist(),
        'seasonal_depts': top3,
        'seasonal_labels': MNAMES,
        'seasonal_data':  seasonal,
    })


@app.route('/api/analytics/promotions')
def analytics_promotions():
    df = df_hist.copy()
    date_to = request.args.get('date_to')
    if date_to:
        df = df[df['Date'] <= pd.to_datetime(date_to + '-28')]
    md_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df['has_promo'] = df[md_cols].fillna(0).sum(axis=1) > 0

    w = df[df['has_promo']]['Weekly_Sales'].mean()
    n = df[~df['has_promo']]['Weekly_Sales'].mean()
    w = float(w) if not pd.isna(w) else 0.0
    n = float(n) if not pd.isna(n) else 0.0

    md_lifts = []
    for col in md_cols:
        h  = df[df[col].fillna(0) > 0]['Weekly_Sales'].mean()
        no = df[df[col].fillna(0) == 0]['Weekly_Sales'].mean()
        h  = float(h)  if not pd.isna(h)  else 0.0
        no = float(no) if not pd.isna(no) else 0.0
        md_lifts.append(round((h - no) / no * 100, 1) if no > 0 else 0)

    return jsonify({
        'with_promo':    round(w),
        'without_promo': round(n),
        'lift':          round((w - n) / n * 100, 1) if n > 0 else 0,
        'pct_promo':     round(df['has_promo'].mean() * 100, 1),
        'md_lifts':      md_lifts,
    })


@app.route('/api/analytics/holidays')
def analytics_holidays():
    df = df_hist.copy()
    date_to = request.args.get('date_to')
    if date_to:
        df = df[df['Date'] <= pd.to_datetime(date_to + '-28')]
    h_avg = df[df['IsHoliday']]['Weekly_Sales'].mean()
    n_avg = df[~df['IsHoliday']]['Weekly_Sales'].mean()

    type_lifts = {}
    for t in ['A', 'B', 'C']:
        sub = df[df['Type'] == t]
        h = sub[sub['IsHoliday']]['Weekly_Sales'].mean()
        n = sub[~sub['IsHoliday']]['Weekly_Sales'].mean()
        type_lifts[t] = round((h - n) / n * 100, 1) if n > 0 else 0

    HOLIDAY_DATES = {
        'Super Bowl':   ['2010-02-12', '2011-02-11', '2012-02-10'],
        'Labor Day':    ['2010-09-10', '2011-09-09', '2012-09-07'],
        'Thanksgiving': ['2010-11-26', '2011-11-25', '2012-11-23'],
        'Christmas':    ['2010-12-31', '2011-12-30', '2012-12-28'],
    }
    non_h = df[~df['IsHoliday']]['Weekly_Sales'].mean()
    fete_lifts = {}
    for fete, dates in HOLIDAY_DATES.items():
        sub = df[df['Date'].isin(pd.to_datetime(dates))]
        if not sub.empty:
            fete_lifts[fete] = round((sub['Weekly_Sales'].mean() - non_h) / non_h * 100, 1)
        else:
            fete_lifts[fete] = 0.0

    return jsonify({
        'holiday_avg': round(h_avg),
        'normal_avg':  round(n_avg),
        'lift':        round((h_avg - n_avg) / n_avg * 100, 1),
        'type_lifts':  type_lifts,
        'fete_names':  list(fete_lifts.keys()),
        'fete_lifts':  list(fete_lifts.values()),
    })


@app.route('/api/analytics/economic')
def analytics_economic():
    df = df_hist.copy()
    store_param = request.args.get('store')
    date_to     = request.args.get('date_to')
    if store_param:
        df = df[df['Store'] == int(store_param)]
    if date_to:
        df = df[df['Date'] <= pd.to_datetime(date_to + '-28')]
    configs = {
        'temp':  {'col': 'Temperature', 'bins': [0, 10, 25, 40, 55, 70, 85, 200],
                  'labels': ['<10°F', '10-25°F', '25-40°F', '40-55°F', '55-70°F', '70-85°F', '>85°F']},
        'fuel':  {'col': 'Fuel_Price',  'bins': [0, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 10],
                  'labels': ['<$2.5', '$2.5-2.7', '$2.7-2.9', '$2.9-3.1', '$3.1-3.3', '$3.3-3.5', '>$3.5']},
        'cpi':   {'col': 'CPI',         'bins': [0, 140, 155, 170, 185, 200, 215, 300],
                  'labels': ['<140', '140-155', '155-170', '170-185', '185-200', '200-215', '>215']},
        'unemp': {'col': 'Unemployment', 'bins': [0, 6, 7, 8, 9, 10, 11, 20],
                  'labels': ['<6%', '6-7%', '7-8%', '8-9%', '9-10%', '10-11%', '>11%']},
    }
    result = {}
    for key, cfg in configs.items():
        df['_b'] = pd.cut(df[cfg['col']], bins=cfg['bins'], labels=cfg['labels'])
        avgs = (df.groupby('_b', observed=False)['Weekly_Sales'].mean()
                .reindex(cfg['labels']).fillna(0).round().astype(int))
        result[key] = {'labels': cfg['labels'], 'values': avgs.tolist()}
    return jsonify(result)


@app.route('/api/analytics/correlations')
def analytics_correlations():
    df = df_hist.copy()
    df['IsHoliday'] = df['IsHoliday'].astype(int)
    features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                'Size', 'IsHoliday']
    corrs = []
    for f in features:
        if f in df.columns:
            c = df[f].fillna(0).corr(df['Weekly_Sales'])
            corrs.append({'feature': f, 'corr': round(float(c), 3)})
    corrs.sort(key=lambda x: x['corr'], reverse=True)
    return jsonify(corrs)


# ── Prédiction ────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    try:
        store_raw = data['store']
        dept_raw  = data['dept']
        markdown  = float(data.get('markdown', 0))
        date_str  = data.get('date', '')

        all_store = (str(store_raw) == 'all')
        all_dept  = (str(dept_raw)  == 'all')
        store_id  = None if all_store else int(store_raw)
        dept_id   = None if all_dept  else int(dept_raw)

        subset = df_hist.copy()
        if not all_store:
            subset = subset[subset['Store'] == store_id]
        if not all_dept:
            subset = subset[subset['Dept'] == dept_id]
        if subset.empty:
            return jsonify({'error': 'Combinaison Store/Dept introuvable'}), 404

        row = subset.tail(1).copy()

        if date_str:
            pred_date = pd.to_datetime(date_str)
        else:
            pred_date = row['Date'].iloc[0]

        row['Year']      = pred_date.year
        row['Month']     = pred_date.month
        row['Week']      = pred_date.isocalendar()[1]
        row['Quarter']   = pred_date.quarter
        row['DayOfYear'] = pred_date.day_of_year
        row['Month_sin'] = np.sin(2 * np.pi * pred_date.month / 12)
        row['Month_cos'] = np.cos(2 * np.pi * pred_date.month / 12)
        row['Week_sin']  = np.sin(2 * np.pi * pred_date.isocalendar()[1] / 52)
        row['Week_cos']  = np.cos(2 * np.pi * pred_date.isocalendar()[1] / 52)
        row['IsHoliday'] = 0
        row['Type_ord']  = row['Type'].map({'A': 3, 'B': 2, 'C': 1}) if 'Type' in row.columns else 2

        for col in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
            if col not in row.columns:
                row[col] = 0
        row['MarkDown_total'] = row[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].sum(axis=1)
        row['HasMarkDown']    = (row['MarkDown_total'] > 0).astype(int)

        row['Store'] = store_map.get(store_id, store_map[list(store_map.keys())[0]]) if not all_store else list(store_map.values())[0]
        row['Dept']  = dept_map.get(dept_id,   dept_map[list(dept_map.keys())[0]])   if not all_dept  else list(dept_map.values())[0]

        base = subset['Weekly_Sales']
        for lag in ['Lag_1', 'Lag_4', 'Lag_12', 'Lag_52', 'MA_4', 'MA_12']:
            row[lag] = float(base.mean()) if not base.empty else 0

        temp_df = row.copy()
        available = [c for c in COLS_SCALE if c in temp_df.columns]
        if available:
            temp_df[available] = scaler.transform(
                temp_df[available].values.reshape(1, -1)
                if len(available) == len(COLS_SCALE)
                else temp_df[available]
            )

        X = temp_df[feature_cols].fillna(0)
        pred_sans = float(model.predict(X)[0])

        pred_avec = pred_sans
        if markdown > 0 and 'MarkDown1' in COLS_SCALE:
            idx = COLS_SCALE.index('MarkDown1')
            scaled_md = (markdown - scaler.mean_[idx]) / scaler.scale_[idx]
            X_avec = X.copy()
            X_avec['MarkDown1'] = scaled_md
            pred_avec = float(model.predict(X_avec)[0])

        gain = pred_avec - pred_sans
        roi  = (gain * 0.30 - markdown * 0.60) / (markdown * 0.60) * 100 if markdown > 0 else 0

        return jsonify({
            'pred_sans': round(pred_sans, 2),
            'pred_avec': round(pred_avec, 2),
            'gain':      round(gain, 2),
            'roi':       round(roi, 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/horizon', methods=['POST'])
def api_predict_horizon():
    data = request.get_json()
    try:
        store_id  = int(data['store'])
        dept_id   = int(data['dept'])
        max_weeks = int(data.get('weeks', 4))
        base_date = pd.to_datetime(data.get('baseDate', '2012-10-26'))
        md1  = float(data.get('md1', 0));  md2 = float(data.get('md2', 0))
        md3  = float(data.get('md3', 0));  md4 = float(data.get('md4', 0))
        md5  = float(data.get('md5', 0))
        temp = float(data.get('temperature', 55))
        fuel = float(data.get('fuelPrice', 3.45))
        cpi  = float(data.get('cpi', 210))
        unem = float(data.get('unemployment', 8.0))

        subset = df_hist[(df_hist['Store'] == store_id) & (df_hist['Dept'] == dept_id)]
        if subset.empty:
            return jsonify({'error': 'Combinaison Store/Dept introuvable'}), 404

        row_base = subset.tail(1).copy()
        base_sales = subset['Weekly_Sales']

        results = []
        for w in range(1, max_weeks + 1):
            pred_date = base_date + pd.Timedelta(weeks=w)
            row = row_base.copy()
            row['Year']        = pred_date.year
            row['Month']       = pred_date.month
            row['Week']        = pred_date.isocalendar()[1]
            row['Quarter']     = pred_date.quarter
            row['DayOfYear']   = pred_date.day_of_year
            row['Month_sin']   = np.sin(2 * np.pi * pred_date.month / 12)
            row['Month_cos']   = np.cos(2 * np.pi * pred_date.month / 12)
            row['Week_sin']    = np.sin(2 * np.pi * pred_date.isocalendar()[1] / 52)
            row['Week_cos']    = np.cos(2 * np.pi * pred_date.isocalendar()[1] / 52)
            row['IsHoliday']   = 0
            row['Type_ord']    = row['Type'].map({'A':3,'B':2,'C':1}) if 'Type' in row.columns else 2
            row['Temperature'] = temp;  row['Fuel_Price'] = fuel
            row['CPI']         = cpi;   row['Unemployment'] = unem
            row['MarkDown1']   = md1;   row['MarkDown2'] = md2
            row['MarkDown3']   = md3;   row['MarkDown4'] = md4;  row['MarkDown5'] = md5
            row['MarkDown_total'] = md1+md2+md3+md4+md5
            row['HasMarkDown']    = int(row['MarkDown_total'] > 0)
            row['Store'] = store_map.get(store_id, list(store_map.values())[0])
            row['Dept']  = dept_map.get(dept_id,   list(dept_map.values())[0])
            for lag in ['Lag_1','Lag_4','Lag_12','Lag_52','MA_4','MA_12']:
                row[lag] = float(base_sales.mean()) if not base_sales.empty else 0

            temp_df = row.copy()
            available = [c for c in COLS_SCALE if c in temp_df.columns]
            if available:
                temp_df[available] = scaler.transform(temp_df[available])
            X = temp_df[feature_cols].fillna(0)
            pred = round(float(model.predict(X)[0]), 2)
            results.append({
                'week':  w,
                'date':  str(pred_date.date()),
                'pred':  pred,
                'lower': round(pred * 0.95, 2),
                'upper': round(pred * 1.05, 2),
            })
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulate_promo', methods=['POST'])
def api_simulate_promo():
    data = request.get_json()
    try:
        store_id   = int(data['store'])
        dept_id    = int(data['dept'])
        md1        = float(data.get('md1', 0))
        md2        = float(data.get('md2', 0))
        md3        = float(data.get('md3', 0))
        md4        = float(data.get('md4', 0))
        md5        = float(data.get('md5', 0))
        md_total   = md1 + md2 + md3 + md4 + md5
        cout_promo = float(data.get('cout_promo', 3000))
        marge      = float(data.get('marge', 0.30))
        date_str   = data.get('date', '2012-09-07')

        subset = df_hist[(df_hist['Store'] == store_id) & (df_hist['Dept'] == dept_id)]
        if subset.empty:
            return jsonify({'error': 'Combinaison Store/Dept introuvable'}), 404

        row_base   = subset.tail(1).copy()
        base_sales = subset['Weekly_Sales']
        pred_date  = pd.to_datetime(date_str)

        def build_row(bmd1=0, bmd2=0, bmd3=0, bmd4=0, bmd5=0):
            row = row_base.copy()
            row['Year']        = pred_date.year
            row['Month']       = pred_date.month
            row['Week']        = pred_date.isocalendar()[1]
            row['Quarter']     = pred_date.quarter
            row['DayOfYear']   = pred_date.day_of_year
            row['Month_sin']   = np.sin(2 * np.pi * pred_date.month / 12)
            row['Month_cos']   = np.cos(2 * np.pi * pred_date.month / 12)
            row['Week_sin']    = np.sin(2 * np.pi * pred_date.isocalendar()[1] / 52)
            row['Week_cos']    = np.cos(2 * np.pi * pred_date.isocalendar()[1] / 52)
            row['IsHoliday']   = 0
            row['Type_ord']    = row['Type'].map({'A':3,'B':2,'C':1}) if 'Type' in row.columns else 2
            row['MarkDown1']   = bmd1; row['MarkDown2'] = bmd2
            row['MarkDown3']   = bmd3; row['MarkDown4'] = bmd4; row['MarkDown5'] = bmd5
            row['MarkDown_total'] = bmd1+bmd2+bmd3+bmd4+bmd5
            row['HasMarkDown']    = int(bmd1+bmd2+bmd3+bmd4+bmd5 > 0)
            row['Store'] = store_map.get(store_id, list(store_map.values())[0])
            row['Dept']  = dept_map.get(dept_id,   list(dept_map.values())[0])
            for lag in ['Lag_1','Lag_4','Lag_12','Lag_52','MA_4','MA_12']:
                row[lag] = float(base_sales.mean()) if not base_sales.empty else 0
            temp_df   = row.copy()
            available = [c for c in COLS_SCALE if c in temp_df.columns]
            if available:
                temp_df[available] = scaler.transform(temp_df[available])
            X = temp_df[feature_cols].fillna(0)
            return float(model.predict(X)[0])

        pred_sans = build_row(0, 0, 0, 0, 0)

        # Lift global toutes promotions (MarkDown1-5, tout dataset)
        _md_all = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
        _active = df_hist[_md_all].fillna(0).sum(axis=1) > 0
        g_avec  = df_hist[_active]['Weekly_Sales'].mean()
        g_sans  = df_hist[~_active]['Weekly_Sales'].mean()
        global_lift = (g_avec / g_sans - 1) if g_sans > 0 else 0.10
        global_ref  = df_hist[_active][_md_all].fillna(0).sum(axis=1).mean()
        if pd.isna(global_ref) or global_ref <= 0:
            global_ref = 5000.0

        # Lift local store/dept (blendé avec global si assez de données)
        sub_all   = df_hist[(df_hist['Store'] == store_id) & (df_hist['Dept'] == dept_id)].copy()
        sub_act   = sub_all[_md_all].fillna(0).sum(axis=1) > 0
        loc_avec  = sub_all[sub_act]['Weekly_Sales']
        loc_sans  = sub_all[~sub_act]['Weekly_Sales']
        if len(loc_avec) >= 5 and len(loc_sans) >= 5:
            local_lift = (loc_avec.mean() / loc_sans.mean()) - 1
            emp_lift   = 0.5 * local_lift + 0.5 * global_lift
        else:
            emp_lift = global_lift

        # Référence = total moyen des markdowns actifs (local ou global)
        loc_ref_s = sub_all[sub_act][_md_all].fillna(0).sum(axis=1)
        ref_md    = loc_ref_s.mean() if len(loc_ref_s) >= 5 else global_ref

        cout_pct = float(data.get('cout_pct', 0.20))

        def pred_avec_empirique(total_md):
            scale = min(np.sqrt(total_md / max(ref_md, 1.0)), 2.0)
            return pred_sans * (1 + emp_lift * scale)

        pred_avec  = pred_avec_empirique(md_total)
        gain       = pred_avec - pred_sans
        gain_pct   = (gain / pred_sans * 100) if pred_sans > 0 else 0
        gain_marge = gain * marge
        roi        = (gain_marge - cout_promo) / cout_promo * 100 if cout_promo > 0 else 0
        decision   = 'RENTABLE' if roi > 0 else ('NEUTRE' if roi == 0 else 'NON RENTABLE')

        # Scénarios : le montant = total markdown investi (indépendant de la décomposition)
        montants  = [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000]
        scenarios = []
        for m in montants:
            p = pred_avec_empirique(m)   # total = m
            g = p - pred_sans
            c = m * cout_pct
            r = (g * marge - c) / c * 100 if c > 0 else 0
            scenarios.append({'markdown': m, 'ventes': round(p,2),
                               'gain': round(g,2), 'cout': round(c,2), 'roi': round(r,1)})

        return jsonify({
            'pred_sans': round(pred_sans, 2),
            'pred_avec': round(pred_avec, 2),
            'gain':      round(gain, 2),
            'gain_pct':  round(gain_pct, 1),
            'roi':       round(roi, 1),
            'decision':  decision,
            'md_total':  round(md_total, 2),
            'scenarios': scenarios,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def api_predict_batch():
    try:
        f = request.files.get('file')
        if not f:
            return jsonify({'error': 'Aucun fichier reçu'}), 400

        fname = f.filename.lower()
        if fname.endswith('.xlsx') or fname.endswith('.xls'):
            df_in = pd.read_excel(f, parse_dates=['Date'])
        else:
            df_in = pd.read_csv(f, parse_dates=['Date'])

        required = ['Store', 'Dept', 'Date']
        missing = [c for c in required if c not in df_in.columns]
        if missing:
            return jsonify({'error': f'Colonnes manquantes : {missing}'}), 400

        results = []
        for _, r in df_in.iterrows():
            try:
                sid = int(r['Store'])
                did = int(r['Dept'])
                pred_date = pd.to_datetime(r['Date'])

                base = df_hist[(df_hist['Store'] == sid) & (df_hist['Dept'] == did)]
                row = base.tail(1).copy()
                if row.empty:
                    results.append({'Store': sid, 'Dept': did, 'Date': str(pred_date.date()), 'Prevision': None, 'Erreur': 'introuvable'})
                    continue

                row['Year']      = pred_date.year
                row['Month']     = pred_date.month
                row['Week']      = pred_date.isocalendar()[1]
                row['Quarter']   = pred_date.quarter
                row['DayOfYear'] = pred_date.day_of_year
                row['Month_sin'] = np.sin(2 * np.pi * pred_date.month / 12)
                row['Month_cos'] = np.cos(2 * np.pi * pred_date.month / 12)
                row['Week_sin']  = np.sin(2 * np.pi * pred_date.isocalendar()[1] / 52)
                row['Week_cos']  = np.cos(2 * np.pi * pred_date.isocalendar()[1] / 52)
                row['IsHoliday'] = int(r['IsHoliday']) if 'IsHoliday' in r else 0
                row['Type_ord']  = row['Type'].map({'A': 3, 'B': 2, 'C': 1}) if 'Type' in row.columns else 2

                for col in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
                    row[col] = float(r[col]) if col in r and pd.notna(r[col]) else 0
                row['MarkDown_total'] = row[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].sum(axis=1)
                row['HasMarkDown']    = (row['MarkDown_total'] > 0).astype(int)

                for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']:
                    if col in r and pd.notna(r[col]):
                        row[col] = float(r[col])
                if 'Type' in r and pd.notna(r['Type']):
                    row['Type']     = str(r['Type'])
                    row['Type_ord'] = {'A':3,'B':2,'C':1}.get(str(r['Type']), 2)

                row['Store'] = store_map.get(sid, list(store_map.values())[0])
                row['Dept']  = dept_map.get(did,  list(dept_map.values())[0])

                sales_base = base['Weekly_Sales']
                for lag in ['Lag_1', 'Lag_4', 'Lag_12', 'Lag_52', 'MA_4', 'MA_12']:
                    row[lag] = float(sales_base.mean()) if not sales_base.empty else 0

                temp_df = row.copy()
                available = [c for c in COLS_SCALE if c in temp_df.columns]
                if available:
                    temp_df[available] = scaler.transform(temp_df[available])

                X = temp_df[feature_cols].fillna(0)
                pred = round(float(model.predict(X)[0]), 2)
                results.append({'Store': sid, 'Dept': did, 'Date': str(pred_date.date()), 'Prevision': pred, 'Erreur': ''})
            except Exception as ex:
                results.append({'Store': r.get('Store','?'), 'Dept': r.get('Dept','?'), 'Date': str(r.get('Date','')), 'Prevision': None, 'Erreur': str(ex)})

        return jsonify({'results': results, 'total': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Gestion des stocks (EOQ + Alertes) ─────────────────────────

@app.route('/api/stock')
def api_stock():
    store_f = request.args.get('store', 'all')
    dept_f  = request.args.get('dept',  'all')
    S       = float(request.args.get('S', 50))
    H_rate  = float(request.args.get('H', 0.20))
    L       = float(request.args.get('L', 1))
    Z       = float(request.args.get('Z', 1.65))

    try:
        dp = pd.read_csv(REPORTS / 'previsions_store.csv', parse_dates=['Date'])
    except Exception:
        return jsonify({'error': 'previsions_store.csv introuvable'}), 404

    if store_f != 'all': dp = dp[dp['Store'] == int(store_f)]
    if dept_f  != 'all': dp = dp[dp['Dept']  == int(dept_f)]

    grp = dp.groupby(['Store', 'Dept']).agg(
        moy=('Weekly_Sales_Predicted', 'mean'),
        std=('Weekly_Sales_Predicted', 'std'),
    ).reset_index().fillna(0)

    grp['D']   = grp['moy'] * 52
    grp['EOQ'] = np.where(grp['moy'] > 0, np.sqrt(2 * grp['D'] * S / H_rate), 0).round(2)
    grp['Stock_securite'] = (Z * grp['std'] * np.sqrt(L)).round(2)
    grp['Stock_cible']    = (grp['Stock_securite'] + grp['moy'] * L).round(2)
    grp['CV']             = np.where(grp['moy'] > 0, grp['std'] / grp['moy'], 0).round(4)
    grp['Score_risque']   = (grp['CV'] * grp['moy']).round(2)

    def urgence(cv):
        if cv > 0.40: return 'CRITIQUE'
        if cv > 0.25: return 'HAUTE'
        if cv > 0.15: return 'MODEREE'
        return 'OK'

    grp['Urgence'] = grp['CV'].apply(urgence)
    grp = grp.sort_values('Score_risque', ascending=False)

    rows = grp[['Store','Dept','moy','std','EOQ','Stock_securite','Stock_cible','CV','Score_risque','Urgence']].rename(
        columns={'moy':'Ventes_moy','std':'Ventes_std'}
    ).to_dict('records')

    counts = grp['Urgence'].value_counts().to_dict()
    return jsonify({
        'rows':   rows,
        'counts': {k: int(counts.get(k,0)) for k in ['CRITIQUE','HAUTE','MODEREE','OK']},
    })


# ── Historique pour graphique simulateur ───────────────────────

@app.route('/api/history')
def api_history():
    store_raw = request.args.get('store', 'all')
    dept_raw  = request.args.get('dept',  'all')
    try:
        dp = pd.read_csv(REPORTS / 'previsions_store.csv', parse_dates=['Date'])
    except Exception:
        return jsonify({'error': 'previsions_store.csv introuvable'}), 404

    subset = dp.copy()
    if store_raw != 'all': subset = subset[subset['Store'] == int(store_raw)]
    if dept_raw  != 'all': subset = subset[subset['Dept']  == int(dept_raw)]

    agg = subset.groupby('Date').agg(
        Weekly_Sales=('Weekly_Sales', 'sum'),
        Weekly_Sales_Predicted=('Weekly_Sales_Predicted', 'sum')
    ).reset_index().sort_values('Date')

    agg['Lower_bound'] = (agg['Weekly_Sales_Predicted'] * 0.95).round(2)
    agg['Upper_bound'] = (agg['Weekly_Sales_Predicted'] * 1.05).round(2)
    agg['Difference']  = (agg['Weekly_Sales_Predicted'] - agg['Weekly_Sales']).round(2)
    agg['Date'] = agg['Date'].dt.strftime('%Y-%m-%d')

    return jsonify({
        'dates':      agg['Date'].tolist(),
        'real':       agg['Weekly_Sales'].round(2).tolist(),
        'predicted':  agg['Weekly_Sales_Predicted'].round(2).tolist(),
        'lower':      agg['Lower_bound'].tolist(),
        'upper':      agg['Upper_bound'].tolist(),
        'difference': agg['Difference'].tolist(),
    })


# ── Dashboard monitoring (EPIC 3.5) ────────────────────────────

@app.route('/api/dashboard')
def api_dashboard():
    try:
        dp = pd.read_csv(REPORTS / 'previsions_store.csv', parse_dates=['Date'])
    except Exception:
        return jsonify({'error': 'previsions_store.csv introuvable'}), 404

    store_f = request.args.get('store', 'all')
    dept_f  = request.args.get('dept',  'all')
    weeks_f = request.args.get('weeks', type=int, default=12)

    df_all = dp.copy()
    df = dp.copy()
    if store_f != 'all': df = df[df['Store'] == int(store_f)]
    if dept_f  != 'all': df = df[df['Dept']  == int(dept_f)]

    # Filtre temporel : garder les N dernières semaines
    all_dates = sorted(dp['Date'].unique())
    if weeks_f < len(all_dates):
        cutoff = all_dates[-weeks_f]
        df = df[df['Date'] >= cutoff]

    # ── KPI 1 : Ventes totales prévues
    total_pred = round(df['Weekly_Sales_Predicted'].sum() / 1e6, 2)

    # ── KPI 2 : Précision R² et WMAE
    y_true = df['Weekly_Sales']; y_pred = df['Weekly_Sales_Predicted']
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    r2 = round((1 - ss_res/ss_tot)*100, 1) if ss_tot > 0 else 0
    wmae = round((y_true - y_pred).abs().mean(), 0)

    # ── KPI 3 : Alertes critiques (CV > 0.25 = rupture potentielle)
    grp = df.groupby(['Store','Dept']).agg(
        mean_pred=('Weekly_Sales_Predicted','mean'),
        std_pred =('Weekly_Sales_Predicted','std'),
    ).reset_index()
    grp['cv'] = (grp['std_pred'] / grp['mean_pred'].replace(0, np.nan)).fillna(0)
    grp['urgence'] = grp['cv'].apply(lambda x: 'CRITIQUE' if x>0.4 else ('HAUTE' if x>0.25 else 'MODÉRÉE'))
    nb_critiques = int((grp['urgence']=='CRITIQUE').sum())

    # ── KPI 4 : EOQ moyen
    Z = 1.65  # 95% service level
    grp['sigma']  = grp['std_pred'] * np.sqrt(4)
    grp['stock_s'] = (Z * grp['sigma']).round(0)
    grp['eoq']    = (2 * grp['mean_pred'] * 52 * 5 / 0.20).apply(lambda x: round(np.sqrt(max(x,0)), 0))
    eoq_moyen = int(grp['eoq'].mean())

    # ── KPI 5 : Meilleur magasin
    best_store_row = df.groupby('Store')['Weekly_Sales_Predicted'].sum().idxmax()
    best_store_val = round(df.groupby('Store')['Weekly_Sales_Predicted'].sum().max() / 1e6, 1)

    # ── KPI 6 : Magasin à risque (le + d'alertes CRITIQUE)
    grp_store_crit = grp[grp['urgence']=='CRITIQUE'].groupby('Store').size()
    risk_store = int(grp_store_crit.idxmax()) if len(grp_store_crit) > 0 else '-'
    risk_count = int(grp_store_crit.max()) if len(grp_store_crit) > 0 else 0

    # ── KPI 7 : Période couverte
    periode = f"{df['Date'].min().strftime('%d/%m/%Y')} → {df['Date'].max().strftime('%d/%m/%Y')}"

    # ── Chart 1 : Réel vs Prévu (agrégé par date)
    weekly_agg = df.groupby('Date').agg(
        reel =('Weekly_Sales','sum'),
        prevu=('Weekly_Sales_Predicted','sum'),
    ).reset_index().sort_values('Date')
    chart_dates  = weekly_agg['Date'].dt.strftime('%Y-%m-%d').tolist()
    chart_reel   = weekly_agg['reel'].round().astype(int).tolist()
    chart_prevu  = weekly_agg['prevu'].round().astype(int).tolist()

    # ── Chart 2 : Classement des magasins par ventes prévues
    store_rank = df.groupby('Store')['Weekly_Sales_Predicted'].sum().reset_index()
    store_rank.columns = ['Store','Total']
    store_rank = store_rank.sort_values('Total', ascending=True).tail(20)
    store_avg_total = store_rank['Total'].mean()
    rank_stores  = [f"Store {s}" for s in store_rank['Store'].tolist()]
    rank_totaux  = store_rank['Total'].round().astype(int).tolist()
    rank_colors  = ['#27AE60' if v >= store_avg_total else '#E74C3C' for v in store_rank['Total']]

    # ── Chart 3 : Store × Dept (Top 15 par Dept)
    dept_agg = df.groupby('Dept').agg(reel=('Weekly_Sales','sum'), prevu=('Weekly_Sales_Predicted','sum')).reset_index()
    dept_top = dept_agg.nlargest(15, 'prevu')
    dept_labels = [f"Dept {d}" for d in dept_top['Dept'].tolist()]
    dept_reel   = dept_top['reel'].round().astype(int).tolist()
    dept_prevu  = dept_top['prevu'].round().astype(int).tolist()

    # ── Table alertes (top 30)
    alert_table = grp[grp['urgence'].isin(['CRITIQUE','HAUTE'])].nlargest(30,'cv')[
        ['Store','Dept','mean_pred','cv','urgence','eoq','stock_s']
    ].copy()
    alert_table['mean_pred'] = alert_table['mean_pred'].round(0).astype(int)
    alert_table['cv']        = (alert_table['cv']*100).round(1)
    alert_table['eoq']       = alert_table['eoq'].astype(int)
    alert_table['stock_s']   = alert_table['stock_s'].astype(int)
    alertes = alert_table.to_dict(orient='records')

    return jsonify({
        'kpi': {
            'total_pred':   total_pred,
            'r2':           r2,
            'wmae':         int(wmae),
            'nb_critiques': nb_critiques,
            'eoq_moyen':    eoq_moyen,
            'best_store':   int(best_store_row),
            'best_store_val': best_store_val,
            'risk_store':   risk_store,
            'risk_count':   risk_count,
            'periode':      periode,
        },
        'chart_dates':  chart_dates,
        'chart_reel':   chart_reel,
        'chart_prevu':  chart_prevu,
        'rank_stores':  rank_stores,
        'rank_totaux':  rank_totaux,
        'rank_colors':  rank_colors,
        'dept_labels':  dept_labels,
        'dept_reel':    dept_reel,
        'dept_prevu':   dept_prevu,
        'alertes':      alertes,
    })


# ── Metadata ───────────────────────────────────────────────────

@app.route('/api/metadata')
def metadata():
    stores = sorted(df_hist['Store'].unique().tolist())
    depts  = sorted(df_hist['Dept'].unique().tolist())
    return jsonify({'stores': stores, 'depts': depts})


# ── Export ─────────────────────────────────────────────────────

@app.route('/export/csv')
def export_csv():
    output = io.StringIO()
    df_hist.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        download_name='rapport_walmart.csv',
        as_attachment=True,
        mimetype='text/csv',
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)