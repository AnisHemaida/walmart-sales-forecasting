"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SIPV — Tableau de bord d'aide à la décision (Walmart Sales)          ║
║  Streamlit Dashboard — Résumé de l'analyse exploratoire               ║
╚══════════════════════════════════════════════════════════════════════════╝
Lancement :  streamlit run dashboard_walmart.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.stats import spearmanr

# ─────────────────── CONFIG ───────────────────
st.set_page_config(
    page_title="SIPV Walmart — Aide à la décision",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────── CHARGEMENT DES DONNÉES ───────────────────
DATA_DIR = Path(__file__).resolve().parent / "Data"

@st.cache_data
def load_data():
    train    = pd.read_csv(DATA_DIR / "train.csv",    parse_dates=["Date"])
    features = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["Date"])
    stores_d = pd.read_csv(DATA_DIR / "stores.csv")

    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores_d, on="Store", how="left")

    md_cols = [c for c in df.columns if c.startswith("MarkDown")]
    df["HasPromo"]      = df[md_cols].notna().any(axis=1).astype(int)
    df["Total_MarkDown"] = df[md_cols].sum(axis=1, min_count=1)
    df["Month"]  = df["Date"].dt.month
    df["Year"]   = df["Date"].dt.year
    df["Week"]   = df["Date"].dt.isocalendar().week.astype(int)

    return df, stores_d

df_train, stores_info = load_data()

# ─────────────────── SIDEBAR ───────────────────
IMG_PATH = Path(__file__).resolve().parent / "images" / "Page de garde.png"
st.sidebar.image(str(IMG_PATH), width=180)
st.sidebar.markdown("## 🛒 Navigation")

pages = [
    "📊 Vue d'ensemble",
    "📈 Tendances temporelles",
    "🏪 Types de magasins",
    "📦 Analyse par département",
    "🏷️ Impact des promotions",
    "🎄 Effet des jours fériés",
    "🌡️ Variables économiques",
    "🔗 Corrélations & Features",
    "✅ Recommandations",
]
page = st.sidebar.radio("Section", pages, label_visibility="collapsed")

# Filtres globaux
st.sidebar.markdown("---")
st.sidebar.markdown("### Filtres")
types_sel = st.sidebar.multiselect("Type de magasin", ["A", "B", "C"], default=["A", "B", "C"])
years_sel = st.sidebar.multiselect("Année", sorted(df_train["Year"].unique()), default=sorted(df_train["Year"].unique()))

all_depts = sorted(df_train["Dept"].unique())
depts_sel = st.sidebar.multiselect("Département", all_depts, default=all_depts,
                                   help="Filtrer par numéro de département")

df = df_train[
    (df_train["Type"].isin(types_sel)) &
    (df_train["Year"].isin(years_sel)) &
    (df_train["Dept"].isin(depts_sel))
]

# Bouton de téléchargement
st.sidebar.markdown("---")
st.sidebar.markdown("### Exporter les données")
csv_export = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="📥 Télécharger (CSV)",
    data=csv_export,
    file_name="walmart_filtered.csv",
    mime="text/csv",
)

# ─────────────────── COULEURS ───────────────────
COLOR_TYPE  = {"A": "#3498db", "B": "#2ecc71", "C": "#e74c3c"}
COLOR_MAIN  = "#2E86C1"
COLOR_ACC   = "#E67E22"
COLOR_POS   = "#27AE60"
COLOR_NEG   = "#E74C3C"

# ═══════════════════════════════════════════════════════════════════
#  PAGE 1 — VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════════
if page == pages[0]:
    st.title("📊 Vue d'ensemble — Walmart Sales")
    st.markdown("Synthèse des indicateurs clés issus de l'analyse exploratoire")

   
    c1, c2, c3 = st.columns(3)
    c1.metric("Magasins", f'{df["Store"].nunique()}')
    c2.metric("Départements", f'{df["Dept"].nunique()}')
    c3.metric("Observations", f'{len(df):,}')
    

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Ventes moy./sem.", f'{df["Weekly_Sales"].mean():,.0f} $')
    c5.metric("Ventes médianes", f'{df["Weekly_Sales"].median():,.0f} $')
    c6.metric("CA total", f'{df["Weekly_Sales"].sum()/1e9:,.2f} Md$')
    c7.metric("% ventes négatives", f'{(df["Weekly_Sales"] < 0).mean()*100:.2f} %')

    c8,= st.columns(1)
    c8.metric("Période", f'{df["Date"].min().strftime("%Y-%m")} → {df["Date"].max().strftime("%Y-%m")}')
    

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Répartition par type de magasin")
        type_stats = (df.groupby("Type")["Weekly_Sales"]
                      .agg(["count", "mean", "median", "sum"])
                      .rename(columns={"count": "Nb obs", "mean": "Moy ($)", "median": "Méd ($)", "sum": "CA total ($)"})
                      .reset_index())
        fig_type = px.pie(type_stats, values="CA total ($)", names="Type",
                          color="Type", color_discrete_map=COLOR_TYPE,
                          title="Part du CA total par type")
        st.plotly_chart(fig_type, use_container_width=True)

    with col_b:
        st.subheader("Distribution des ventes hebdo.")
        fig_dist = px.histogram(df, x="Weekly_Sales", nbins=100,
                                title="Distribution de Weekly_Sales",
                                color_discrete_sequence=[COLOR_MAIN])
        fig_dist.update_layout(xaxis_title="Weekly_Sales ($)", yaxis_title="Fréquence")
        st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Top 10 magasins par CA total")
    top10 = (df.groupby("Store")["Weekly_Sales"].sum()
             .nlargest(10).reset_index()
             .rename(columns={"Weekly_Sales": "CA total ($)"}))
    top10 = top10.merge(stores_info, on="Store")
    fig_top = px.bar(top10, x="Store", y="CA total ($)", color="Type",
                     color_discrete_map=COLOR_TYPE, text_auto=".2s",
                     title="Top 10 magasins par chiffre d'affaires total")
    fig_top.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_top, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 2 — TENDANCES TEMPORELLES
# ═══════════════════════════════════════════════════════════════════
elif page == pages[1]:
    st.title("📈 Tendances temporelles")

    weekly = (df.groupby("Date")["Weekly_Sales"]
              .agg(["mean", "sum"]).reset_index()
              .rename(columns={"mean": "Moy", "sum": "Total"}))
    fig_ts = px.line(weekly, x="Date", y="Moy",
                     title="Ventes moyennes hebdomadaires",
                     color_discrete_sequence=[COLOR_MAIN])
    fig_ts.update_layout(yaxis_title="Ventes moyennes ($)", height=400)
    st.plotly_chart(fig_ts, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Saisonnalité mensuelle")
        monthly = (df.groupby("Month")["Weekly_Sales"].mean().reset_index()
                   .rename(columns={"Weekly_Sales": "Moy"}))
        month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                        "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
        monthly["Mois"] = monthly["Month"].map(lambda m: month_labels[m-1])
        fig_m = px.bar(monthly, x="Mois", y="Moy",
                       title="Ventes moyennes par mois",
                       color_discrete_sequence=[COLOR_MAIN])
        fig_m.update_layout(xaxis=dict(categoryorder="array",
                                       categoryarray=month_labels))
        st.plotly_chart(fig_m, use_container_width=True)

    with col2:
        st.subheader("Comparaison annuelle")
        yearly = (df.groupby(["Year", "Month"])["Weekly_Sales"]
                  .mean().reset_index())
        yearly["Mois"] = yearly["Month"].map(lambda m: month_labels[m-1])
        fig_y = px.line(yearly, x="Mois", y="Weekly_Sales", color="Year",
                        title="Profil mensuel par année",
                        color_discrete_sequence=["#3498db", "#e74c3c", "#2ecc71"])
        fig_y.update_layout(xaxis=dict(categoryorder="array",
                                       categoryarray=month_labels))
        st.plotly_chart(fig_y, use_container_width=True)

    st.subheader("Saisonnalité par type de magasin")
    monthly_type = (df.groupby(["Month", "Type"])["Weekly_Sales"]
                    .mean().reset_index())
    monthly_type["Mois"] = monthly_type["Month"].map(lambda m: month_labels[m-1])
    fig_mt = px.line(monthly_type, x="Mois", y="Weekly_Sales", color="Type",
                     color_discrete_map=COLOR_TYPE,
                     title="Profil saisonnier par type de magasin")
    fig_mt.update_layout(xaxis=dict(categoryorder="array",
                                    categoryarray=month_labels))
    st.plotly_chart(fig_mt, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 3 — TYPES DE MAGASINS
# ═══════════════════════════════════════════════════════════════════
elif page == pages[2]:
    st.title("🏪 Analyse par type de magasin")

    col1, col2, col3 = st.columns(3)
    for i, t in enumerate(["A", "B", "C"]):
        sub = df[df["Type"] == t]
        [col1, col2, col3][i].metric(
            f"Type {t}",
            f'{sub["Store"].nunique()} magasins',
            f'Moy: {sub["Weekly_Sales"].mean():,.0f} $'
        )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_box = px.box(df, x="Type", y="Weekly_Sales", color="Type",
                         color_discrete_map=COLOR_TYPE,
                         title="Distribution des ventes par type")
        fig_box.update_layout(height=450)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b:
        size_df = stores_info.groupby("Type")["Size"].mean().reset_index()
        fig_size = px.bar(size_df, x="Type", y="Size", color="Type",
                          color_discrete_map=COLOR_TYPE,
                          title="Taille moyenne par type", text_auto=".0f")
        fig_size.update_layout(yaxis_title="Taille (sq ft)", height=450)
        st.plotly_chart(fig_size, use_container_width=True)

    st.subheader("Performance par magasin")
    store_perf = (df.groupby(["Store", "Type"])["Weekly_Sales"]
                  .agg(["mean", "median", "std"]).reset_index()
                  .rename(columns={"mean": "Moyenne", "median": "Médiane", "std": "Écart-type"}))
    store_perf["CV"] = store_perf["Écart-type"] / store_perf["Moyenne"]
    store_perf = store_perf.sort_values("Moyenne", ascending=False)

    fig_store = px.bar(store_perf, x="Store", y="Moyenne", color="Type",
                       color_discrete_map=COLOR_TYPE,
                       title="Ventes moyennes par magasin (trié par performance)")
    fig_store.update_layout(xaxis=dict(dtick=1), height=400)
    st.plotly_chart(fig_store, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 4 — ANALYSE PAR DÉPARTEMENT
# ═══════════════════════════════════════════════════════════════════
elif page == pages[3]:
    st.title("📦 Analyse par département")
    st.markdown("Exploration des performances des départements de vente")

    # KPIs départements
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Départements actifs", f'{df["Dept"].nunique()}')
    top_dept = df.groupby("Dept")["Weekly_Sales"].sum().idxmax()
    c2.metric("Meilleur département", f'Dept {top_dept}')
    dept_moy = df.groupby("Dept")["Weekly_Sales"].mean()
    c3.metric("CA moyen / dept", f'{dept_moy.mean():,.0f} $')
    c4.metric("Depts rentables (moy > 0)", f'{(dept_moy > 0).sum()}')

    st.markdown("---")

    # Top 15 et flop 10 départements
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 15 départements par CA total")
        top15_dept = (df.groupby("Dept")["Weekly_Sales"].sum()
                      .nlargest(15).reset_index()
                      .rename(columns={"Weekly_Sales": "CA total ($)"}))
        fig_top_dept = px.bar(top15_dept, x="Dept", y="CA total ($)",
                              text_auto=".2s",
                              color_discrete_sequence=[COLOR_MAIN],
                              title="Top 15 départements par chiffre d'affaires")
        fig_top_dept.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig_top_dept, use_container_width=True)

    with col_b:
        st.subheader("Flop 10 départements par CA total")
        flop10_dept = (df.groupby("Dept")["Weekly_Sales"].sum()
                       .nsmallest(10).reset_index()
                       .rename(columns={"Weekly_Sales": "CA total ($)"}))
        fig_flop_dept = px.bar(flop10_dept, x="Dept", y="CA total ($)",
                               text_auto=".2s",
                               color_discrete_sequence=[COLOR_NEG],
                               title="Flop 10 départements (CA le plus faible)")
        fig_flop_dept.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig_flop_dept, use_container_width=True)

    # Saisonnalité par département (top 5)
    st.subheader("Saisonnalité mensuelle — Top 5 départements")
    top5_dept = (df.groupby("Dept")["Weekly_Sales"].sum()
                 .nlargest(5).index.tolist())
    dept_month = (df[df["Dept"].isin(top5_dept)]
                  .groupby(["Dept", "Month"])["Weekly_Sales"].mean().reset_index())
    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                    "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    dept_month["Mois"] = dept_month["Month"].map(lambda m: month_labels[m - 1])
    fig_dept_season = px.line(dept_month, x="Mois", y="Weekly_Sales",
                              color="Dept", color_discrete_sequence=px.colors.qualitative.Set2,
                              title="Profil saisonnier des 5 meilleurs départements")
    fig_dept_season.update_layout(xaxis=dict(categoryorder="array",
                                             categoryarray=month_labels))
    st.plotly_chart(fig_dept_season, use_container_width=True)

    # Impact promotions par département
    st.subheader("Impact des promotions par département (Top 15)")
    dept_promo = []
    for d in df["Dept"].unique():
        sub = df[df["Dept"] == d]
        avec = sub[sub["HasPromo"] == 1]["Weekly_Sales"].mean()
        sans = sub[sub["HasPromo"] == 0]["Weekly_Sales"].mean()
        if pd.notna(avec) and pd.notna(sans) and sans > 0:
            dept_promo.append({"Dept": d, "Lift (%)": (avec - sans) / sans * 100})
    if dept_promo:
        dept_promo_df = (pd.DataFrame(dept_promo)
                         .sort_values("Lift (%)", ascending=False)
                         .head(15))
        fig_dept_promo = px.bar(dept_promo_df, x="Dept", y="Lift (%)",
                                color="Lift (%)",
                                color_continuous_scale="RdYlGn",
                                text_auto="+.1f",
                                title="Lift promotion (%) — Top 15 départements")
        fig_dept_promo.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig_dept_promo, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 5 — IMPACT DES PROMOTIONS
# ═══════════════════════════════════════════════════════════════════
elif page == pages[4]:
    st.title("🏷️ Impact des promotions (MarkDowns)")

    md_cols = [c for c in df.columns if c.startswith("MarkDown")]

    c1, c2, c3 = st.columns(3)
    n_promo = df["HasPromo"].sum()
    pct_promo = n_promo / len(df) * 100
    moy_avec = df[df["HasPromo"] == 1]["Weekly_Sales"].mean()
    moy_sans = df[df["HasPromo"] == 0]["Weekly_Sales"].mean()
    lift = (moy_avec - moy_sans) / moy_sans * 100
    c1.metric("Semaines avec promo", f'{n_promo:,}', f'{pct_promo:.1f}%')
    c2.metric("Lift moyen", f'{lift:+.1f}%')
    c3.metric("Moy avec promo", f'{moy_avec:,.0f} $', f'vs {moy_sans:,.0f} $ sans')

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Avec vs Sans promotion")
        promo_comp = pd.DataFrame({
            "Statut": ["Sans promo", "Avec promo"],
            "Moyenne": [moy_sans, moy_avec]
        })
        fig_promo = px.bar(promo_comp, x="Statut", y="Moyenne",
                           color="Statut",
                           color_discrete_sequence=[COLOR_NEG, COLOR_POS],
                           text_auto=",.0f",
                           title="Ventes moyennes : avec vs sans promotion")
        st.plotly_chart(fig_promo, use_container_width=True)

    with col_b:
        st.subheader("Impact par MarkDown")
        md_impact = []
        for md in md_cols:
            has = df[df[md].notna() & (df[md] > 0)]
            no  = df[df[md].isna() | (df[md] <= 0)]
            if len(has) > 0 and len(no) > 0:
                l = (has["Weekly_Sales"].mean() - no["Weekly_Sales"].mean()) / no["Weekly_Sales"].mean() * 100
                md_impact.append({"MarkDown": md, "Lift (%)": l, "n": len(has)})
        md_df = pd.DataFrame(md_impact)
        if not md_df.empty:
            fig_md = px.bar(md_df, x="MarkDown", y="Lift (%)",
                            color="Lift (%)",
                            color_continuous_scale="RdYlGn",
                            text_auto="+.1f",
                            title="Lift (%) par variable MarkDown")
            st.plotly_chart(fig_md, use_container_width=True)

    st.subheader("Évolution temporelle des promotions")
    promo_ts = (df.groupby("Date")
                .agg(Moy_Ventes=("Weekly_Sales", "mean"),
                     Pct_Promo=("HasPromo", "mean"))
                .reset_index())
    fig_promo_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_promo_ts.add_trace(
        go.Scatter(x=promo_ts["Date"], y=promo_ts["Moy_Ventes"],
                   mode="lines", name="Ventes moy ($)",
                   line=dict(color=COLOR_MAIN)), secondary_y=False)
    fig_promo_ts.add_trace(
        go.Scatter(x=promo_ts["Date"], y=promo_ts["Pct_Promo"]*100,
                   mode="lines", name="% promo",
                   line=dict(color=COLOR_ACC, dash="dot")), secondary_y=True)
    fig_promo_ts.update_layout(title="Ventes et taux de promotion dans le temps", height=400)
    fig_promo_ts.update_yaxes(title_text="Ventes moy ($)", secondary_y=False)
    fig_promo_ts.update_yaxes(title_text="% semaines promo", secondary_y=True)
    st.plotly_chart(fig_promo_ts, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 5 — EFFET DES JOURS FÉRIÉS
# ═══════════════════════════════════════════════════════════════════
elif page == pages[5]:
    st.title("🎄 Effet des jours fériés sur les ventes")

    hol = df[df["IsHoliday"] == True]
    norm = df[df["IsHoliday"] == False]
    moy_hol = hol["Weekly_Sales"].mean()
    moy_norm = norm["Weekly_Sales"].mean()
    lift_hol = (moy_hol - moy_norm) / moy_norm * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Semaines fériées", f'{len(hol):,}', f'{len(hol)/len(df)*100:.1f}%')
    c2.metric("Lift jours fériés", f'{lift_hol:+.1f}%')
    c3.metric("Moy fériée", f'{moy_hol:,.0f} $', f'vs {moy_norm:,.0f} $')

    st.markdown("---")

    # Identifier les fêtes Walmart
    fetes_walmart = {
        "Super Bowl":    ["2010-02-12", "2011-02-11", "2012-02-10"],
        "Labor Day":     ["2010-09-10", "2011-09-09", "2012-09-07"],
        "Thanksgiving":  ["2010-11-26", "2011-11-25", "2012-11-23"],
        "Christmas":     ["2010-12-31", "2011-12-30", "2012-12-28"],
    }
    fete_data = []
    for fete, dates_list in fetes_walmart.items():
        for d_str in dates_list:
            d = pd.Timestamp(d_str)
            sub = df[df["Date"] == d]
            if len(sub) > 0:
                fete_data.append({"Fête": fete, "Date": d, "Moy_Ventes": sub["Weekly_Sales"].mean()})

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hol = px.box(df, x="IsHoliday", y="Weekly_Sales",
                         color="IsHoliday",
                         color_discrete_sequence=[COLOR_MAIN, COLOR_ACC],
                         title="Distribution : jours fériés vs normaux",
                         labels={"IsHoliday": "Jour férié"})
        st.plotly_chart(fig_hol, use_container_width=True)

    with col_b:
        if fete_data:
            fete_df = pd.DataFrame(fete_data)
            fete_avg = fete_df.groupby("Fête")["Moy_Ventes"].mean().reset_index()
            fete_avg["Lift (%)"] = (fete_avg["Moy_Ventes"] - moy_norm) / moy_norm * 100
            fete_avg = fete_avg.sort_values("Lift (%)", ascending=True)
            fig_fete = px.bar(fete_avg, x="Lift (%)", y="Fête", orientation="h",
                              color="Lift (%)", color_continuous_scale="YlOrRd",
                              title="Lift par fête Walmart",
                              text_auto="+.1f")
            st.plotly_chart(fig_fete, use_container_width=True)

    st.subheader("Impact par type de magasin")
    hol_type = (df.groupby(["Type", "IsHoliday"])["Weekly_Sales"]
                .mean().reset_index()
                .pivot(index="Type", columns="IsHoliday", values="Weekly_Sales")
                .rename(columns={False: "Normal", True: "Férié"})
                .reset_index())
    hol_type["Lift (%)"] = (hol_type["Férié"] - hol_type["Normal"]) / hol_type["Normal"] * 100
    fig_hol_type = px.bar(hol_type, x="Type", y="Lift (%)", color="Type",
                          color_discrete_map=COLOR_TYPE,
                          title="Lift jour férié par type de magasin", text_auto="+.1f")
    st.plotly_chart(fig_hol_type, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 6 — VARIABLES ÉCONOMIQUES
# ═══════════════════════════════════════════════════════════════════
elif page == pages[6]:
    st.title("🌡️ Variables économiques et saisonnières")

    eco_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    for i, col in enumerate(eco_cols):
        [c1, c2, c3, c4][i].metric(
            col,
            f'{df[col].mean():.1f}',
            f'σ = {df[col].std():.1f}'
        )

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🌡 Température", "⛽ Fuel Price", "📊 CPI", "📉 Chômage"])

    with tab1:
        df_temp = df.copy()
        df_temp["Temp_Bin"] = pd.cut(df_temp["Temperature"], bins=8)
        temp_sales = df_temp.groupby("Temp_Bin", observed=True)["Weekly_Sales"].mean().reset_index()
        temp_sales["Temp_Bin"] = temp_sales["Temp_Bin"].astype(str)
        fig_temp = px.bar(temp_sales, x="Temp_Bin", y="Weekly_Sales",
                          title="Ventes moyennes par tranche de température",
                          color_discrete_sequence=[COLOR_MAIN])
        st.plotly_chart(fig_temp, use_container_width=True)

        temp_type = df.groupby(["Month", "Type"]).agg(
            Moy_Temp=("Temperature", "mean"),
            Moy_Ventes=("Weekly_Sales", "mean")).reset_index()
        fig_tt = px.scatter(temp_type, x="Moy_Temp", y="Moy_Ventes",
                            color="Type", color_discrete_map=COLOR_TYPE,
                            title="Température vs Ventes par mois et type",
                            trendline="ols")
        st.plotly_chart(fig_tt, use_container_width=True)

    with tab2:
        df_fuel = df.copy()
        df_fuel["Fuel_Bin"] = pd.cut(df_fuel["Fuel_Price"], bins=8)
        fuel_sales = df_fuel.groupby("Fuel_Bin", observed=True)["Weekly_Sales"].mean().reset_index()
        fuel_sales["Fuel_Bin"] = fuel_sales["Fuel_Bin"].astype(str)
        fig_fuel = px.bar(fuel_sales, x="Fuel_Bin", y="Weekly_Sales",
                          title="Ventes moyennes par tranche de prix carburant",
                          color_discrete_sequence=[COLOR_ACC])
        st.plotly_chart(fig_fuel, use_container_width=True)

    with tab3:
        store_eco = df.groupby(["Store", "Type"]).agg(
            CPI_Moy=("CPI", "mean"), Ventes_Moy=("Weekly_Sales", "mean")).reset_index()
        fig_cpi = px.scatter(store_eco, x="CPI_Moy", y="Ventes_Moy",
                             color="Type", color_discrete_map=COLOR_TYPE,
                             size="Ventes_Moy", size_max=18,
                             title="CPI moyen vs Ventes moyennes par magasin",
                             trendline="ols")
        st.plotly_chart(fig_cpi, use_container_width=True)

    with tab4:
        store_unemp = df.groupby(["Store", "Type"]).agg(
            Unemp_Moy=("Unemployment", "mean"), Ventes_Moy=("Weekly_Sales", "mean")).reset_index()
        fig_unemp = px.scatter(store_unemp, x="Unemp_Moy", y="Ventes_Moy",
                               color="Type", color_discrete_map=COLOR_TYPE,
                               size="Ventes_Moy", size_max=18,
                               title="Chômage moyen vs Ventes moyennes par magasin",
                               trendline="ols")
        st.plotly_chart(fig_unemp, use_container_width=True)

    st.subheader("Corrélations globales avec Weekly_Sales")
    corr_eco = df[eco_cols + ["Weekly_Sales"]].corr()["Weekly_Sales"].drop("Weekly_Sales").sort_values(key=abs, ascending=True)
    fig_eco_bar = px.bar(x=corr_eco.values, y=corr_eco.index, orientation="h",
                         title="Corrélation Pearson : variables économiques ↔ ventes",
                         color=corr_eco.values,
                         color_continuous_scale="RdBu_r",
                         range_color=[-0.2, 0.2])
    fig_eco_bar.update_layout(xaxis_title="Pearson r", yaxis_title="", height=300)
    st.plotly_chart(fig_eco_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 7 — CORRÉLATIONS & FEATURES
# ═══════════════════════════════════════════════════════════════════
elif page == pages[7]:
    st.title("🔗 Corrélations entre variables et ventes")

    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in ("Store", "Dept", "Weekly_Sales")]

    # Pearson global
    pearson = df[num_cols + ["Weekly_Sales"]].corr()["Weekly_Sales"].drop("Weekly_Sales").sort_values(key=abs, ascending=False)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Classement Pearson")
        colors_p = ["#e74c3c" if v < 0 else "#2ecc71" for v in pearson.values]
        fig_rank = go.Figure(go.Bar(
            x=pearson.values, y=pearson.index, orientation="h",
            marker_color=colors_p,
            text=[f"{v:+.4f}" for v in pearson.values],
            textposition="outside"
        ))
        fig_rank.update_layout(yaxis=dict(autorange="reversed"),
                               height=max(400, len(pearson)*28),
                               margin=dict(l=140))
        st.plotly_chart(fig_rank, use_container_width=True)

    with col_b:
        st.subheader("Matrice de corrélation")
        key_vars = ["Weekly_Sales", "Size", "Temperature", "Fuel_Price",
                     "CPI", "Unemployment", "IsHoliday"]
        for md in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "Total_MarkDown"]:
            if md in df.columns:
                key_vars.append(md)
        key_vars = [v for v in key_vars if v in df.columns]
        corr_mat = df[key_vars].corr()
        fig_hm = go.Figure(go.Heatmap(
            z=corr_mat.values, x=key_vars, y=key_vars,
            colorscale="RdBu_r", zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr_mat.values],
            texttemplate="%{text}", textfont=dict(size=9)
        ))
        fig_hm.update_layout(height=550)
        st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("Pearson vs Spearman — Détection de non-linéarité")
    comp = []
    for col in num_cols:
        valid = df[["Weekly_Sales", col]].dropna()
        if len(valid) > 30:
            pr = valid["Weekly_Sales"].corr(valid[col])
            sr, _ = spearmanr(valid["Weekly_Sales"], valid[col])
            comp.append({"Variable": col, "Pearson": pr, "Spearman": sr,
                         "|Δ|": abs(sr - pr)})
    comp_df_dash = pd.DataFrame(comp).sort_values("|Δ|", ascending=False)

    fig_ps = go.Figure()
    fig_ps.add_trace(go.Scatter(
        x=comp_df_dash["Pearson"], y=comp_df_dash["Spearman"],
        mode="markers+text", text=comp_df_dash["Variable"],
        textposition="top center",
        marker=dict(size=10, color=comp_df_dash["|Δ|"],
                    colorscale="YlOrRd", showscale=True),
        textfont=dict(size=9)
    ))
    fig_ps.add_shape(type="line", x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
                     line=dict(dash="dash", color="gray"))
    fig_ps.update_layout(
        title="Pearson vs Spearman (écart à la diag. = non-linéarité)",
        xaxis_title="Pearson r", yaxis_title="Spearman ρ", height=500)
    st.plotly_chart(fig_ps, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 9 — RECOMMANDATIONS
# ═══════════════════════════════════════════════════════════════════
elif page == pages[8]:
    st.title("✅ Recommandations stratégiques")
    st.markdown("Synthèse des enseignements de l'analyse et leviers d'action pour Walmart")

    st.markdown("---")

    # Calcul de métriques pour contextualiser les recommandations
    top3_stores = (df.groupby(["Store", "Type"])["Weekly_Sales"].sum()
                   .nlargest(3).reset_index())
    top3_depts = (df.groupby("Dept")["Weekly_Sales"].sum()
                  .nlargest(3).index.tolist())
    moy_avec = df[df["HasPromo"] == 1]["Weekly_Sales"].mean()
    moy_sans = df[df["HasPromo"] == 0]["Weekly_Sales"].mean()
    lift_promo = (moy_avec - moy_sans) / moy_sans * 100 if moy_sans > 0 else 0
    hol = df[df["IsHoliday"] == True]
    norm = df[df["IsHoliday"] == False]
    lift_hol = (hol["Weekly_Sales"].mean() - norm["Weekly_Sales"].mean()) / norm["Weekly_Sales"].mean() * 100 if len(norm) > 0 else 0

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏪 Gestion des magasins")
        st.info(
            f"**Concentrer les investissements sur les magasins de type A.**\n\n"
            f"Les magasins A représentent la part la plus élevée du CA total. "
            f"Le top 3 des magasins (stores {', '.join(str(s) for s in top3_stores['Store'].tolist())}) "
            f"génèrent une part disproportionnée des revenus. "
            f"Il est recommandé de prioriser leurs réapprovisionnements et d'y concentrer les promotions."
        )

        st.subheader("📦 Gestion des départements")
        st.success(
            f"**Optimiser les stocks pour les départements clés.**\n\n"
            f"Les départements {', '.join(str(d) for d in top3_depts)} sont les plus performants. "
            f"Assurer leur disponibilité en stock, notamment en période de fêtes, "
            f"permettrait d'éviter les ruptures et de maximiser le CA."
        )

    with col2:
        st.subheader("🏷️ Stratégie promotionnelle")
        if lift_promo > 0:
            st.success(
                f"**Renforcer les campagnes de MarkDown.**\n\n"
                f"Les semaines avec promotion affichent un lift moyen de **{lift_promo:+.1f}%** "
                f"par rapport aux semaines sans promotion. "
                f"Il est conseillé d'amplifier les MarkDown 1 et 2 qui présentent le plus fort impact, "
                f"et de les synchroniser avec les périodes creuses pour lisser la demande."
            )
        else:
            st.warning(
                f"**Revoir la stratégie de MarkDown.**\n\n"
                f"Les promotions n'affichent pas de lift positif ({lift_promo:+.1f}%) sur la sélection actuelle. "
                f"Analyser les départements et magasins concernés pour cibler les promotions plus efficacement."
            )

        st.subheader("🎄 Planification des fêtes")
        st.info(
            f"**Anticiper les pics de Thanksgiving et Noël.**\n\n"
            f"Les jours fériés génèrent un lift moyen de **{lift_hol:+.1f}%** sur les ventes. "
            f"Thanksgiving est la fête la plus impactante. Il est recommandé d'augmenter les stocks "
            f"4 à 6 semaines avant ces périodes et de renforcer les effectifs en magasin."
        )

    st.markdown("---")
    st.subheader("🌡️ Adaptation aux variables économiques")
    st.markdown(
        """
        | Variable | Observation | Levier d'action |
        |---|---|---|
        | **Température** | Impact modéré sur les ventes | Adapter l'assortiment saisonnier (été/hiver) |
        | **Prix du carburant** | Légère corrélation négative | Surveiller les hausses pour ajuster les prix en rayon |
        | **CPI** | Corrélation variable selon le type de magasin | Positionner les produits d'entrée de gamme en période d'inflation |
        | **Taux de chômage** | Corrélation négative | Cibler les promotions dans les régions à fort chômage |
        """
    )

    st.markdown("---")
    st.subheader("📈 Synthèse des priorités")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**🥇 Priorité Haute**")
        st.markdown("- Anticiper Thanksgiving & Noël\n\n- Renforcer les magasins Type A\n\n- Optimiser MarkDown 1 & 2")
    with col_b:
        st.markdown("**🥈 Priorité Moyenne**")
        st.markdown(f"- Surveiller les Depts {', '.join(str(d) for d in top3_depts)}\n\n- Adapter aux cycles économiques\n\n- Lisser les promotions hors-saison")
    with col_c:
        st.markdown("**🥉 Priorité Basse**")
        st.markdown("- Optimiser les magasins Type C\n\n- Réduire les ventes négatives\n\n- Analyser les MarkDown 3, 4 et 5")

# ─────────────────── FOOTER ───────────────────
st.sidebar.markdown("---")
st.sidebar.caption("SIPV Walmart — IFM30546 | Session 4 | La Cité")

