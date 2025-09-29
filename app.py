import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import base64

# ------------------- Page config -------------------
st.set_page_config(page_title="☀️ Sunlytics 2.0", layout="wide", initial_sidebar_state="expanded")

# ------------------- Background & fonts -------------------
def set_bg(png_file):
    try:
        with open(png_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except Exception:
        b64 = ""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Merriweather:wght@700&display=swap');
    .stApp {{
        font-family: 'Lato', sans-serif;
        background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
        background-size: cover;
        position: relative;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top:0; left:0; right:0; bottom:0;
        background: rgba(0,0,0,0.55);
        z-index: 0;
    }}
    .stApp * {{ position: relative; z-index: 1; color: white; }}
    h1 {{ font-family: 'Merriweather', serif; font-weight:700; color:#FFD700; }}
    h2 {{ font-weight:700; color:#FFA500; }}
    h3 {{ font-weight:500; color:#FFFFFF; }}
    p, li {{ font-family: 'Lato', sans-serif; font-size:15px; color:white; }}
    .stMetric > div {{ text-align:center; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg("sunshine.jpeg")

# ------------------- Data loading + type safety -------------------
@st.cache_data
def load_urban_data(path="URBAN DATA.csv"):
    df = pd.read_csv(path)
    numeric_guess = ['slab_max_kwh','rate_per_kwh','area_m2','centroid_lat','centroid_lon','rooftop_area_deg2','payback_years']
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    for m in months:
        numeric_guess.append(m+'_kwh')

    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)
    for col in set(numeric_guess).intersection(df.columns):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # create monthly estimations if missing
    for m in months:
        col = m + '_kwh'
        if col not in df.columns:
            if 'slab_max_kwh' in df.columns:
                df[col] = df['slab_max_kwh'] * np.random.uniform(0.07, 0.10, size=len(df))
            else:
                df[col] = np.random.uniform(1, 10, size=len(df))

    df['rate_per_kwh'] = pd.to_numeric(df.get('rate_per_kwh', 0), errors='coerce').fillna(0)
    df['Monthly_Savings'] = (df['slab_max_kwh'].fillna(0) * df['rate_per_kwh']).astype(float)
    df['CO2'] = (df['slab_max_kwh'].fillna(0) * 0.82).astype(float)
    df['Utilization_%'] = (df['slab_max_kwh'].fillna(0) / df['area_m2'].replace({0:np.nan})).fillna(0)*100
    # payback years safe
    if 'payback_years' not in df.columns:
        df['payback_years'] = (df['Monthly_Savings'].replace(0,np.nan).fillna(1)).apply(lambda x: round(1000/x,1))

    return df, months

@st.cache_data
def load_village_data(path="village_data.csv"):
    df = pd.read_csv(path)
    if 'village_name' in df.columns:
        df['village_name'] = df['village_name'].astype(str)
    numeric_cols = ['latitude','longitude','distance_to_road','percent_households_electrified',
                    'hours_supply','solar_potential','energy_deficit','potential_savings_per_household',
                    'payback_years','co2_reduction','urgency_score','total_population','num_households','num_schools']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'distance_to_road' in df.columns:
        df['accessibility_score'] = df['distance_to_road'].apply(lambda x: 10 - min(x/2, 10) if not pd.isna(x) else np.nan)
    else:
        df['accessibility_score'] = np.nan
    return df

# Load datasets
urban_df, months = load_urban_data()
village_df = load_village_data()

# ------------------- Compute Metrics -------------------
@st.cache_data
def compute_urban_metrics(df):
    total_energy = df['slab_max_kwh'].fillna(0).sum()
    total_savings = df['Monthly_Savings'].fillna(0).sum()
    total_co2 = df['CO2'].fillna(0).sum()
    avg_energy = df['slab_max_kwh'].mean()
    top10 = df.nlargest(10, 'slab_max_kwh')[['id','slab_max_kwh','Monthly_Savings','CO2','Utilization_%','payback_years','building_type']].round(2)
    top_sum = df.nlargest(10,'slab_max_kwh')['slab_max_kwh'].sum()
    other_sum = total_energy - top_sum
    return {
        "total_energy": total_energy,
        "total_savings": total_savings,
        "total_co2": total_co2,
        "avg_energy": avg_energy,
        "top10": top10,
        "top_sum": top_sum,
        "other_sum": other_sum,
        "n_buildings": len(df)
    }

@st.cache_data
def compute_village_metrics(df):
    avg_urgency = df['urgency_score'].mean()
    avg_access = df['accessibility_score'].mean()
    low_elec = int(df[df['percent_households_electrified'] < 50].shape[0]) if 'percent_households_electrified' in df.columns else 0
    urgency_series = df['urgency_score'].apply(lambda x: 'High' if x>=7 else ('Medium' if x>=4 else 'Low')).value_counts()
    top10 = df.sort_values('urgency_score', ascending=False).head(10)[['village_name','percent_households_electrified','hours_supply','energy_deficit','solar_potential','urgency_score']]
    return {
        "avg_urgency": avg_urgency,
        "avg_access": avg_access,
        "low_elec": low_elec,
        "urgency_series": urgency_series,
        "top10": top10,
        "n_villages": len(df)
    }

urban_metrics = compute_urban_metrics(urban_df)
village_metrics = compute_village_metrics(village_df)

# ------------------- Pie Charts -------------------
@st.cache_data
def create_building_type_pie(df):
    type_counts = df['building_type'].value_counts()
    fig = px.pie(values=type_counts.values, names=type_counts.index, title="Buildings by Type",
                 color_discrete_sequence=px.colors.sequential.Plasma, hole=0.4)
    return fig

@st.cache_data
def create_village_elec_pie(df):
    bins = [0,25,50,75,100]
    labels = ['0-25%','26-50%','51-75%','76-100%']
    df['elec_bin'] = pd.cut(df['percent_households_electrified'], bins=bins, labels=labels, include_lowest=True)
    counts = df['elec_bin'].value_counts().reindex(labels).fillna(0)
    fig = px.pie(values=counts.values, names=counts.index, title="Village Electrification %",
                 color_discrete_sequence=px.colors.sequential.Plasma, hole=0.4)
    return fig

# ------------------- Maps -------------------
@st.cache_data
def create_village_map(df):
    center_lat = df['latitude'].mean() if 'latitude' in df.columns else 0
    center_lon = df['longitude'].mean() if 'longitude' in df.columns else 0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")
    marker_cluster = MarkerCluster().add_to(m)
    for r in df.itertuples(index=False):
        lat = getattr(r, 'latitude', None)
        lon = getattr(r, 'longitude', None)
        name = getattr(r, 'village_name', 'village')
        urgency = getattr(r, 'urgency_score', None)
        if pd.isna(lat) or pd.isna(lon):
            continue
        color = 'red' if (urgency is not None and urgency>=7) else ('orange' if (urgency is not None and urgency>=4) else 'green')
        folium.CircleMarker(location=[lat, lon],
                            radius=6, color=color, fill=True, fill_opacity=0.7,
                            popup=f"{name} — Urgency: {urgency}").add_to(marker_cluster)
    return m

# ------------------- Sidebar -------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard", ["Welcome", "Urban", "Village"])

# ------------------- WELCOME PAGE -------------------
if page=="Welcome":
    st.markdown("<h1 style='text-align:center;'>☀️ Sunlytics 2.0</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#FFDFA0;'>Data-Driven Insights for Sustainable Energy Planning</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Urban Dashboard")
        st.metric("Total Buildings", urban_metrics['n_buildings'])
        st.metric("Total Energy (kWh/month)", f"{urban_metrics['total_energy']:,.0f}")
        st.metric("Total Savings (₹/month)", f"{urban_metrics['total_savings']:,.0f}")
        st.metric("Total CO₂ Avoided (kg/month)", f"{urban_metrics['total_co2']:,.0f}")
        fig_u = create_building_type_pie(urban_df)
        st.plotly_chart(fig_u, width='stretch')

    with col2:
        st.subheader("Village Dashboard")
        st.metric("Total Villages", village_metrics['n_villages'])
        st.metric("Average Urgency Score", f"{village_metrics['avg_urgency']:.1f}")
        st.metric("Average Accessibility Score", f"{village_metrics['avg_access']:.1f}")
        st.metric("Villages <50% Electrified", village_metrics['low_elec'])
        fig_v = create_village_elec_pie(village_df)
        st.plotly_chart(fig_v, width='stretch')

# ------------------- URBAN DASHBOARD -------------------
elif page=="Urban":
    urban_page = st.sidebar.radio("Urban Navigation", ["Overview","Building Details","Comparison"])
    
    if urban_page=="Overview":
        col1,col2=st.columns(2)
        with col1:
            st.subheader("Citywide Key Metrics")
            st.metric("Total Buildings", urban_metrics['n_buildings'])
            st.metric("Total Energy (kWh/month)", f"{urban_metrics['total_energy']:,.0f}")
            st.metric("Total Savings (₹/month)", f"{urban_metrics['total_savings']:,.0f}")
            st.metric("Total CO₂ Avoided (kg/month)", f"{urban_metrics['total_co2']:,.0f}")
        with col2:
            st.subheader("Building Type Distribution")
            fig = create_building_type_pie(urban_df)
            st.plotly_chart(fig, width='stretch')
    
    elif urban_page=="Building Details":
        building_id = st.selectbox("Select Building ID", urban_df['id'].unique())
        b = urban_df.loc[urban_df['id']==building_id].iloc[0]
        st.subheader(f"Building ID: {building_id}")
        # 👥 Demographics
        st.markdown("### 👥 Demographics")
        st.write(f"Building Type: {b.get('building_type','N/A')}")
        st.write(f"Rooftop Area (m²): {b.get('area_m2','N/A')}")
        # ⚡ Energy
        st.markdown("### ⚡ Energy")
        st.write(f"Predicted Energy (kWh/month): {b['slab_max_kwh']:.1f}")
        st.write(f"Monthly Savings (₹): {b['Monthly_Savings']:.1f}")
        st.write(f"Utilization (%): {b['Utilization_%']:.1f}")
        # 🌍 Sustainability
        st.markdown("### 🌍 Sustainability")
        st.write(f"CO₂ Avoided (kg/month): {b['CO2']:.1f}")
        st.write(f"Payback Years: {b.get('payback_years','N/A')}")
        # 🗺️ Map
        st.markdown("### 🗺️ Location")
        if not pd.isna(b.get('centroid_lat')) and not pd.isna(b.get('centroid_lon')):
            m = folium.Map(location=[b['centroid_lat'],b['centroid_lon']], zoom_start=16)
            folium.Marker([b['centroid_lat'],b['centroid_lon']],
                          popup=f"Building {building_id}\nEnergy: {b['slab_max_kwh']:.0f} kWh").add_to(m)
            st_folium(m, width=700, height=400)
        # Policy Recommendations
        st.markdown("### Policy Recommendations")
        st.write("• MNRE subsidy (where applicable) may reduce up-front cost.")
        st.write("• Encourage high-efficiency solar panels to maximize savings.")

    elif urban_page=="Comparison":
        ids = st.multiselect("Select Buildings to compare", urban_df['id'].unique())
        if ids:
            comp = urban_df.loc[urban_df['id'].isin(ids)].copy()
            metrics = ['slab_max_kwh','Monthly_Savings','CO2','Utilization_%','payback_years']
            melt = comp.melt(id_vars='id', value_vars=metrics, var_name='Metric', value_name='Value')
            bar_fig = px.bar(melt, x='Metric', y='Value', color='id', barmode='group', title="Building Comparison")
            st.plotly_chart(bar_fig, width='stretch')
            st.dataframe(comp[['id']+metrics].round(2), width=700)

# ------------------- VILLAGE DASHBOARD -------------------
elif page=="Village":
    village_page = st.sidebar.radio("Village Navigation", ["Overview","Village Details","Comparison"])
    
    if village_page=="Overview":
        col1,col2=st.columns(2)
        with col1:
            st.subheader("Village Metrics")
            st.metric("Total Villages", village_metrics['n_villages'])
            st.metric("Average Urgency Score", f"{village_metrics['avg_urgency']:.1f}")
            st.metric("Average Accessibility Score", f"{village_metrics['avg_access']:.1f}")
            st.metric("Villages <50% Electrified", village_metrics['low_elec'])
        with col2:
            st.subheader("Electrification Distribution")
            fig = create_village_elec_pie(village_df)
            st.plotly_chart(fig, width='stretch')
        st.subheader("Village Map")
        v_map = create_village_map(village_df)
        st_folium(v_map, width=900, height=450)
        st.subheader("Top 10 Villages by Urgency")
        st.dataframe(village_metrics['top10'], width=900)
    
    elif village_page=="Village Details":
        v_name = st.selectbox("Select Village", village_df['village_name'].unique())
        v = village_df.loc[village_df['village_name']==v_name].iloc[0]
        st.subheader(f"Village: {v_name}")
        # 👥 Demographics
        st.markdown("### 👥 Demographics")
        st.write(f"Population: {v.get('total_population','N/A')}")
        st.write(f"Households: {v.get('num_households','N/A')}")
        st.write(f"Schools: {v.get('num_schools','N/A')}")
        # ⚡ Energy
        st.markdown("### ⚡ Energy & Electrification")
        st.write(f"% Electrified: {v.get('percent_households_electrified','N/A')}")
        st.write(f"Hours of Supply: {v.get('hours_supply','N/A')}")
        st.write(f"Energy Deficit: {v.get('energy_deficit','N/A')}")
        # 🌍 Sustainability
        st.markdown("### 🌍 Sustainability")
        st.write(f"CO₂ Reduction (kg/month): {v.get('co2_reduction','N/A')}")
        st.write(f"Payback Years: {v.get('payback_years','N/A')}")
        # 🗺️ Map
        st.markdown("### 🗺️ Map")
        if not pd.isna(v.get('latitude')) and not pd.isna(v.get('longitude')):
            m = folium.Map(location=[v['latitude'],v['longitude']], zoom_start=13)
            folium.Marker([v['latitude'],v['longitude']],
                          popup=f"{v_name}\nElectrified: {v.get('percent_households_electrified','N/A')}%").add_to(m)
            st_folium(m, width=700, height=400)
        # Policy Recommendations
        st.markdown("### Policy Recommendations")
        st.write("• Microgrid deployment for villages <50% electrified.")
        st.write("• Subsidy advice via MNRE schemes.")
        st.write("• Priority intervention for high urgency villages.")

    elif village_page=="Comparison":
        vnames = st.multiselect("Select Villages to compare", village_df['village_name'].unique())
        if vnames:
            comp = village_df.loc[village_df['village_name'].isin(vnames)].copy()
            metrics = ['percent_households_electrified','energy_deficit','urgency_score','accessibility_score']
            melt = comp.melt(id_vars='village_name', value_vars=metrics, var_name='Metric', value_name='Value')
            bar_fig = px.bar(melt, x='Metric', y='Value', color='village_name', barmode='group', title="Village Comparison")
            st.plotly_chart(bar_fig, width='stretch')
            st.dataframe(comp[['village_name']+metrics].round(2), width=900)
