import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---- Page Configuration ----
st.set_page_config(
    page_title="Netflix Titles EDA Dashboard", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS for Better Styling ----
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #E50914;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #0f1419;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎬 Netflix Titles EDA Dashboard</h1>', unsafe_allow_html=True)

# ---- Load and Prepare Data with Error Handling ----
@st.cache_data
def load_data():
    try:
        # Update this path to your actual CSV file location
        df = pd.read_csv(r"D:\Netflix Project\netflix_titles_cleaned.csv")  
        
        # Data preprocessing
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        
        # Extract year from date_added if year_added doesn't exist
        if "year_added" not in df.columns:
            df["year_added"] = df["date_added"].dt.year
        else:
            df["year_added"] = pd.to_numeric(df["year_added"], errors="coerce")
        
        # Extract duration in minutes for movies
        if 'duration' in df.columns:
            df['duration_mins'] = df.loc[df['type'].str.lower() == "movie", "duration"].str.extract(r"(\d+)").astype(float)
        
        # Clean country data
        df['country'] = df['country'].fillna('Unknown')
        
        # Clean missing values
        df['rating'] = df['rating'].fillna('Not Rated')
        df['listed_in'] = df['listed_in'].fillna('Unknown Genre')
        
        return df
    except FileNotFoundError:
        st.error("❌ Netflix dataset file not found! Please ensure 'netflix_titles_cleaned.csv' is in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_data()

if df.empty:
    st.stop()

# ---- Dataset Information ----
st.sidebar.markdown("## 📊 Dataset Info")
st.sidebar.info(f"""
**Total Records:** {len(df):,}  
**Date Range:** {df['release_year'].min():.0f} - {df['release_year'].max():.0f}  
**Countries:** {df['country'].nunique()}  
**Genres:** {len(set([g.strip() for genres in df['listed_in'].str.split(', ') for g in genres]))}
""")

# ---- Sidebar Filters ----
st.sidebar.markdown("## 🔍 Filter Data")

# Type filter
all_types = sorted(df['type'].dropna().unique())
selected_types = st.sidebar.multiselect(
    "Content Type", 
    all_types, 
    default=all_types,
    help="Select content types to include"
)

# Country filter
country_list = sorted({c.strip() for clist in df['country'].dropna().str.split(', ') for c in clist if c.strip() != 'Unknown'})
selected_countries = st.sidebar.multiselect(
    "Country", 
    country_list,
    help="Leave empty to include all countries"
)

# Year range filter
year_range = st.sidebar.slider(
    "Release Year Range",
    min_value=int(df['release_year'].min()),
    max_value=int(df['release_year'].max()),
    value=(int(df['release_year'].min()), int(df['release_year'].max())),
    step=1
)

# Rating filter
all_ratings = sorted(df['rating'].dropna().unique())
selected_ratings = st.sidebar.multiselect(
    "Content Rating",
    all_ratings,
    help="Leave empty to include all ratings"
)

# ---- Data Filtering ----
mask = pd.Series(True, index=df.index)

if selected_types:
    mask &= df['type'].isin(selected_types)

if selected_countries:
    mask &= df['country'].apply(lambda x: any(c in (x or "") for c in selected_countries))

mask &= (df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])

if selected_ratings:
    mask &= df['rating'].isin(selected_ratings)

filtered = df[mask].copy()

# ---- Key Metrics ----
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📽️ Total Titles",
        value=f"{len(filtered):,}",
        delta=f"{len(filtered) - len(df):,}" if len(filtered) != len(df) else None
    )

with col2:
    if len(filtered) > 0:
        movies_pct = (filtered['type'] == 'Movie').mean() * 100
        st.metric(
            label="🎬 Movies %",
            value=f"{movies_pct:.1f}%"
        )

with col3:
    if len(filtered) > 0:
        recent_content = (filtered['release_year'] >= 2020).sum()
        st.metric(
            label="🆕 Recent Content (2020+)",
            value=f"{recent_content:,}"
        )

with col4:
    if len(filtered) > 0:
        countries_count = len({c.strip() for clist in filtered['country'].str.split(', ') for c in clist})
        st.metric(
            label="🌍 Countries",
            value=f"{countries_count:,}"
        )

st.markdown("---")

# ---- Summary Statistics ----
with st.expander("📈 Summary Statistics", expanded=False):
    if len(filtered) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Release Year**")
            st.write(f"• Mean: {filtered['release_year'].mean():.1f}")
            st.write(f"• Median: {filtered['release_year'].median():.1f}")
            st.write(f"• Mode: {filtered['release_year'].mode().iloc[0]:.0f}")
        
        with col2:
            if filtered['year_added'].notna().any():
                st.markdown("**Year Added**")
                st.write(f"• Mean: {filtered['year_added'].mean():.1f}")
                st.write(f"• Median: {filtered['year_added'].median():.1f}")
                st.write(f"• Mode: {filtered['year_added'].mode().iloc[0]:.0f}")
        
        with col3:
            if 'duration_mins' in filtered.columns and filtered['duration_mins'].notna().any():
                st.markdown("**Movie Duration (mins)**")
                st.write(f"• Mean: {filtered['duration_mins'].mean():.1f}")
                st.write(f"• Median: {filtered['duration_mins'].median():.1f}")
                st.write(f"• Mode: {filtered['duration_mins'].mode().iloc[0]:.0f}")

# ---- Main Visualizations ----
col1, col2 = st.columns(2)

# Content Type Distribution
with col1:
    st.subheader("📊 Content Type Distribution")
    if len(filtered) > 0:
        type_counts = filtered["type"].value_counts()
        fig_bar = px.bar(
            x=type_counts.index, 
            y=type_counts.values,
            labels={"x": "Content Type", "y": "Count"},
            title="Movies vs TV Shows",
            text=type_counts.values,
            color=type_counts.index,
            color_discrete_sequence=['#E50914', '#F5F5F1']
        )
        fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No data to display")

# Top Genres
with col2:
    st.subheader("🎭 Top 10 Genres")
    if len(filtered) > 0:
        genres = filtered['listed_in'].dropna().str.split(', ')
        top_genres = pd.Series([g.strip() for sub in genres for g in sub]).value_counts().head(10)
        fig_genres = px.bar(
            x=top_genres.values,
            y=top_genres.index,
            orientation='h',
            labels={"x": "Count", "y": "Genre"},
            title="Most Popular Genres",
            text=top_genres.values,
            color=top_genres.values,
            color_continuous_scale='Reds'
        )
        fig_genres.update_traces(texttemplate='%{text}', textposition='outside')
        fig_genres.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_genres, use_container_width=True)
    else:
        st.warning("No data to display")

# Content Share Pie Chart
col1, col2 = st.columns(2)

with col1:
    st.subheader("🥧 Content Type Share")
    if len(filtered) > 0:
        type_counts = filtered["type"].value_counts()
        fig_pie = px.pie(
            names=type_counts.index,
            values=type_counts.values,
            title="Movies vs TV Shows Distribution",
            hole=0.4,
            color_discrete_sequence=['#E50914', '#F5F5F1']
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("🌍 Top 5 Countries")
    if len(filtered) > 0:
        country_counts = filtered["country"].str.split(', ').explode()
        top_countries = country_counts.value_counts().head(5)
        fig_pie_countries = px.pie(
            names=top_countries.index,
            values=top_countries.values,
            hole=0.4,
            title="Top Content Producing Countries",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie_countries.update_traces(textinfo="percent+label")
        fig_pie_countries.update_layout(height=400)
        st.plotly_chart(fig_pie_countries, use_container_width=True)

# Release Year Distribution
st.subheader("📅 Release Year Distribution")
if len(filtered) > 0:
    fig_hist = px.histogram(
        filtered,
        x="release_year",
        color="type",
        nbins=30,
        barmode="group",
        labels={"release_year": "Release Year", "count": "Number of Titles"},
        title="Content Release Timeline",
        color_discrete_sequence=['#E50914', '#F5F5F1']
    )
    fig_hist.update_layout(height=500, bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

# Netflix Growth Over Time
if "year_added" in filtered.columns and filtered['year_added'].notna().any():
    st.subheader("📈 Netflix Content Growth")
    year_added_counts = filtered.groupby("year_added").size().reset_index()
    year_added_counts.columns = ['Year', 'Titles Added']
    
    fig_area = px.area(
        year_added_counts,
        x='Year',
        y='Titles Added',
        title="Netflix Content Addition Over Time",
        color_discrete_sequence=['#E50914']
    )
    fig_area.update_layout(height=400)
    st.plotly_chart(fig_area, use_container_width=True)

# Advanced Visualizations
col1, col2 = st.columns(2)

with col1:
    # Movie Duration Analysis
    if "duration_mins" in filtered.columns and filtered['duration_mins'].notna().any():
        st.subheader("⏱️ Movie Duration Analysis")
        movies = filtered[filtered['type'].str.lower() == 'movie']
        fig_box = px.box(
            movies,
            y='duration_mins',
            x='rating',
            title="Movie Duration by Rating",
            color='rating',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

with col2:
    # Content Rating Distribution
    st.subheader("🔖 Content Rating Distribution")
    if len(filtered) > 0:
        rating_counts = filtered['rating'].value_counts()
        fig_rating = px.bar(
            x=rating_counts.values,
            y=rating_counts.index,
            orientation='h',
            title="Content by Rating",
            text=rating_counts.values,
            color=rating_counts.values,
            color_continuous_scale='Blues'
        )
        fig_rating.update_traces(texttemplate='%{text}', textposition='outside')
        fig_rating.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rating, use_container_width=True)

# Geographic Visualization
st.subheader("🗺️ Global Netflix Content Distribution")
if len(filtered) > 0:
    country_counts = filtered["country"].str.split(', ').explode()
    country_title_count = country_counts.value_counts().reset_index()
    country_title_count.columns = ["country", "num_titles"]
    
    # Filter out unknown countries
    country_title_count = country_title_count[country_title_count['country'] != 'Unknown']
    
    if not country_title_count.empty:
        fig_map = px.choropleth(
            country_title_count,
            locations='country',
            locationmode='country names',
            color='num_titles',
            color_continuous_scale='Reds',
            title='Netflix Content Distribution by Country',
            labels={'country': 'Country', 'num_titles': 'Number of Titles'},
            hover_name='country'
        )
        fig_map.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
        fig_map.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

# Data Insights
st.subheader("💡 Key Insights")
if len(filtered) > 0:
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**Content Composition:**")
        movie_count = (filtered['type'] == 'Movie').sum()
        tv_count = (filtered['type'] == 'TV Show').sum()
        st.write(f"• Movies: {movie_count:,} ({movie_count/len(filtered)*100:.1f}%)")
        st.write(f"• TV Shows: {tv_count:,} ({tv_count/len(filtered)*100:.1f}%)")
        
        # Peak year
        peak_year = filtered['release_year'].mode().iloc[0]
        st.write(f"• Peak Release Year: {peak_year:.0f}")
    
    with insights_col2:
        st.markdown("**Geographic Reach:**")
        unique_countries = len({c.strip() for clist in filtered['country'].str.split(', ') for c in clist})
        st.write(f"• Countries Represented: {unique_countries}")
        
        top_country = filtered['country'].str.split(', ').explode().mode().iloc[0]
        st.write(f"• Top Producer: {top_country}")
        
        # Most common rating
        top_rating = filtered['rating'].mode().iloc[0]
        st.write(f"• Most Common Rating: {top_rating}")

# Data Table
with st.expander("📋 Data Table (First 100 Rows)", expanded=False):
    if len(filtered) > 0:
        display_columns = ['title', 'type', 'country', 'release_year', 'rating', 'listed_in', 'duration']
        available_columns = [col for col in display_columns if col in filtered.columns]
        
        st.dataframe(
            filtered[available_columns].head(100),
            use_container_width=True,
            height=400
        )
        
        # Download functionality
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=csv,
            file_name=f"Netflix_Filtered_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the current filtered dataset"
        )
    else:
        st.warning("No data available for the selected filters.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Netflix Titles EDA Dashboard | Built with Streamlit & Plotly</p>
    <p>Data insights updated in real-time based on your filter selections</p>
</div>
""", unsafe_allow_html=True)
