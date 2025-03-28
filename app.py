# Install Streamlit (already installed, but included for completeness)
!pip install streamlit

# Install Cloudflared for tunneling
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

# Kill any existing Streamlit and Cloudflared processes to avoid conflicts
!pkill streamlit
!pkill cloudflared

# Write the Streamlit app to a file (app.py)
with open('app.py', 'w') as f:
    f.write('''
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MiniMap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import datetime

# Set page configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Load the data
file_path = '/content/coimbatore_city_crime_data_spatial.xlsx'
if not os.path.exists(file_path):
    st.error(f"Data file not found at {file_path}. Please ensure the file is uploaded.")
    st.stop()
else:
    df = pd.read_excel(file_path)

# Convert 'date' to datetime (Excel serial dates)
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    if pd.api.types.is_numeric_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], origin='1899-12-30', unit='d')
    else:
        raise ValueError("The 'date' column is neither datetime nor numeric. Please check the data.")

# Extract year and month from date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Prepare data for crime occurrence prediction
min_date = df['date'].min()
max_date = df['date'].max()
areas = df['area'].unique()
date_range = pd.date_range(start=min_date, end=max_date)
index = pd.MultiIndex.from_product([areas, date_range], names=['area', 'date'])
daily_crimes = df.groupby(['area', 'date'])['incidents'].sum().reset_index()
all_dates = pd.DataFrame(index=index).reset_index()
all_dates = all_dates.merge(daily_crimes, on=['area', 'date'], how='left')
all_dates['incidents'] = all_dates['incidents'].fillna(0)
all_dates['crime_occurred'] = (all_dates['incidents'] > 0).astype(int)

# Extract features for occurrence prediction
all_dates['year'] = all_dates['date'].dt.year
all_dates['month'] = all_dates['date'].dt.month
all_dates['day'] = all_dates['date'].dt.day
all_dates['day_of_week'] = all_dates['date'].dt.dayofweek
all_dates = pd.get_dummies(all_dates, columns=['area'], prefix='area')

# Features and target for occurrence model
X_occ = all_dates.drop(['date', 'incidents', 'crime_occurred'], axis=1)
y_occ = all_dates['crime_occurred']

# Train-test split (time-based)
train_mask = all_dates['date'] < '2022-01-01'
X_occ_train = X_occ[train_mask]
y_occ_train = y_occ[train_mask]
X_occ_test = X_occ[~train_mask]
y_occ_test = y_occ[~train_mask]

# Train crime occurrence model
model_occurrence = RandomForestClassifier(random_state=42, n_estimators=100)
model_occurrence.fit(X_occ_train, y_occ_train)

# Streamlit app layout
st.title("Coimbatore City Crime Dashboard")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a Page", ["Home", "Crime Mapping", "Interactive Dashboard", "Behavioral Analysis", "Patrol Optimization"])

# Home Page
if page == "Home":
    st.header("Welcome to the Cyber Crime Dashboard")
    st.write("Select an option from the sidebar to explore crime data for Coimbatore City (2020–2022).")
    st.markdown("""
    - **Crime Mapping**: Visualize crime patterns with different mapping techniques.
    - **Interactive Dashboard**: Explore crime trends over time with filters.
    - **Behavioral Analysis**: Predict crime types and occurrences using machine learning.
    - **Patrol Optimization**: Identify patrol areas with crime predictions.
    """)

# Crime Mapping
elif page == "Crime Mapping":
    st.header("Crime Mapping with Layer Buttons")
    center_lat = df['latitude'].mean()
    center_long = df['longitude'].mean()
    m = folium.Map(
        location=[center_lat, center_long],
        zoom_start=12,
        tiles='Stamen Terrain',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    )
    minimap = MiniMap()
    m.add_child(minimap)

    thematic_layer = folium.FeatureGroup(name="Thematic Mapping", show=True)
    for area, group in df.groupby('area'):
        total_incidents = group['incidents'].sum()
        avg_lat = group['latitude'].mean()
        avg_long = group['longitude'].mean()
        folium.Marker(
            location=[avg_lat, avg_long],
            popup=f"{area}: {total_incidents} incidents",
            icon=folium.Icon(color="blue")
        ).add_to(thematic_layer)
    thematic_layer.add_to(m)

    crime_counts = df['crime_type'].value_counts().to_dict()
    non_graphical_layer = folium.FeatureGroup(name="Non-Graphical Indicators", show=False)
    folium.Marker(
        location=[center_lat, center_long],
        popup=str(crime_counts),
        icon=folium.Icon(color="green")
    ).add_to(non_graphical_layer)
    non_graphical_layer.add_to(m)

    hotspot_layer = folium.FeatureGroup(name="Hot Spot Analysis", show=False)
    heat_data = [[row['latitude'], row['longitude'], row['incidents']] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=20, blur=10).add_to(hotspot_layer)
    hotspot_layer.add_to(m)

    spatial_regression_layer = folium.FeatureGroup(name="Spatial Regression", show=False)
    for area, group in df.groupby('area'):
        total_incidents = group['incidents'].sum()
        avg_lat = group['latitude'].mean()
        avg_long = group['longitude'].mean()
        if total_incidents > 5:
            folium.Marker(
                location=[avg_lat, avg_long],
                popup=f"{area}: High Risk (Predicted)",
                icon=folium.Icon(color="purple")
            ).add_to(spatial_regression_layer)
    spatial_regression_layer.add_to(m)

    geo_profiling_layer = folium.FeatureGroup(name="Geographic Profiling", show=False)
    murder_data = [[row['latitude'], row['longitude']] for _, row in df[df['crime_type'] == 'Murder'].iterrows()]
    HeatMap(murder_data, radius=20, blur=10).add_to(geo_profiling_layer)
    geo_profiling_layer.add_to(m)

    folium.LayerControl().add_to(m)
    with st.spinner("Loading map..."):
        st.components.v1.html(m._repr_html_(), height=600)

# Interactive Dashboard
elif page == "Interactive Dashboard":
    st.header("Interactive Dashboard (2020–2022)")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        total_incidents = df['incidents'].sum()
        st.metric("Total Incidents", f"{total_incidents:,}")
    with col2:
        most_common_crime = df['crime_type'].mode()[0]
        st.metric("Most Common Crime", most_common_crime)
    with col3:
        most_affected_area = df.groupby('area')['incidents'].sum().idxmax()
        st.metric("Most Affected Area", most_affected_area)
    with col4:
        total_crime_types = df['crime_type'].nunique()
        st.metric("Total Crime Types", total_crime_types)

    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        year_options = ['All Years'] + sorted(df['year'].unique().tolist())
        selected_year = st.selectbox("Select Year", year_options)
    with col2:
        crime_options = ['All Crimes'] + sorted(df['crime_type'].unique().tolist())
        selected_crime = st.selectbox("Select Crime Type", crime_options)
    with col3:
        area_options = ['All Areas'] + sorted(df['area'].unique().tolist())
        selected_area = st.selectbox("Select Area", area_options)
    with col4:
        date_range = st.slider(
            "Select Date Range",
            min_value=df['date'].min().to_pydatetime(),
            max_value=df['date'].max().to_pydatetime(),
            value=(df['date'].min().to_pydatetime(), df['date'].max().to_pydatetime()),
            format="YYYY-MM-DD"
        )

    filtered_df = df.copy()
    if selected_year != 'All Years':
        filtered_df = filtered_df[filtered_df['year'] == selected_year]
    if selected_crime != 'All Crimes':
        filtered_df = filtered_df[filtered_df['crime_type'] == selected_crime]
    if selected_area != 'All Areas':
        filtered_df = filtered_df[filtered_df['area'] == selected_area]
    filtered_df = filtered_df[
        (filtered_df['date'] >= pd.to_datetime(date_range[0])) &
        (filtered_df['date'] <= pd.to_datetime(date_range[1]))
    ]

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Top 10 Crime-Occurring Areas")
        area_incidents = df.groupby('area')['incidents'].sum().reset_index()
        prev_year = df['year'].max() - 1
        prev_year_df = df[df['year'] == prev_year].groupby('area')['incidents'].sum().reset_index()
        prev_year_df.rename(columns={'incidents': 'prev_incidents'}, inplace=True)
        area_incidents = area_incidents.merge(prev_year_df, on='area', how='left')
        area_incidents['prev_incidents'] = area_incidents['prev_incidents'].fillna(0)
        area_incidents['change'] = ((area_incidents['incidents'] - area_incidents['prev_incidents']) / area_incidents['prev_incidents'] * 100).replace([float('inf'), -float('inf')], 0)
        top_10_areas = area_incidents.sort_values(by='incidents', ascending=False).head(10)
        table_data = []
        for _, row in top_10_areas.iterrows():
            trend = "🔺" if row['change'] > 0 else "🔻" if row['change'] < 0 else "➖"
            trend_color = "red" if row['change'] > 0 else "orange" if row['change'] < 0 else "grey"
            table_data.append({
                "Area": row['area'],
                "Incidents": f"{int(row['incidents']):,}",
                "Trend": f"<span style='color:{trend_color}'>{trend} {row['change']:.1f}%</span>"
            })
        st.write(pd.DataFrame(table_data).to_html(escape=False, index=False), unsafe_allow_html=True)

    with col2:
        st.subheader("Crime Trends Over Time")
        trend_df = filtered_df.groupby(['date', 'crime_type'])['incidents'].sum().reset_index()
        fig_trend = px.line(
            trend_df,
            x='date',
            y='incidents',
            color='crime_type',
            title='Crime Incidents Over Time by Type',
            labels={'incidents': 'Number of Incidents', 'date': 'Date'},
            height=400
        )
        fig_trend.update_layout(
            legend_title_text='Crime Type',
            xaxis_title="Date",
            yaxis_title="Number of Incidents",
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("Geographical Distribution of Crimes")
        center_lat = df['latitude'].mean()
        center_long = df['longitude'].mean()
        m = folium.Map(
            location=[center_lat, center_long],
            zoom_start=12,
            tiles='Stamen Terrain',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
        )
        minimap = MiniMap()
        m.add_child(minimap)
        heat_data = [[row['latitude'], row['longitude'], row['incidents']] for _, row in filtered_df.iterrows()]
        HeatMap(heat_data, radius=20, blur=10).add_to(m)
        with st.spinner("Loading map..."):
            st.components.v1.html(m._repr_html_(), height=400)

    st.subheader("Crime Type Distribution")
    crime_dist = filtered_df.groupby('crime_type')['incidents'].sum().reset_index()
    fig_dist = px.bar(
        crime_dist,
        x='incidents',
        y='crime_type',
        orientation='h',
        title='Distribution of Crime Types',
        labels={'incidents': 'Number of Incidents', 'crime_type': 'Crime Type'},
        height=300
    )
    fig_dist.update_layout(
        xaxis_title="Number of Incidents",
        yaxis_title="Crime Type",
        showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Behavioral Analysis
elif page == "Behavioral Analysis":
    st.header("Behavioral Analysis")

    # Crime Type Prediction Model
    X_type = df[['year', 'month', 'latitude', 'longitude']]
    y_type = df['crime_type']
    X_type_train, X_type_test, y_type_train, y_type_test = train_test_split(X_type, y_type, test_size=0.2, random_state=42)
    model_type = RandomForestClassifier(random_state=42, n_estimators=100)
    model_type.fit(X_type_train, y_type_train)
    type_accuracy = model_type.score(X_type_test, y_type_test)

    # Crime Occurrence Prediction Accuracy
    occ_accuracy = model_occurrence.score(X_occ_test, y_occ_test)

    st.markdown("### Crime Occurrence Prediction")
    st.write(f"Model Accuracy: {occ_accuracy:.2f}")
    selected_area = st.selectbox("Select Area", areas)
    selected_date = st.date_input("Select Date", value=datetime.date.today())
    input_occ = pd.DataFrame({'area': [selected_area], 'date': [pd.to_datetime(selected_date)]})
    input_occ['year'] = input_occ['date'].dt.year
    input_occ['month'] = input_occ['date'].dt.month
    input_occ['day'] = input_occ['date'].dt.day
    input_occ['day_of_week'] = input_occ['date'].dt.dayofweek
    input_occ = pd.get_dummies(input_occ, columns=['area'], prefix='area')
    for col in X_occ_train.columns:
        if col not in input_occ.columns:
            input_occ[col] = 0
    input_occ = input_occ[X_occ_train.columns]
    prob = model_occurrence.predict_proba(input_occ)[:, 1][0]
    st.write(f"Predicted Probability of Crime Occurrence: {prob:.2f}")

    st.markdown("### Crime Type Prediction")
    st.write(f"Model Accuracy: {type_accuracy:.2f}")
    lat = df[df['area'] == selected_area]['latitude'].iloc[0]
    long = df[df['area'] == selected_area]['longitude'].iloc[0]
    input_type = pd.DataFrame({
        'year': [selected_date.year],
        'month': [selected_date.month],
        'latitude': [lat],
        'longitude': [long]
    })
    predicted_type = model_type.predict(input_type)[0]
    st.write(f"Predicted Most Likely Crime Type: {predicted_type}")

    # Feature Importance for Crime Type Model
    importances = pd.DataFrame({'feature': X_type.columns, 'importance': model_type.feature_importances_})
    st.write("Feature Importance for Crime Type Prediction:")
    st.dataframe(importances.sort_values(by='importance', ascending=False))

# Patrol Optimization
elif page == "Patrol Optimization":
    st.header("Patrol Optimization")

    # Metrics at the top (consistent with dashboard theme)
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        total_incidents = df['incidents'].sum()
        st.metric("Total Incidents", f"{total_incidents:,}")
    with col2:
        most_common_crime = df['crime_type'].mode()[0]
        st.metric("Most Common Crime", most_common_crime)
    with col3:
        most_affected_area = df.groupby('area')['incidents'].sum().idxmax()
        st.metric("Most Affected Area", most_affected_area)
    with col4:
        total_crime_types = df['crime_type'].nunique()
        st.metric("Total Crime Types", total_crime_types)

    # Date input for prediction
    current_time = datetime.datetime.now()
    st.write(f"Current Date and Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    selected_date = st.date_input("Select Date for Prediction", value=current_time.date())

    # Step 1: Create pred_df with 'area' and 'date'
    pred_df = pd.DataFrame({'area': areas, 'date': [pd.to_datetime(selected_date)] * len(areas)})
    pred_df['year'] = pred_df['date'].dt.year
    pred_df['month'] = pred_df['date'].dt.month
    pred_df['day'] = pred_df['date'].dt.day
    pred_df['day_of_week'] = pred_df['date'].dt.dayofweek

    # Step 2: Create a copy for prediction with one-hot encoding
    pred_df_for_pred = pred_df.copy()
    pred_df_for_pred = pd.get_dummies(pred_df_for_pred, columns=['area'], prefix='area')

    # Step 3: Align columns with X_occ_train for prediction
    for col in X_occ_train.columns:
        if col not in pred_df_for_pred.columns:
            pred_df_for_pred[col] = 0
    pred_df_for_pred = pred_df_for_pred[X_occ_train.columns]

    # Step 4: Make predictions
    probs = model_occurrence.predict_proba(pred_df_for_pred)[:, 1]

    # Step 5: Add predictions back to the original pred_df
    pred_df['probability'] = probs

    # Step 6: Merge with area_latlong to get latitude and longitude
    area_latlong = df.groupby('area')[['latitude', 'longitude']].first().reset_index()
    pred_df = pred_df.merge(area_latlong, on='area', how='left')

    # Step 7: Create enhanced map
    st.subheader("Predicted Crime Hotspots")
    center_lat = df['latitude'].mean()
    center_long = df['longitude'].mean()
    m = folium.Map(
        location=[center_lat, center_long],
        zoom_start=12,
        tiles='Stamen Terrain',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    )
    minimap = MiniMap()
    m.add_child(minimap)
    heat_data = [[row['latitude'], row['longitude'], row['probability']] for _, row in pred_df.iterrows()]
    HeatMap(
        heat_data,
        radius=20,
        blur=10,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
    ).add_to(m)

    # Add patrol markers for top 5 high-risk areas
    top_5 = pred_df.sort_values(by='probability', ascending=False).head(5)
    for _, row in top_5.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['area']}: {row['probability']:.2f}",
            icon=folium.Icon(color="red", icon="car", prefix="fa")
        ).add_to(m)

    with st.spinner("Loading map..."):
        st.components.v1.html(m._repr_html_(), height=600)

    # Display top 5 high-risk areas
    st.subheader("Top 5 High-Risk Areas")
    st.dataframe(top_5[['area', 'probability']])
    st.write("**Recommendation**: Deploy patrols to the above high-risk areas.")
''')

# Run Streamlit in the background
!streamlit run app.py &>/dev/null &

# Create a tunnel with Cloudflared
!./cloudflared tunnel --url http://localhost:8501
