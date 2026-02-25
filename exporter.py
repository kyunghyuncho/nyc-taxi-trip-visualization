import io
import zipfile
import pandas as pd
import os
import base64

def create_stlite_export_zip(df, embeddings):
    """
    Takes the dynamically trained embeddings and original dataframe,
    merges them natively into a CSV, generates the `index.html` and 
    `static_app.py` wrapper, and zips them into a file buffer.
    """
    buffer = io.BytesIO()
    
    # Merge embeddings into df for easy transport
    export_df = df.copy()
    export_df['Dim 1'] = embeddings[:, 0]
    export_df['Dim 2'] = embeddings[:, 1]
    
    # Generate CSV in memory - optimize size by dropping unused complex cols if needed
    # But for EDA, we keep all of them.
    csv_buffer = io.BytesIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()

    # Load static assets
    viz_content = ""
    with open("viz.py", "r", encoding="utf-8") as f:
        viz_content = f.read()
        
    centroids_bytes = b""
    if os.path.exists("taxi_zones_centroids.csv"):
        with open("taxi_zones_centroids.csv", "rb") as f:
            centroids_bytes = f.read()

    # Create static_app.py logic
    # This strips out PyTorch, `data.py`, and `model.py` but keeps the Dashboards
    static_app_content = """import sys
import types
try:
    import pyarrow as pa
    if not hasattr(pa, 'ChunkedArray'):
        pa.ChunkedArray = type('ChunkedArray', (), {})
    if not hasattr(pa, 'Table'):
        pa.Table = type('Table', (), {})
except ImportError:
    pa = types.ModuleType('pyarrow')
    pa.ChunkedArray = type('ChunkedArray', (), {})
    pa.Table = type('Table', (), {})
    sys.modules['pyarrow'] = pa

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import folium
from streamlit_folium import st_folium
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from viz import plot_embeddings

st.set_page_config(page_title="NYC Taxi Dimensionality Reduction Export", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("export_data.csv")
    embeddings = df[['Dim 1', 'Dim 2']].values
    return df, embeddings

st.title("🚖 NYC Taxi Dimensionality Reduction (Standalone Export)")
st.markdown("This beautifully self-contained application is running **100% inside your browser** via WebAssembly (stlite). It visualizes NYC Taxi data mathematically clustered into a 2-Dimensional learned representation!")

with st.spinner("Loading embedded dataset..."):
    df, embeddings = load_data()

# --- Sidebar ---
st.sidebar.header("Visualization Setup")
all_features = [c for c in df.columns if c not in ['Dim 1', 'Dim 2']]
color_column = st.sidebar.selectbox("Color Mapping Feature", options=all_features, index=all_features.index('pu_borough') if 'pu_borough' in all_features else 0)

highlight_categories = None
if color_column in df.columns and df[color_column].dtype == 'object':
    unique_cats = sorted([str(x) for x in pd.Series(df[color_column].unique()).dropna().tolist()])
    highlight_categories = st.sidebar.multiselect(
        f"Highlight specific {color_column}s", 
        options=unique_cats, 
        default=unique_cats,
        help="Categories not selected will be grayed out in the plot. If all are removed, everything is grayed out."
    )

st.markdown("---")
tab1, tab2 = st.tabs(["Latent Space Visualization", "Data Analysis"])

with tab1:
    st.subheader("Latent Space Visualization")
    
    hover_cols = [col for col in ['trip_distance', 'fare_amount', 'pickup_hour'] if col in df.columns]

    fig = plot_embeddings(
        df=df,
        embeddings=embeddings,
        color_column=color_column,
        hover_cols=hover_cols,
        highlight_categories=highlight_categories
    )
    
    event = st.plotly_chart(
        fig, 
        use_container_width=True, 
        on_select="rerun", 
        selection_mode="points"
    )

    selected_points = []
    if event:
        if hasattr(event, 'selection'):
            sel = event.selection
            if isinstance(sel, dict):
                selected_points = sel.get('points', [])
            elif hasattr(sel, 'points'):
                selected_points = getattr(sel, 'points', [])
        elif isinstance(event, dict) and 'selection' in event:
            selected_points = event['selection'].get('points', [])
            
    if selected_points:
        st.markdown("### Selected Point & 10 Nearest Neighbors")
        
        clicked_x = selected_points[0].get('x') if isinstance(selected_points[0], dict) else getattr(selected_points[0], 'x', None)
        clicked_y = selected_points[0].get('y') if isinstance(selected_points[0], dict) else getattr(selected_points[0], 'y', None)
        
        if clicked_x is not None and clicked_y is not None:
            nn = NearestNeighbors(n_neighbors=11)
            nn.fit(embeddings)
            
            distances, indices = nn.kneighbors([[clicked_x, clicked_y]])
            neighbor_indices = indices[0]
            
            def render_card(row, distance=None, is_ref=False):
                with st.container(border=True):
                    if is_ref:
                        st.markdown("**🔴 Reference Point**")
                    else:
                        st.caption(f"Latent Distance: {distance:.4f}")
                    
                    cols = st.columns(3)
                    cols[0].metric("Fare Amount", f"${row.get('fare_amount', 0):.2f}")
                    cols[1].metric("Trip Distance", f"{row.get('trip_distance', 0):.2f} mi")
                    cols[2].metric("Total Amount", f"${row.get('total_amount', 0):.2f}")
                    
                    st.markdown(f"**Pickup:** {row.get('pu_borough', 'N/A')} ({row.get('pu_zone', 'N/A')}) at {row.get('pickup_hour', 'N/A')}:00")
                    st.markdown(f"**Dropoff:** {row.get('do_borough', 'N/A')} ({row.get('do_zone', 'N/A')})")
                    st.markdown(f"**Payment:** {row.get('payment_type', 'N/A')} | **Day:** {row.get('pickup_dayofweek', 'N/A')}")
            
            ref_idx = neighbor_indices[0]
            ref_row = df.iloc[ref_idx]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                render_card(ref_row, is_ref=True)
                if len(neighbor_indices) > 1:
                    st.markdown("#### Closest Neighbor")
                    idx = neighbor_indices[1]
                    dist = distances[0][1]
                    render_card(df.iloc[idx], distance=dist)
                    
            with col2:
                st.markdown("#### Geographic Trip Mapping")
                m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="CartoDB dark_matter")
                has_valid_cords = False
                
                def add_trip_to_map(row, color, weight, opacity, label_prefix):
                    try:
                        pLat = float(row.get('pu_lat'))
                        pLon = float(row.get('pu_lon'))
                        dLat = float(row.get('do_lat'))
                        dLon = float(row.get('do_lon'))
                        
                        if pd.notna(pLat) and pd.notna(pLon) and pd.notna(dLat) and pd.notna(dLon):
                            popup_html = f"<b>{label_prefix}</b><br>Fare: ${row.get('fare_amount', 0):.2f}<br>Dist: {row.get('trip_distance', 0):.2f} mi<br>PU: {row.get('pu_zone')}<br>DO: {row.get('do_zone')}"
                            folium.PolyLine(
                                locations=[(pLat, pLon), (dLat, dLon)],
                                color=color, weight=weight, opacity=opacity,
                                popup=folium.Popup(popup_html, max_width=250),
                                tooltip=f"{label_prefix} Trip"
                            ).add_to(m)
                            
                            folium.CircleMarker(location=(pLat, pLon), radius=4, color="green", fill=True, fillOpacity=1, popup="Pickup").add_to(m)
                            folium.CircleMarker(location=(dLat, dLon), radius=4, color="white", fill=True, fillOpacity=1, popup="Dropoff").add_to(m)
                            return True
                    except Exception:
                        pass
                    return False
    
                for i in range(1, len(neighbor_indices)):
                    idx = neighbor_indices[i]
                    success = add_trip_to_map(df.iloc[idx], color="#00ffff", weight=2, opacity=0.4, label_prefix=f"Neighbor {i}")
                    if success: has_valid_cords = True
    
                success = add_trip_to_map(ref_row, color="#ff0000", weight=4, opacity=0.8, label_prefix="Reference")
                if success: has_valid_cords = True
                
                if has_valid_cords:
                    st_folium(m, use_container_width=True, height=450)
                else:
                    st.warning("Coordinates not available for these trips to map geographically.")
                    
            st.markdown("#### Neighborhood Statistics")
            neighbors_df = df.iloc[neighbor_indices]
            hist_cols = st.columns(3)
            with hist_cols[0]:
                fig_dist = px.histogram(neighbors_df, x="trip_distance", title="Trip Distances (mi)", nbins=10, template="plotly_dark", color_discrete_sequence=["#00ffff"])
                fig_dist.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_dist, use_container_width=True)
            with hist_cols[1]:
                fig_fare = px.histogram(neighbors_df, x="fare_amount", title="Fare Amounts ($)", nbins=10, template="plotly_dark", color_discrete_sequence=["#00ffff"])
                fig_fare.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_fare, use_container_width=True)
                
            with hist_cols[2]:
                fig_hour = px.histogram(neighbors_df, x="pickup_hour", title="Pickup Hours", nbins=10, template="plotly_dark", color_discrete_sequence=["#00ffff"])
                fig_hour.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(dtick=1))
                st.plotly_chart(fig_hour, use_container_width=True)
                    
    else:
        st.info("Click on any point in the scatter plot above to view its geographic trip and nearest neighbors.")

with tab2:
    st.subheader("Data Analysis")
    st.write("Explore the distributions and relationships of the raw dataset.")
    
    st.markdown("### 1D Distribution (Histogram)")
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numerical_cols:
        hist_var = st.selectbox("Select Numerical Feature for Histogram", options=numerical_cols)
        fig_1d = px.histogram(df, x=hist_var, title=f"Distribution of {hist_var}", template="plotly_dark", nbins=50)
        st.plotly_chart(fig_1d, use_container_width=True)
    else:
        st.warning("No numerical columns found for histograms.")
        
    st.markdown("---")
    st.markdown("### 2D Relationship (Scatter Plot)")
    
    col_x, col_y, col_color = st.columns(3)
    
    with col_x:
        scatter_x = st.selectbox("X-Axis Feature", options=all_features, index=all_features.index('trip_distance') if 'trip_distance' in all_features else 0)
    with col_y:
        scatter_y = st.selectbox("Y-Axis Feature", options=all_features, index=all_features.index('fare_amount') if 'fare_amount' in all_features else 0)
    with col_color:
        scatter_c = st.selectbox("Color Mapping", options=all_features, index=all_features.index('pu_borough') if 'pu_borough' in all_features else 0)
        
    plot_sample_size = min(5000, len(df))
    df_sample = df.sample(n=plot_sample_size, random_state=42)
    
    fig_2d = px.scatter(
        df_sample, x=scatter_x, y=scatter_y, color=scatter_c,
        title=f"{scatter_y} vs {scatter_x}", template="plotly_dark", opacity=0.6, render_mode='webgl'
    )
    fig_2d.update_traces(marker=dict(size=4, line=dict(width=0.2, color='DarkSlateGrey')))
    if pd.api.types.is_numeric_dtype(df_sample[scatter_c]) and scatter_c in ['trip_distance', 'fare_amount', 'total_amount', 'tip_amount']:
         df_sample[f'Log1p({scatter_c})'] = np.log1p(df_sample[scatter_c].clip(lower=0))
         fig_2d = px.scatter(
             df_sample, x=scatter_x, y=scatter_y, color=f'Log1p({scatter_c})',
             title=f"{scatter_y} vs {scatter_x}", template="plotly_dark", opacity=0.6, render_mode='webgl'
         )
         fig_2d.update_traces(marker=dict(size=4, line=dict(width=0.2, color='DarkSlateGrey')))
         
    st.plotly_chart(fig_2d, use_container_width=True)
    st.caption(f"Scatter plot showing {plot_sample_size} random samples for performance.")
"""

    # Encode files to base64 for safe inline JS injection without breaking f-strings/quotes
    app_b64 = base64.b64encode(static_app_content.encode('utf-8')).decode('utf-8')
    viz_b64 = base64.b64encode(viz_content.encode('utf-8')).decode('utf-8')
    csv_b64 = base64.b64encode(csv_bytes).decode('utf-8')
    
    centroids_b64 = ""
    if len(centroids_bytes) > 0:
        centroids_b64 = base64.b64encode(centroids_bytes).decode('utf-8')

    index_html_content = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <title>NYC Taxi Netlify Export</title>
    <!-- Use the latest stlite build -->
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.55.0/build/stlite.css"
    />
  </head>
  <body>
    <div id="root"></div>
    
    <script>
      // Helper function to decode base64 back into a Uint8Array
      function b64ToUint8Array(base64) {{
          var binary_string = window.atob(base64);
          var len = binary_string.length;
          var bytes = new Uint8Array(len);
          for (var i = 0; i < len; i++) {{
              bytes[i] = binary_string.charCodeAt(i);
          }}
          return bytes;
      }}
    </script>
    <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.55.0/build/stlite.js"></script>
    <script>
      stlite.mount(
        {{
          requirements: ["pandas", "plotly", "scikit-learn", "folium", "streamlit-folium"],
          entrypoint: "static_app.py",
          files: {{
            "static_app.py": {{ data: b64ToUint8Array("{app_b64}") }},
            "viz.py": {{ data: b64ToUint8Array("{viz_b64}") }},
            "export_data.csv": {{ data: b64ToUint8Array("{csv_b64}") }}
            {f', "taxi_zones_centroids.csv": {{ data: b64ToUint8Array("{centroids_b64}") }}' if centroids_b64 else ''}
          }},
        }},
        document.getElementById("root")
      );
    </script>
  </body>
</html>"""

    # Create the zip bundle
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("static_app.py", static_app_content)
        zf.writestr("index.html", index_html_content)
        zf.writestr("viz.py", viz_content)
        zf.writestr("export_data.csv", csv_bytes)
        if len(centroids_bytes) > 0:
            zf.writestr("taxi_zones_centroids.csv", centroids_bytes)
            
    buffer.seek(0)
    return buffer
