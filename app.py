import streamlit as st
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from torch.utils.data import TensorDataset, DataLoader
import plotly.express as px
from sklearn.decomposition import PCA

# Local imports
from data import fetch_data, preprocess_data, transform_data
from model import Autoencoder, StreamlitProgressCallback
from viz import plot_embeddings
from sklearn.neighbors import NearestNeighbors
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="NYC Taxi Dimensionality Reduction", layout="wide")

# --- UI Setup ---
st.title("🚖 NYC Taxi Dimensionality Reduction Demo")
st.markdown("This app uses a PyTorch Lightning Autoencoder to learn a 2-dimensional latent space representation of NYC Taxi data. Play around with the features and hyperparameters to see how they affect the learned clusters.")

# Fetch and prep data
with st.spinner("Fetching Data..."):
    raw_df = fetch_data(limit=10000)

if raw_df.empty:
    st.error("Failed to fetch data or dataset is empty.")
    st.stop()

df = preprocess_data(raw_df)

# Store original dataframe for hovering
st.session_state.setdefault('df', df)

# --- Sidebar ---
st.sidebar.header("1. Data Configuration")

# Possible features to select
all_features = df.columns.tolist()
# Typical defaults that make sense for a taxi trip
default_features = ['trip_distance', 'fare_amount', 'total_amount', 'pickup_hour', 'pickup_dayofweek', 'payment_type', 'pu_borough', 'do_borough']
available_defaults = [f for f in default_features if f in all_features]

selected_features = st.sidebar.multiselect(
    "Select Input Features", 
    options=all_features,
    default=available_defaults
)

st.sidebar.header("2. Hyperparameters")
use_pca = st.sidebar.checkbox("Use PCA (Linear Autoencoder)", value=False, help="Skips neural network training and uses Standard PCA.")
learning_rate = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f", step=0.0005, disabled=use_pca)
optimizer_name = st.sidebar.selectbox("Optimizer", options=['Adam', 'Muon', 'SGD'], disabled=use_pca)
nonlinearity = st.sidebar.selectbox("Nonlinearity", options=['ReLU', 'Tanh', 'GELU'], disabled=use_pca)
epochs = st.sidebar.slider("Epochs (Max)", min_value=10, max_value=150, value=50, step=10, disabled=use_pca)
batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64, 128], index=2, disabled=use_pca)
hidden_layers = st.sidebar.text_input("Hidden Layers (comma separated)", value="64, 32", disabled=use_pca)

st.sidebar.header("3. Denoising & Regularization")
noise_factor = st.sidebar.slider("Input Noise Factor", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Standard deviation of Gaussian noise added to inputs during training (0 = Standard Autoencoder).", disabled=use_pca)
early_stopping_patience = st.sidebar.slider("Early Stopping Patience", min_value=3, max_value=20, value=5, help="Stop training if validation loss doesn't improve for this many epochs.", disabled=use_pca)

st.sidebar.header("4. Visualization Setup")
color_column = st.sidebar.selectbox("Color Mapping Feature", options=all_features, index=all_features.index('pu_borough') if 'pu_borough' in all_features else 0)

highlight_categories = None
if color_column in df.columns and df[color_column].dtype == 'object':
    unique_cats = sorted(df[color_column].astype(str).dropna().unique().tolist())
    highlight_categories = st.sidebar.multiselect(
        f"Highlight specific {color_column}s",
        options=unique_cats,
        default=unique_cats[:5] if len(unique_cats) > 5 else unique_cats,
        help="Categories not selected will be grayed out in the plot. If all are removed, everything is grayed out."
    )

# Main Page - Display a sample of the data
st.subheader("Data Overview")
st.dataframe(df.head(10))
st.caption(f"Total Rows Ready for Training: {len(df)}")

# --- Training Logic ---
button_text = "Run PCA" if use_pca else "Train Autoencoder"

if st.button(button_text):
    if not selected_features:
        st.error("Please select at least one input feature.")
    else:
        # Prepare Data
        with st.spinner("Transforming Data..."):
            try:
                X_train_tensor, X_val_tensor, preprocessor, num_numeric, cat_sizes = transform_data(df, selected_features)
            except Exception as e:
                st.error(f"Error during transformation: {e}")
                st.stop()

            input_dim = X_train_tensor.shape[1]
            train_dataset = TensorDataset(X_train_tensor)
            val_dataset = TensorDataset(X_val_tensor)
            
            # Using the whole dataset for final embeddings rendering
            full_dataset_tensor = torch.cat([X_train_tensor, X_val_tensor], dim=0)
            full_dataset = TensorDataset(full_dataset_tensor)
            
            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            inference_loader = DataLoader(full_dataset, batch_size=512, shuffle=False)

        if use_pca:
            with st.spinner("Running PCA..."):
                pca = PCA(n_components=2)
                # PCA doesn't need PyTorch batches, fit directly on the whole preprocessed tensor
                all_embeddings = pca.fit_transform(full_dataset_tensor.numpy())
                
                st.session_state['embeddings'] = all_embeddings
                st.session_state['trained'] = True
                st.success("PCA Complete!")
        else:
            st.subheader("Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize Model
            model = Autoencoder(
                input_dim=input_dim,
                num_numeric_features=num_numeric,
                cat_sizes=cat_sizes,
                hidden_layers=hidden_layers,
                lr=learning_rate,
                optimizer_name=optimizer_name,
                nonlinearity_name=nonlinearity,
                input_noise_factor=noise_factor
            )

            st_callback = StreamlitProgressCallback(progress_bar, status_text, epochs)
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=early_stopping_patience, verbose=False, mode="min")

            # PyTorch Lightning Trainer
            trainer = pl.Trainer(
                max_epochs=epochs,
                callbacks=[st_callback, early_stop_callback],
                enable_checkpointing=False,
                logger=False, # Disable TensorBoard logger for simplicity
                accelerator='auto', # uses GPU/MPS if available
                devices=1,
                enable_progress_bar=False # Disable default tqdm
            )

            # Train
            with st.spinner("Training..."):
                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # Extract Embeddings
            with st.spinner("Extracting Embeddings..."):
                embeddings_list = []
                for batch in inference_loader:
                    x = batch[0]
                    emb = model.get_embeddings(x)
                    embeddings_list.append(emb)
                
                all_embeddings = torch.cat(embeddings_list, dim=0).numpy()

                # Orthogonalize the latent space using PCA
                with st.spinner("Applying PCA (Orthogonalization)..."):
                    pca = PCA(n_components=2)
                    all_embeddings = pca.fit_transform(all_embeddings)

                # Store in session state
                st.session_state['embeddings'] = all_embeddings
                st.session_state['trained'] = True
                
                st.success("Training Complete!")

# --- Render Plot ---
st.markdown("---")
st.subheader("Latent Space Visualization")

if st.session_state.get('trained', False):
    embeddings = st.session_state['embeddings']
    
    # We want a few useful columns for hover data
    hover_cols = [col for col in ['trip_distance', 'fare_amount', 'pickup_hour'] if col in df.columns]

    fig = plot_embeddings(
        df=st.session_state['df'],
        embeddings=embeddings,
        color_column=color_column,
        hover_cols=hover_cols,
        highlight_categories=highlight_categories
    )
    
    # Render with selection events enabled
    event = st.plotly_chart(
        fig, 
        use_container_width=True, 
        on_select="rerun", 
        selection_mode="points"
    )

    # --- Data Instance Cards (KNN) ---
    selected_points = []
    if event:
        # Safely extract points depending on how Streamlit wraps the event
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
        
        # Get the first clicked point coordinates
        clicked_x = selected_points[0].get('x') if isinstance(selected_points[0], dict) else getattr(selected_points[0], 'x', None)
        clicked_y = selected_points[0].get('y') if isinstance(selected_points[0], dict) else getattr(selected_points[0], 'y', None)
        
        if clicked_x is None or clicked_y is None:
            st.warning("Could not extract coordinates from the selected point.")
            st.stop()
            
        # Find Nearest Neighbors in the 2D latent space
        nn = NearestNeighbors(n_neighbors=11) # 1 for the point itself + 10 neighbors
        nn.fit(embeddings)
        
        distances, indices = nn.kneighbors([[clicked_x, clicked_y]])
        neighbor_indices = indices[0]
        
        # Helper to render a row as a card
        def render_card(row, distance=None, is_ref=False):
            with st.container(border=True):
                if is_ref:
                    st.markdown("**🔴 Reference Point**")
                else:
                    st.caption(f"Latent Distance: {distance:.4f}")
                
                # Highlight the key metrics
                cols = st.columns(3)
                cols[0].metric("Fare Amount", f"${row.get('fare_amount', 0):.2f}")
                cols[1].metric("Trip Distance", f"{row.get('trip_distance', 0):.2f} mi")
                cols[2].metric("Total Amount", f"${row.get('total_amount', 0):.2f}")
                
                # Show categorical info
                st.markdown(f"**Pickup:** {row.get('pu_borough', 'N/A')} ({row.get('pu_zone', 'N/A')}) at {row.get('pickup_hour', 'N/A')}:00")
                st.markdown(f"**Dropoff:** {row.get('do_borough', 'N/A')} ({row.get('do_zone', 'N/A')})")
                st.markdown(f"**Payment:** {row.get('payment_type', 'N/A')} | **Day:** {row.get('pickup_dayofweek', 'N/A')}")
        
        # Render the Reference Point Card
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
            
            # Create a Map containing all the relevant lines
            # Default center NYC
            m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="CartoDB dark_matter")
            has_valid_cords = False
            
            # Helper to draw trip
            def add_trip_to_map(row, color, weight, opacity, label_prefix):
                try:
                    pLat = float(row.get('pu_lat'))
                    pLon = float(row.get('pu_lon'))
                    dLat = float(row.get('do_lat'))
                    dLon = float(row.get('do_lon'))
                    
                    if pd.notna(pLat) and pd.notna(pLon) and pd.notna(dLat) and pd.notna(dLon):
                        # Add line between pickup and dropoff
                        popup_html = f"<b>{label_prefix}</b><br>Fare: ${row.get('fare_amount', 0):.2f}<br>Dist: {row.get('trip_distance', 0):.2f} mi<br>PU: {row.get('pu_zone')}<br>DO: {row.get('do_zone')}"
                        folium.PolyLine(
                            locations=[(pLat, pLon), (dLat, dLon)],
                            color=color,
                            weight=weight,
                            opacity=opacity,
                            popup=folium.Popup(popup_html, max_width=250),
                            tooltip=f"{label_prefix} Trip"
                        ).add_to(m)
                        
                        # Add Start and End markers as tiny circles
                        folium.CircleMarker(location=(pLat, pLon), radius=4, color="green", fill=True, fillOpacity=1, popup="Pickup").add_to(m)
                        folium.CircleMarker(location=(dLat, dLon), radius=4, color="white", fill=True, fillOpacity=1, popup="Dropoff").add_to(m)
                        return True
                except Exception:
                    pass
                return False

            # Draw neighbors first (so they are under the reference)
            for i in range(1, len(neighbor_indices)):
                idx = neighbor_indices[i]
                row = df.iloc[idx]
                success = add_trip_to_map(row, color="#00ffff", weight=2, opacity=0.4, label_prefix=f"Neighbor {i}")
                if success: has_valid_cords = True

            # Draw reference line last (over top)
            success = add_trip_to_map(ref_row, color="#ff0000", weight=4, opacity=0.8, label_prefix="Reference")
            if success: has_valid_cords = True
            
            if has_valid_cords:
                st_folium(m, use_container_width=True, height=450, returned_objects=[])
            else:
                st.warning("Coordinates not available for these trips to map geographically. Ensure the shapefile was processed successfully.")
                
        # Neighborhood Stats
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

else:
    st.info("Click 'Train Autoencoder' or 'Run PCA' to generate the visualization.")
