import streamlit as st
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
import plotly.express as px

# Local imports
from data import fetch_data, preprocess_data, transform_data
from model import Autoencoder, StreamlitProgressCallback
from viz import plot_embeddings

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
default_features = ['trip_distance', 'fare_amount', 'total_amount', 'pickup_hour', 'pickup_dayofweek', 'payment_type']
available_defaults = [f for f in default_features if f in all_features]

selected_features = st.sidebar.multiselect(
    "Select Input Features", 
    options=all_features,
    default=available_defaults
)

st.sidebar.header("2. Hyperparameters")
learning_rate = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f", step=0.0005)
optimizer_name = st.sidebar.selectbox("Optimizer", options=['Adam', 'Muon', 'SGD'])
nonlinearity = st.sidebar.selectbox("Nonlinearity", options=['ReLU', 'Tanh', 'GELU'])
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=150, value=30, step=10)
batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64, 128], index=2)
hidden_layers = st.sidebar.text_input("Hidden Layers (comma separated)", value="64, 32")

st.sidebar.header("3. Visualization Setup")
color_column = st.sidebar.selectbox("Color Mapping Feature", options=all_features, index=all_features.index('payment_type') if 'payment_type' in all_features else 0)

# Main Page - Display a sample of the data
st.subheader("Data Overview")
st.dataframe(df.head(10))
st.caption(f"Total Rows Ready for Training: {len(df)}")

# --- Training Logic ---
if st.button("Train Autoencoder"):
    if not selected_features:
        st.error("Please select at least one input feature.")
    else:
        # Prepare Data
        with st.spinner("Transforming Data..."):
            try:
                X_tensor, preprocessor = transform_data(df, selected_features)
            except Exception as e:
                st.error(f"Error during transformation: {e}")
                st.stop()

            input_dim = X_tensor.shape[1]
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            inference_loader = DataLoader(dataset, batch_size=512, shuffle=False)

        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize Model
        model = Autoencoder(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            lr=learning_rate,
            optimizer_name=optimizer_name,
            nonlinearity_name=nonlinearity
        )

        st_callback = StreamlitProgressCallback(progress_bar, status_text, epochs)

        # PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[st_callback],
            enable_checkpointing=False,
            logger=False, # Disable TensorBoard logger for simplicity
            accelerator='auto', # uses GPU/MPS if available
            devices=1,
            enable_progress_bar=False # Disable default tqdm
        )

        # Train
        with st.spinner("Training..."):
            trainer.fit(model, dataloader)

        # Extract Embeddings
        with st.spinner("Extracting Embeddings..."):
            embeddings_list = []
            for batch in inference_loader:
                x = batch[0]
                emb = model.get_embeddings(x)
                embeddings_list.append(emb)
            
            all_embeddings = torch.cat(embeddings_list, dim=0).numpy()

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
        hover_cols=hover_cols
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click 'Train Autoencoder' to generate the visualization.")
