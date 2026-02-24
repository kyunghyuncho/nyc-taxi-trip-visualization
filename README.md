# 🚖 NYC Taxi Dimensionality Reduction Demo

This is an interactive Streamlit application demonstrating nonlinear dimensionality reduction. The app fetches a subset of the NYC Yellow Taxi dataset, allows users to configure and train a PyTorch Lightning Autoencoder with a 2D bottleneck, and visualizes the resulting latent space using Plotly.

## 🌟 Features

*   **Interactive Data Fetching:** Pulls real-world data quickly from the NYC TLC Parquet data dump.
*   **Dynamic Data Preprocessing:** Cleans and processes the data, including One-Hot Encoding for categorical variables and `StandardScaler` for numerical ones.
*   **Configurable Autoencoder:** Train a PyTorch Lightning Autoencoder dynamically with Early Stopping and Denoising inputs. Customize:
    *   Input Features, Hidden Layers, Denoising Factor
    *   Learning Rate and Batch Size
    *   Optimizer (`Adam`, `SGD`, or a custom `Muon` implementation)
    *   Nonlinearity (`ReLU`, `Tanh`, `GELU`)
*   **Linear Baseline (PCA):** Skip neural network training altogether by checking the "Use PCA" box to calculate an exact orthogonal 2D projection using Scikit-Learn for comparison.
*   **Real-time Visualization:** Uses Plotly Express to vividly render the 2D bottleneck of the trained autoencoder.
    *   **Interactive Spotlighting:** Select subsets of categorical clusters from the UI to specifically highlight them, explicitly graying-out unselected clusters in the background.
    *   **Data Instance Cards:** Click on any point in the 2D scatter plot to instantly find its **10 Nearest Neighbors** in the mathematical latent space.
*   **Geospatial Mapping:** View the actual Pickup-to-Dropoff geographic routes of the selected point and its latent neighbors cleanly drawn onto an interactive **Folium NYC Map**. It dynamically parses centroids projected down from the official TLC Shapefile!

## 🚀 Getting Started

### Prerequisites

Ensure you have a Python environment manager installed (e.g., `uv`). We recommend `uv` for managing virtual environments and installing fast dependencies.

### Installation

1.  **Clone the repository:**
    *(Assuming you have the files locally)*

2.  **Set up the environment:**
    ```bash
    # Create the virtual environment using uv
    uv venv
    
    # Activate the environment (macOS/Linux)
    source ./.venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Configure API Key:**
    The application pulls from the Socrata NYC Open Data API. To avoid severe rate limiting, you should provide an App Token.
    
    *   Copy the template:
        ```bash
        cp .env.template .env
        ```
    *   Open `.env` and add your token:
        ```env
        SOCRATA_APP_TOKEN=your_token_here
        ```

### Running the App

Once your environment is active and dependencies are installed, start the Streamlit server:

```bash
streamlit run app.py
```

## 🧠 Project Architecture

*   **`app.py`**: The main entry point. Handles the Streamlit UI layout, state management (`st.session_state`), PCA baseline, KNN instance selection, Folium geospatial rendering, and the training execution logic.
*   **`data.py`**: Handles pulling data from the Socrata API, merging Map coordinate centroids, and applying `sklearn` transformers (OHE, StandardScaler, Train-Test split).
*   **`model.py`**: Defines the `Autoencoder` via PyTorch Lightning. Includes dynamic architecture building and Denoising injection logic.
*   **`get_centroids.py`**: Independent utility script using `geopandas` to parse the 260+ official TLC Taxi Zone shapefiles mapping Local IDs to physical Latitude/Longitude coordinate centroids.
*   **`optimizer.py`**: Contains a custom PyTorch implementation of the `Muon` (MomentUm Orthogonalized by Newton-schulz) optimizer.
*   **`viz.py`**: Wraps the configuration for `plotly.express.scatter` including the complex transparent styling for explicitly highlighting categorical subsets dynamically.

## 🛠 Tech Stack

*   **Frontend / UI:** [Streamlit](https://streamlit.io/)
*   **Machine Learning / Deep Learning:** [PyTorch](https://pytorch.org/) & [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/) & [Scikit-Learn](https://scikit-learn.org/)
*   **Geospatial & Visualization:** [Plotly Express](https://plotly.com/python/plotly-express/), [Folium](https://python-visualization.github.io/folium/), & [GeoPandas](https://geopandas.org/)