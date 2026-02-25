# 🚖 NYC Taxi Dimensionality Reduction Dashboard

This is an interactive Streamlit application demonstrating nonlinear dimensionality reduction applied to the NYC Yellow Taxi dataset. The application empowers users to train a custom PyTorch Lightning Autoencoder, dynamically explore the learned latent representations within a visually rich space, and thoroughly analyze the raw dataset variables.

## 🌟 Core Features

### 1. Latent Space Visualization
Uses a **PyTorch Lightning Autoencoder** to compress complex taxi trips into a 2D bottleneck and maps those embeddings interactively.
*   **Dynamic Data Preprocessing**: Leverages Scikit-Learn pipelines to automatically clean invalid rows, apply One-Hot Encoding for categorical features, apply `StandardScaler` to numerical variables, and dynamically inject `log1p` transformations for heavily skewed features like trip distance and fares.
*   **Configurable Autoencoder**: Train an Autoencoder dynamically with Early Stopping and Denoising inputs. Customize:
    *   Input Features, Hidden Layers, Denoising Factor
    *   Learning Rate and Batch Size
    *   Optimizer (`Adam`, `SGD`, or custom `Muon`)
    *   Nonlinearity (`ReLU`, `Tanh`, `GELU`)
*   **Optional Orthogonalization (PCA)**: Toggle Scikit-Learn PCA on or off before rendering to force the learned Neural Network embeddings to be orthogonal to one another, or disable Neural Networks altogether to use a pure PCA baseline.
*   **Real-time Visualization**: Uses WebGL-accelerated Plotly Express to vividly render the bottleneck embeddings. You can specifically highlight and single out exact categorical clusters on the fly over a grayed-out background.

### 2. Deep Dive Neighborhood Analysis
What happens when you click on a cluster point in the Latent Space?
*   **K-Nearest Neighbors Extraction**: Instantly computes the **10 Nearest Neighbors** of the specifically clicked coordinate within the latent mathematical space.
*   **Data Instance Cards**: Renders localized statistics (Fare, Payment, Pickups) and explicit distributions (Histograms for distance, fare, and hour) for the specific neighborhood vs the reference point.
*   **Geospatial Mapping**: Draws the exact real-world pickup-to-dropoff driving paths of the matched neighborhood physically onto a dark-mode interactive **Folium Map of NYC**.

### 3. Raw Data Analysis
A fully isolated Tab exclusively for rigorous Exploratory Data Analysis of the NYC Taxi dataset before it hits a neural network.
*   **1D Distributions**: Select any numerical variable to instantly generate high-resolution Plotly Histograms visualizing its frequency distribution. 
*   **2D Correlations**: Explicitly choose an X-Axis feature, Y-Axis feature, and a Color Mapping variable to generate interactive Scatter Plots showing how any two physical attributes of a taxi ride correlate.

### 4. Standalone Serverless Export
Instantly package your trained neural network representation into a beautiful, offline **WebAssembly (`stlite`)** application.
*   **One-Click ZIP Generation**: Generates a mathematically compressed `.zip` artifact directly from the Streamlit sidebar containing your modeled `export_data.csv`, the Pyodide WebAssembly runtime wrappers, and your exact visualization logic.
*   **Zero-Backend Deployment**: WebAssembly runs locally entirely inside the browser. Extract the ZIP and double click `index.html` to run the Latent Dashboards completely offline, or drag the folder directly into Netlify to host it globally for free without ever paying for a backing Python server!

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

### Running the App

Once your environment is active and dependencies are installed, start the Streamlit server:

```bash
streamlit run app.py
```

## 🧠 Project Architecture

*   **`app.py`**: The main interface. Handles Streamlit layout state management, KNN nearest neighbor index extractions, Folium map rendering, Data Analysis charting, and PyTorch model orchestration.
*   **`data.py`**: Handles API fetching, TLC CSV loading, data cleaning, and complex preprocessing logic mapping PyTorch Tensors sequentially to Pandas Dataframes to ensure pixel-perfect tooltip generation.
*   **`model.py`**: Defines the `Autoencoder` via PyTorch Lightning. Includes dynamic component building and Denoising injection logic.
*   **`exporter.py`**: The Static WebAssembly App compiler. Dynamically strips PyTorch variables, unifies Latent embeddings, and constructs a completely pure Python/JS/HTML offline frontend package using `stlite` that circumvents local CORS restrictions natively.
*   **`viz.py`**: Wraps the configuration for `plotly.express.scatter` and complex transparent styling definitions.
*   **`get_centroids.py`**: Independent utility pulling actual geospatial centroid lat/lon coordinates natively from massive TLC Shapefiles (`.shp`).

## 🛠 Tech Stack

*   **Frontend / UI:** [Streamlit](https://streamlit.io/)
*   **Machine Learning / Deep Learning:** [PyTorch](https://pytorch.org/) & [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/) & [Scikit-Learn](https://scikit-learn.org/)
*   **Geospatial & Visualization:** [Plotly Express](https://plotly.com/python/plotly-express/), [Folium](https://python-visualization.github.io/folium/), & [GeoPandas](https://geopandas.org/)