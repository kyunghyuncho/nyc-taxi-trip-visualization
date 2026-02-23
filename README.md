# 🚖 NYC Taxi Dimensionality Reduction Demo

This is an interactive Streamlit application demonstrating nonlinear dimensionality reduction. The app fetches a subset of the NYC Yellow Taxi dataset, allows users to configure and train a PyTorch Lightning Autoencoder with a 2D bottleneck, and visualizes the resulting latent space using Plotly.

## 🌟 Features

*   **Interactive Data Fetching:** Pulls real-world data from the NYC Open Data API (Socrata).
*   **Dynamic Data Preprocessing:** Cleans and processes the data, including One-Hot Encoding for categorical variables and `StandardScaler` for numerical ones.
*   **Configurable Autoencoder:** Train a PyTorch Lightning Autoencoder dynamically from the UI. Customize:
    *   Input Features
    *   Hidden Layer dimensions
    *   Learning Rate and Batch Size
    *   Optimizer (`Adam`, `SGD`, or a custom `Muon` implementation)
    *   Nonlinearity (`ReLU`, `Tanh`, `GELU`)
*   **Real-time Visualization:** Uses Plotly Express to render the 2D bottleneck of the trained autoencoder.
*   **Dynamic Color Mapping:** Change the color mapping of the scatter plot on the fly without needing to retrain the model.

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

*   **`app.py`**: The main entry point. Handles the Streamlit UI layout, state management (`st.session_state`), and the training execution logic.
*   **`data.py`**: Handles pulling data from the Socrata API, cleaning invalid taxi trips, and applying `sklearn` transformers (OHE, StandardScaler).
*   **`model.py`**: Defines the `Autoencoder` via PyTorch Lightning. Includes dynamic architecture building and a custom Streamlit callback (`StreamlitProgressCallback`) to update the UI during training.
*   **`optimizer.py`**: Contains a custom PyTorch implementation of the `Muon` (MomentUm Orthogonalized by Newton-schulz) optimizer, intended as an alternative to Adam/SGD for internal layers.
*   **`viz.py`**: Wraps the configuration for `plotly.express.scatter` to visualize the extracted 2D embeddings.

## 🛠 Tech Stack

*   **Frontend / UI:** [Streamlit](https://streamlit.io/)
*   **Machine Learning / Deep Learning:** [PyTorch](https://pytorch.org/) & [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/) & [Scikit-Learn](https://scikit-learn.org/)
*   **Visualization:** [Plotly Express](https://plotly.com/python/plotly-express/)