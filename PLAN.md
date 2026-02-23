# Project Plan: Interactive Autoencoder Dimensionality Reduction Demo

## 1. Project Overview
Build an interactive Streamlit application demonstrating nonlinear dimensionality reduction. The app fetches a subset of the NYC Yellow Taxi dataset, allows users to configure and train a PyTorch Lightning Autoencoder with a 2D bottleneck, and visualizes the resulting latent space using Plotly.

## 2. Architecture & Tech Stack
* **Web Framework:** Streamlit (UI layout, interactivity, and state management).
* **Machine Learning:** PyTorch and PyTorch Lightning (Model definition and training).
* **Data Processing:** Pandas, Scikit-Learn (Data fetching, scaling, encoding).
* **Visualization:** Plotly Express (Interactive scatter plots with hover and zoom).

---

## 3. Implementation Phases

### Phase 1: Data Pipeline (`data.py`)
* **Task 1.1: Fetch Data:** Query the NYC Open Data API (Socrata) for a manageable sample (e.g., 5,000 to 10,000 rows). Wrap this function with `@st.cache_data` to prevent re-downloading on every UI interaction.
* **Task 1.2: Preprocessing:** Clean invalid rows (e.g., negative distances/fares). Extract temporal features (hour, day) from datetime columns.
* **Task 1.3: Feature Transformation:** Implement a pipeline taking user-selected features, applying one-hot encoding to categorical data, and applying `StandardScaler` to numerical data.

### Phase 2: Streamlit UI Layout (`app.py`)
* **Task 2.1: Sidebar Setup:** Create `st.multiselect` for input features and `st.selectbox` for the color-mapping feature.
* **Task 2.2: Hyperparameters UI:** Implement the following controls in the sidebar:

| Hyperparameter | Streamlit Component | Options / Default |
| :--- | :--- | :--- |
| **Learning Rate** | `st.number_input` | Default: 0.001, Format: scientific |
| **Optimizer** | `st.selectbox` | Default: 'Adam', Options: 'Adam', 'Muon', 'SGD' |
| **Nonlinearity** | `st.selectbox` | Default: 'ReLU', Options: 'ReLU', 'Tanh', 'GELU' |
| **Epochs** | `st.slider` | Range: 10 to 100 |
| **Batch Size** | `st.selectbox` | Options: 16, 32, 64, 128 |
| **Hidden Layers** | `st.text_input` | Default: "64, 32" |

* **Task 2.3: Main Display:** Place `st.button('Train Autoencoder')` and allocate placeholders for the dataset preview, training progress bar (`st.progress`), and Plotly chart (`st.plotly_chart`).

### Phase 3: PyTorch Lightning Model (`model.py`)

* **Task 3.1: LightningModule Definition:** Create an `Autoencoder(pl.LightningModule)` class. Dynamically parse the hidden layers string to build `nn.Sequential` blocks. Ensure the encoder output strictly equals 2 dimensions.
* **Task 3.2: Nonlinearity & Optimizer:** Inject the user-selected Nonlinearity between linear layers. In `configure_optimizers`, use an `if/elif` block to initialize `torch.optim.Adam`, the Muon optimizer, or `torch.optim.SGD` using the user's Learning Rate.
* **Task 3.3: Streamlit Progress Callback:** Create a custom `pl.Callback`. Override `on_train_epoch_end` to update a Streamlit progress bar and display the current MSE loss in the UI.
* **Task 3.4: Inference:** Add a method to extract the 2D embeddings (pass the dataset through the encoder only) after training completes.

### Phase 4: Visualization & State Management (`viz.py` & `app.py`)

* **Task 4.1: Plotly Scatter:** Create a function using `plotly.express.scatter`. Map the extracted 2D embeddings to the X and Y axes.
* **Task 4.2: Hover Data:** Pass the original (unscaled) pandas dataframe to the `hover_data` argument so tooltips show interpretable, real-world values.
* **Task 4.3: Session State (`st.session_state`):** Store trained 2D embeddings and the trained model in `st.session_state`.
* **Task 4.4: Execution Logic:** If the user clicks "Train", run Phase 3 and save embeddings to state. If the user only changes the Color Mapping dropdown, do not retrain; pull embeddings from `st.session_state` and update the plot.