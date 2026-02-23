import plotly.express as px
import pandas as pd
import streamlit as st

def plot_embeddings(df: pd.DataFrame, embeddings: pd.DataFrame, color_column: str, hover_cols: list) -> st.plotly_chart:
    """
    Plots the 2D bottleneck representations using Plotly Express.
    Embeddings MUST be of shape (N, 2).
    """
    # Create an intermediate dataframe for plotting that contains both the 2D coords and the relevant raw data back
    plot_df = pd.DataFrame({
        'Dim 1': embeddings[:, 0],
        'Dim 2': embeddings[:, 1],
    })
    
    # Merge the required original columns into our plot_df
    # Note: color_column and hover_cols might overlap
    cols_to_add = set([color_column] + hover_cols)
    for col in cols_to_add:
        # Avoid key errors if somehow the column got removed
        if col in df.columns:
            plot_df[col] = df[col].values

    # Plot
    fig = px.scatter(
        plot_df,
        x='Dim 1',
        y='Dim 2',
        color=color_column if color_column in plot_df.columns else None,
        hover_data=hover_cols,
        title="Autoencoder 2D Bottleneck Potential Clusters",
        template="plotly_dark", # Looks better in typical ML demos, change if needed
        opacity=0.7,
        render_mode='webgl' # Essential for 5000+ points
    )
    
    # Adjust axes
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
         margin=dict(l=20, r=20, t=50, b=20),
         coloraxis_colorbar=dict(title=color_column)
    )

    return fig
