import plotly.express as px
import pandas as pd
import streamlit as st

def plot_embeddings(df: pd.DataFrame, embeddings: pd.DataFrame, color_column: str, hover_cols: list, highlight_categories: list = None):
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

    color_col_to_use = color_column if color_column in plot_df.columns else None
    color_discrete_map = None
    
    # Handle highlighting specific categories
    if highlight_categories is not None and color_col_to_use is not None:
        plot_df['_highlight'] = plot_df[color_column].apply(lambda x: x if x in highlight_categories else 'Other')
        color_col_to_use = '_highlight'
        
        # Ensure 'Other' is explicitly mapped to a gray, translucent color
        colors = px.colors.qualitative.Plotly
        color_discrete_map = {cat: colors[i % len(colors)] for i, cat in enumerate(highlight_categories)}
        color_discrete_map['Other'] = 'rgba(128, 128, 128, 0.1)'
        
        # Sort so 'Other' is drawn first (in background)
        plot_df['is_other'] = plot_df['_highlight'] == 'Other'
        plot_df = plot_df.sort_values('is_other', ascending=False).drop(columns=['is_other'])

    # Plot
    fig = px.scatter(
        plot_df,
        x='Dim 1',
        y='Dim 2',
        color=color_col_to_use,
        color_discrete_map=color_discrete_map,
        hover_data=hover_cols,
        title="Autoencoder 2D Bottleneck Potential Clusters",
        template="plotly_dark", # Looks better in typical ML demos, change if needed
        opacity=0.7 if highlight_categories is None else None, # Let individual colors dictate opacity if highlighting
        render_mode='webgl' # Essential for 5000+ points
    )
    
    # Adjust axes
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
         margin=dict(l=20, r=20, t=50, b=20),
         coloraxis_colorbar=dict(title=color_column)
    )

    return fig
