import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots


from ..classes.cable_model import CableModel

def plt_polyfit(x: pd.Series, y: pd.Series, cable_model) -> plt.Figure:
    """
    Plots data with a polynomial fit based on a np.poly1d model.

    Args:
        x (pd.Series): The x-values of the data.
        y (pd.Series): The y-values of the data.
        cable_model (CableModel): The fitted model containing np.poly1d object and quality.

    Returns:
        plt.Figure: The figure object containing the plot.
    """
    # Prepare plot data
    x_plot = np.linspace(cable_model.lower_bound, cable_model.upper_bound, 1000)
    y_plot = cable_model.model(x_plot)  # Use np.poly1d object directly to get y-values

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, label='Original Data', color="blue", s=3)
    ax.plot(x_plot, y_plot, color='red', label=f'Polynomial Fit Order: {cable_model.model.order}')
    ax.set_title('Elongation over Force with Polynomial Model')
    ax.set_xlabel('Force')
    ax.set_ylabel('Elongation')
    ax.legend()

    # Place a text block
    textstr = f'Polynomial Order: {cable_model.model.order}\nQuality (R²): {cable_model.quality_r2:.4f}'
    ax.text(0.80, 0.2, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    fig.tight_layout()

    return fig

