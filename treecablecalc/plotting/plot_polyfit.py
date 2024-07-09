import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..classes.cable_model import CableModel


def plt_polyfit(elongation: pd.Series, force: pd.Series, cable_model) -> plt.Figure:
    """
    Plots data with a polynomial fit based on a np.poly1d model.

    Args:
        elongation (pd.Series): The x-values of the data.
        force (pd.Series): The y-values of the data.
        cable_model (CableModel): The fitted model containing np.poly1d object and quality.

    Returns:
        plt.Figure: The figure object containing the plot.
    """
    # Prepare plot data
    force_poly = np.linspace(cable_model.lower_bound, cable_model.upper_bound, 1000)
    elongation_poly = cable_model.model(force_poly)  # Use np.poly1d object directly to get y-values

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(elongation, force, label='Original Data', color="blue", s=3)
    ax.plot(elongation_poly, force_poly, color='red', label=f'Polynomial Fit Order: {cable_model.model.order}')
    ax.set_title('Elongation over Force with Polynomial Model')
    ax.set_xlabel('Elongation [%]')
    ax.set_ylabel('Force [kN]')
    ax.legend()

    # Place a text block
    textstr = f'Polynomial Order: {cable_model.model.order}\nQuality (R²): {cable_model.quality_r2:.4f}'
    ax.text(0.80, 0.2, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    fig.tight_layout()

    return fig
