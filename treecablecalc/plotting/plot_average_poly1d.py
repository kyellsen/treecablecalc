import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List

from ..classes.cable_model import CableModel


def plt_average_poly1d(system_name: str, cable_models: List['CableModel'], avg_cable_model: 'CableModel') -> plt.Figure:
    """
    Plot the polynomials of multiple CableModel instances and their average.

    :param system_name: Name of the system
    :param cable_models: List of CableModel instances
    :param avg_cable_model: The average CableModel instance
    :return: The matplotlib Figure object
    """
    force_range = np.linspace(avg_cable_model.lower_bound, avg_cable_model.upper_bound, 100)

    fig, ax = plt.subplots()
    for cm in cable_models:
        try:
            elongation = cm.model(force_range)
            ax.plot(elongation, force_range, color='blue', alpha=0.5)
        except Exception as e:
            raise RuntimeError(f"Error plotting polynomial for CableModel ID {cm.cable_model_id}: {e}")

    try:
        elongation_avg = avg_cable_model.model(force_range)
        ax.plot(elongation_avg, force_range, color='red', linewidth=2)
    except Exception as e:
        raise RuntimeError(f"Error plotting average polynomial: {e}")

    ax.set_xlabel('Elongation [%]')
    ax.set_ylabel('Force [kN]')
    ax.set_title(f'Polynomials and their average, {system_name}')
    ax.grid(True)

    return fig
