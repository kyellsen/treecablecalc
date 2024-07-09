import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import List, Dict
from ..classes.measurement_version import MeasurementVersion


def plt_multi_measurement_versions(measurement_versions: List[MeasurementVersion],
                                   label_attributes: Dict[str, str] = None) -> plt.Figure:
    """
    Plots elongation versus force for a list of MeasurementVersion instances, each with a unique color.

    Args:
        measurement_versions (List[MeasurementVersion]): List of MeasurementVersion instances.
        label_attributes (Dict[str, str]): Dictionary describing which attributes to include in the label string and their display names.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Default label attributes
    default_label_attributes = {
        "system_name": "",
        "selection_mode": "",
        "selection_until": "",
        "expansion_insert_count": "exp_ins"
    }

    # Update the default attributes with the provided ones
    if label_attributes:
        default_label_attributes.update(label_attributes)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("deep", len(measurement_versions))

    labels = []

    for i, mv in enumerate(measurement_versions):
        data = mv.data

        label_parts = []
        for attr, display_name in default_label_attributes.items():
            value = getattr(mv, attr, None)
            if display_name:
                label_parts.append(f"{display_name}: {value}")
            else:
                label_parts.append(str(value))

        label_string = ", ".join(label_parts)
        labels.append((label_string, data['e'], data['f'], colors[i]))

    # Sort labels alphabetically
    labels.sort(key=lambda x: x[0])

    for label_string, e, f, color in labels:
        ax.plot(e, f, label=label_string, color=color, marker=None, linestyle='-')

    ax.set_title('Elongation over Force')
    ax.set_xlabel('Elongation [%]')
    ax.set_ylabel('Force [kN]')
    ax.legend()

    fig.tight_layout()
    return fig
