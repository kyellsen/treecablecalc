import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List

from ..classes.cable_model import CableModel


def plt_average_poly1d(system_name, cable_models: List[CableModel], avg_cable_model: CableModel) -> plt.Figure:
    x = np.linspace(avg_cable_model.lower_bound, avg_cable_model.upper_bound, 100)

    fig, ax = plt.subplots()
    for cm in cable_models:
        y = cm.model(x)
        ax.plot(x, y, color='blue', alpha=0.5)

    y_avg = avg_cable_model.model(x)
    ax.plot(x, y_avg, color='red', linewidth=2)

    ax.set_xlabel('Force [kN]')
    ax.set_ylabel('Elongation [%]')
    ax.set_title(f'Polynomials and their average, {system_name}')

    return fig
