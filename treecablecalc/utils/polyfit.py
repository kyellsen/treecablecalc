import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from kj_core.utils.runtime_manager import dec_runtime

from ..classes.cable_model import CableModel


def polyfit_with_np(x: pd.Series, y: pd.Series, degree_min: int = 1, degree_max: int = 10,
                    desired_quality: float = 0.9999):
    best_quality_r2 = -np.inf
    best_degree = None
    best_coeffs = None

    for degree in range(degree_min, degree_max + 1):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        current_quality = r2_score(y, y_pred)

        if current_quality > best_quality_r2:
            best_quality_r2 = current_quality
            best_degree = degree
            best_coeffs = coeffs

            if current_quality >= desired_quality:
                break

    if best_coeffs is None:
        raise ValueError("No model found that meets the desired quality.")

    # Erstellen Sie ein Poly1d-Objekt für das beste Modell
    best_model = np.poly1d(best_coeffs)

    lower_bound, upper_bound = x.min(), x.max()

    return CableModel(model=best_model, degree=best_degree, quality=best_quality_r2,
                      lower_bound=lower_bound, upper_bound=upper_bound)


def add_zeros(x: pd.Series, n: int) -> pd.Series:
    """
    Adds a specified number of zeros to the beginning of a pandas Series.

    Args:
        x (pd.Series): Input series.
        n (int): Number of zeros to add.

    Returns:
        pd.Series: Modified series with zeros prepended.

    Raises:
        ValueError: If n is not greater than 0.
    """
    if n <= 0:
        raise ValueError("Parameter n must be > 0")
    x_zeros = pd.Series(np.zeros(n))
    return pd.concat([x_zeros, x], ignore_index=True)


def add_min_values(x: pd.Series, n: int) -> pd.Series:
    """
    Adds a specified number of minimal values of the series to the beginning of a pandas Series.

    Args:
        x (pd.Series): Input series.
        n (int): Number of minimal values to add.

    Returns:
        pd.Series: Modified series with minimal values prepended.

    Raises:
        ValueError: If n is not greater than 0.
    """
    if n <= 0:
        raise ValueError("Parameter n must be > 0")
    min_value = x.min()
    min_values = pd.Series([min_value] * n)
    return pd.concat([min_values, x], ignore_index=True)
