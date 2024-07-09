import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_cable_models(cable_models: List[Tuple['CableModel', 'SystemVersion']], max_degree: int,
                      force_max: float = None) -> plt.Figure:
    """
    Plot the avg_cable_model polynomials of multiple CableModel instances.

    :param cable_models: List of tuples containing CableModel instances and corresponding SystemVersion instances
    :param max_degree: The maximum polynomial degree to extend to
    :param force_max: The maximum force value for plotting
    :return: The matplotlib Figure object
    :raises RuntimeError: If an error occurs during plotting
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for cm, sv in cable_models:
        try:
            extended_poly = cm.extend_to_degree(max_degree)
            label = f"{sv.system.system}, {sv.system_version_name}"
            force_range = np.linspace(cm.lower_bound, cm.upper_bound, 100)
            elongation = extended_poly(force_range)
            if force_max is not None:
                mask = force_range <= force_max
                force_range = force_range[mask]
                elongation = elongation[mask]
            ax.plot(elongation, force_range, label=label)
        except Exception as e:
            raise RuntimeError(f"Error plotting polynomial for {sv.system.system}, {sv.system_version_name}: {e}")

    ax.set_xlabel('Elongation [%]')
    ax.set_ylabel('Force [kN]')
    ax.set_title('Comparison of avg_cable_model Polynomials')
    if force_max is not None:
        ax.set_ylim([0, force_max])
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2)
    ax.grid(True)
    fig.tight_layout()

    return fig


def plot_cable_models_difference(system_version_1: 'SystemVersion', system_version_2: 'SystemVersion', max_degree: int,
                                 force_max: float = None, skale_ax2_ref: int = 400) -> plt.Figure:
    """
    Plot the differences between avg_cable_model polynomials of two CableModel instances, and their individual polynomials.

    :param system_version_1: First SystemVersion instance
    :param system_version_2: Second SystemVersion instance
    :param max_degree: The maximum polynomial degree to extend to
    :param force_max: The maximum force value for plotting
    :return: The matplotlib Figure object
    :raises RuntimeError: If an error occurs during plotting
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sv1 = system_version_1
    cm1 = sv1.cable_model
    sv2 = system_version_2
    cm2 = sv2.cable_model

    if force_max is None:
        force_max = max(cm1.upper_bound, cm2.upper_bound)

    # Extend polynomials only once
    extended_poly1 = cm1.extend_to_degree(max_degree)
    extended_poly2 = cm2.extend_to_degree(max_degree)

    # Define the force ranges considering the provided force_max
    force_range1 = np.linspace(cm1.lower_bound, cm1.upper_bound, 100)
    force_range2 = np.linspace(cm2.lower_bound, cm2.upper_bound, 100)
    common_force_range = np.linspace(max(cm1.lower_bound, cm2.lower_bound), min(cm1.upper_bound, cm2.upper_bound), 100)

    # Plot the individual polynomials
    try:
        elongation1 = extended_poly1(force_range1)
        elongation2 = extended_poly2(force_range2)

        if force_max is not None:
            mask1 = force_range1 <= force_max
            force_range1 = force_range1[mask1]
            elongation1 = elongation1[mask1]

            mask2 = force_range2 <= force_max
            force_range2 = force_range2[mask2]
            elongation2 = elongation2[mask2]

            mask_common = common_force_range <= force_max
            common_force_range = common_force_range[mask_common]

        ax.plot(elongation1, force_range1, label=f"{sv1.system.system}, {sv1.system_version_name}", color="red")
        ax.plot(elongation2, force_range2, label=f"{sv2.system.system}, {sv2.system_version_name}", color="green")

        common_elongation1 = extended_poly1(common_force_range)
        common_elongation2 = extended_poly2(common_force_range)

        ax.fill_betweenx(common_force_range, common_elongation1, common_elongation2,
                         where=(common_elongation1 >= common_elongation2), interpolate=True, color='gray', alpha=0.5)
        ax.fill_betweenx(common_force_range, common_elongation2, common_elongation1,
                         where=(common_elongation2 >= common_elongation1), interpolate=True, color='gray', alpha=0.5)

        diff_poly = extended_poly1 - extended_poly2
        elongation_diff = abs(diff_poly(common_force_range))
        ax.plot(elongation_diff, common_force_range, label=f"Difference", linestyle='--', color="black")
    except Exception as e:
        raise RuntimeError(
            f"Error plotting polynomial difference for SystemVersion IDs {sv1.system_version_id} and {sv2.system_version_id}: {e}")

    skale_ax2 = skale_ax2_ref / 100
    # Zweite X-Achse
    ax2 = ax.twiny()
    # Berechne den neuen Bereich der X-Achse ax2
    xmin, xmax = ax.get_xlim()
    new_xmin, new_xmax = xmin * skale_ax2, xmax * skale_ax2
    # Skaliere die X-Achse ax2 auf den neuen Bereich
    ax2.set_xlim(new_xmin, new_xmax)

    ax.set_xlabel('Elongation [%]')
    ax2.set_xlabel('Elongation [cm] Sample 400 cm')
    ax.set_ylabel('Force [kN]')
    ax.set_title(f'Difference between {sv1.system.system} and {sv2.system.system}')
    ax.legend()
    ax.grid(True)  # Enable grid only for primary axes
    ax2.grid(False)  # Disable grid for secondary axes
    if force_max is not None:
        ax.set_ylim([0, force_max])  # Set the y-axis limit to the provided force_max
    fig.tight_layout()

    return fig
