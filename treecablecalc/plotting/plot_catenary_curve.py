import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from typing import Tuple, Optional


def plot_catenary_curve(curve: Tuple[np.ndarray, np.ndarray],
                        distance_horizontal: Optional[float],
                        rope_length: Optional[float],
                        slack_absolute: Optional[float],
                        sag_vertical: Optional[float],
                        force: Optional[float] = None,
                        e_percent_by_f: Optional[float] = None,
                        e_absolute_by_f: Optional[float] = None,
                        range_of_motion: Optional[float] = None,
                        system_name: Optional[str] = None,
                        system_version_name: Optional[str] = None):
    """
    Plots the catenary curve along with additional lines indicating the rope in a horizontally tensioned state,
    the vertical sag, the slack absolute, and the range of motion. Anchor points and respective labels are also plotted.

    Args:
        curve (Tuple[np.ndarray, np.ndarray]): x and y values of the catenary curve.
        distance_horizontal (float): Horizontal distance between anchor points.
        rope_length (float): Total length of the rope.
        slack_absolute (float): Absolute slack in the rope.
        sag_vertical (float): Vertical sag of the rope.
        force (Optional[float]): Force applied to the rope in kN.
        e_percent_by_f (Optional[float]): Elongation percentage by force.
        e_absolute_by_f (Optional[float]): Absolute elongation by force.
        range_of_motion (Optional[float]): Range of motion by force.
        system_name (Optional[str]): Name of the system.
        system_version_name
    """
    ROPE_TENSIONED_Y_POSITION = 0.25  # Design decision: Y-position of the tensioned line above the catenary curve
    TEXT_MARGIN = 0.2  # Margin for text labels

    x, y = curve

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))

    # Plot anchor points
    plt.scatter([0, distance_horizontal], [0, 0], color="black", zorder=5, label="Anchor Points")

    # Plot the catenary curve
    plt.plot(x, y, label="Cable curved", color="blue")

    # Plot the Cable tensioned line with T-ends
    plt.plot([0, rope_length], [ROPE_TENSIONED_Y_POSITION, ROPE_TENSIONED_Y_POSITION],
             color="blue", linestyle="--", label="Cable tensioned", marker='|', markersize=12,
             markevery=[0, 1])
    plt.text((distance_horizontal + slack_absolute) / 2, ROPE_TENSIONED_Y_POSITION - TEXT_MARGIN,
             f'{rope_length:.2f} m', verticalalignment='bottom', horizontalalignment='center',
             color='blue')

    # Plot the line representing slack_absolute with T-ends
    plt.plot([distance_horizontal, distance_horizontal + slack_absolute], [0, 0], color="purple",
             linestyle="-", label="Slack absolute", marker='|', markersize=12, markevery=[0, 1])
    plt.text(distance_horizontal + slack_absolute / 2, TEXT_MARGIN,
             f'{slack_absolute:.2f} m', verticalalignment='top', horizontalalignment='center',
             color='purple')

    # Plot the line representing absolute_by_f
    if e_absolute_by_f is not None:
        plt.plot([rope_length, rope_length + e_absolute_by_f], [0, 0],
                 color="orange",
                 linestyle="-", label="Elongation absolute", marker='|', markersize=12, markevery=[0, 1])
        plt.text(rope_length + e_absolute_by_f / 2, -TEXT_MARGIN,
                 f'{e_absolute_by_f:.2f} m', verticalalignment='bottom', horizontalalignment='center',
                 color='orange')

    # Plot the line representing range_of_motion
    if range_of_motion is not None:
        plt.plot([distance_horizontal, distance_horizontal + range_of_motion],
                 [-ROPE_TENSIONED_Y_POSITION, -ROPE_TENSIONED_Y_POSITION],
                 color="red", linestyle="-", label="Range of Motion", marker='|', markersize=12, markevery=[0, 1])
        plt.text(distance_horizontal + range_of_motion / 2, -ROPE_TENSIONED_Y_POSITION - TEXT_MARGIN,
                 f'{range_of_motion:.2f} m', verticalalignment='bottom', horizontalalignment='center',
                 color='red')

    if sag_vertical is not None:
        # Plot the vertical sag line with T-ends
        plt.plot([distance_horizontal / 2, distance_horizontal / 2], [0, -sag_vertical], color="green",
                 linestyle="--", label="Sag absolute", marker='_', markersize=12, markevery=[0, 1])
        plt.text(distance_horizontal / 2 + TEXT_MARGIN, -sag_vertical / 2, f'{sag_vertical:.2f} m',
                 verticalalignment='bottom', horizontalalignment='left', color='green')

    # Add titles and labels
    title_string = "Cable Catenary Curve"
    if system_name is not None:
        title_string += f" {system_name}"
    if system_version_name is not None:
        title_string += f" {system_version_name}"

    plt.title(title_string)
    plt.xlabel("Horizontal Distance [m]")
    plt.ylabel("Vertical Distance [m]")

    # Set y-axis limits to ensure all values from 1 to -2 are visible
    plt.ylim(-1.5, 0.5)

    # Place the legend below the plot
    plt.legend(loc='upper right', bbox_to_anchor=(1, -0.2), ncol=3)

    # Create additional text for force and percent_by_f
    additional_text = ""
    additional_text += f'Distance hor.: {distance_horizontal:.2f} m\n'
    additional_text += f'Rope length: {rope_length:.2f} m\n'
    additional_text += f'Slack abs.: {slack_absolute:.2f} m\n'
    additional_text += f'Sag abs. : {sag_vertical:.2f} m\n'

    if force is not None:
        additional_text += f'Force: {force:.2f} kN\n'
    if e_percent_by_f is not None:
        additional_text += f'Elongation: {e_percent_by_f:.2f} %\n'

    if e_absolute_by_f is not None:
        additional_text += f'Elongation abs.: {e_absolute_by_f:.2f} m\n'

    if range_of_motion is not None:
        additional_text += f'Range of Motion: {range_of_motion:.2f} m'

    # Add the additional text in a box below the legend
    if additional_text:
        plt.text(0, -1.75, additional_text, verticalalignment='top',
                 horizontalalignment='left', bbox=dict(facecolor='white', alpha=1))
    fig.tight_layout()
    return fig
