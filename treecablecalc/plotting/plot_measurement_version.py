import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plotly_filter_data(data: pd.DataFrame):
    # Erstellung eines Plotly-Plots mit zwei y-Achsen
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    marker_s = 2
    # Für den Weg Original [mm]
    fig.add_trace(
        go.Scatter(x=data.index, y=data['raw_x'], name='Way Original [mm]', mode='markers+lines',
                   marker=dict(color='black', size=marker_s),
                   line=dict(color='blue')))

    # Für den gefilterten Weg [mm]
    fig.add_trace(
        go.Scatter(x=data.index, y=data['x'], name='Way Filtered [mm]', mode='markers+lines',
                   marker=dict(color='black', size=marker_s),
                   line=dict(color='purple')))

    # Für die Kraft Original [kN] auf der rechten y-Achse
    fig.add_trace(
        go.Scatter(x=data.index, y=data['raw_f'], name='Force Original [kN]', mode='markers+lines',
                   marker=dict(color='black', size=marker_s),
                   line=dict(color='green')),
        secondary_y=True)

    # Für die gefilterte Kraft [kN] auf der rechten y-Achse
    fig.add_trace(
        go.Scatter(x=data.index, y=data['f'], name='Force Filtered [kN]', mode='markers+lines',
                   marker=dict(color='black', size=marker_s),
                   line=dict(color='red')),  #
        secondary_y=True)

    # Titel und Achsenbeschriftungen
    fig.update_layout(title='Way and Force over Time, Filtering', xaxis_title='Time')
    fig.update_yaxes(title_text='Way [mm]', secondary_y=False, color='black')
    fig.update_yaxes(title_text='Force [kN]', secondary_y=True, color='green')
    return fig


def plt_filter_data(data: pd.DataFrame):
    # Erstellung eines Plots
    fig, ax1 = plt.subplots()

    # Farben für die Linien definieren
    color_way = 'blue'
    color_force = 'green'
    color_way_filtered = 'purple'
    color_force_filtered = 'red'

    # Erste y-Achse für den Weg
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Way [mm]', color=color_way)
    ax1.plot(data.index, data['raw_x'], label='Way Original [mm]', color="black", marker='.', markersize=1,
             linestyle=None)
    ax1.plot(data.index, data['x'], label='Way Filtered [mm]', color=color_way_filtered, marker='.', markersize=1,
             linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color_way)

    # Zweite y-Achse für die Kraft
    ax2 = ax1.twinx()
    ax2.set_ylabel('Force [kN]', color=color_force)
    ax2.plot(data.index, data['raw_f'], label='Force Original [kN]', color="black", marker='.', markersize=1,
             linestyle=None)
    ax2.plot(data.index, data['f'], label='Force Filtered [kN]', color=color_force_filtered, marker='.', markersize=1,
             linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color_force)

    # Legende
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Titel
    plt.title('Way and Force over Time, Filtering')

    return fig


def plt_extrema(data: pd.DataFrame, peaks: pd.DatetimeIndex, valleys: pd.DatetimeIndex, first_drop: pd.DatetimeIndex):
    # Erstellung eines Plots
    fig, ax1 = plt.subplots()
    # Farben für die Linien definieren
    color_way = 'blue'
    color_force = 'green'

    # X-Achsenbeschriftung anpassen
    ax1.set_xlabel('DateTime')
    ax1.set_ylabel('Delta Distance [mm]', color=color_way)

    # Erste y-Achse für den Weg
    ax1.plot(data.index, data['d'], label='Delta Distance [mm]', color=color_way, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color_way)

    # Peaks und Valleys plotten
    ax1.plot(peaks, data.loc[peaks, 'd'], marker="v", label='Peaks', color='red', markersize=5, linestyle='')
    ax1.plot(valleys, data.loc[valleys, 'd'], marker="^", label='Valleys', color='orange', markersize=5, linestyle='')
    #
    # Zweite y-Achse für die Kraft
    ax2 = ax1.twinx()
    ax2.set_ylabel('Force [kN]', color=color_force)
    ax2.plot(data.index, data['f'], label='Force [kN]', color=color_force, linestyle='-')
    #
    # First-Drop plotten
    ax2.plot(first_drop, data['f'].loc[first_drop], marker="v", label='First-Drop', color='magenta', markersize=8, linestyle='')
    ax2.tick_params(axis='y', labelcolor=color_force)

    # Legende aktualisieren
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Titel
    plt.title('Delta-Distance and Force over Time - Extrema')

    return fig


def plotly_f_vs_e(data: pd.DataFrame, plot_raw: bool = False):
    # Erstellung eines Plotly-Plots
    fig = go.Figure()  # Korrektur hier: Hinzufügen der Klammern ()

    # Trace für die korrigierte Kraft
    fig.add_trace(
        go.Scatter(x=data['e'], y=data['f'], name='Force Correct [kN]', mode='lines+markers',
                   marker=dict(color='black', size=5),
                   line=dict(color='red')))

    if plot_raw:
        # Trace für die ursprüngliche Kraft
        fig.add_trace(
            go.Scatter(x=data['raw_e'], y=data['raw_f'], name='Force Original [kN]', mode='lines+markers',
                       marker=dict(color='black', size=5),
                       line=dict(color='blue')))

    # Titel und Achsenbeschriftungen
    fig.update_layout(title='Elongation over Force', xaxis_title='Elongation [%]', yaxis_title='Force [kN]')

    return fig


def plt_f_vs_e(data: pd.DataFrame, plot_raw: bool = False):
    # Optional: Setze Seaborn-Stil für verbesserte Ästhetik
    sns.set_style("whitegrid")

    # Erstelle eine Figure und eine Axes Instanz
    fig, ax = plt.subplots(figsize=(10, 6))  # Einstellen der Größe des Plots

    # Plot der korrigierten Kraft auf den erstellten Axes
    ax.plot(data['e'], data['f'], label='Force [kN]', marker='.', markersize=.5,
            markeredgecolor='black', color='red', linestyle='-')

    if plot_raw:
        # Plot der ursprünglichen Kraft, falls gewünscht, auf den erstellten Axes
        ax.plot(data['raw_e'], data['raw_f'], label='Force Raw [kN]', marker='.', markersize=0.5,
                markeredgecolor='grey', color='blue', linestyle='-')

    # Titel und Achsenbeschriftungen setzen
    ax.set_title('Elongation over Force')
    ax.set_xlabel('Elongation [%]')
    ax.set_ylabel('Force [kN]')
    ax.legend()  # Zeigt die Legende an

    fig.tight_layout()  # Verbessert die Anordnung der Plot-Elemente

    # Rückgabe der Figure, anstatt plt.show() aufzurufen
    return fig

def plt_select_data(data: pd.DataFrame, select_data: pd.DataFrame):
    y1 = "d"  # Delta Distance
    y2 = "f"  # Force

    # Erstellung eines Plots
    fig, ax1 = plt.subplots()

    # Farben für die Linien und Marker definieren
    color_d = 'blue'
    color_f = 'green'
    color_d_selected = 'cyan'  # Anpassung für bessere Sichtbarkeit der ausgewählten Delta Distance
    color_f_selected = 'lime'  # Anpassung für bessere Sichtbarkeit der ausgewählten Kraft

    # Erste y-Achse für Delta Distance
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Delta Distance [mm]', color=color_d)
    ax1.plot(data.index, data[y1], label='Delta Distance [mm]', color=color_d, linestyle='-')

    # Für ausgewählte Delta Distance Daten als Marker
    ax1.scatter(select_data.index, select_data[y1], label='Delta Distance Selected [mm]', color=color_d_selected,
                alpha=0.5, s=30, edgecolor='none')  # s ist die Größe der Marker

    ax1.tick_params(axis='y', labelcolor=color_d)

    # Zweite y-Achse für die Kraft
    ax2 = ax1.twinx()
    ax2.set_ylabel('Force [kN]', color=color_f)
    ax2.plot(data.index, data[y2], label='Force [kN]', color=color_f, linestyle='-')

    # Für ausgewählte Kraft Daten als Marker
    ax2.scatter(select_data.index, select_data[y2], label='Force Selected [kN]', color=color_f_selected,
                alpha=0.5, s=30, edgecolor='none')  # s ist die Größe der Marker

    ax2.tick_params(axis='y', labelcolor=color_f)

    # Legende zusammenführen und anzeigen
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    # Titel
    plt.title('Selection of Delta Distance + Force over Time')

    return fig

