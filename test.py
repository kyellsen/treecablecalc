from pathlib import Path
import numpy as np
import pandas as pd
import sys
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import seaborn as sns

from kj_logger import get_logger

import treecablecalc as tcc
from treecablecalc.classes.measurement import Measurement
from treecablecalc.classes.measurement_version import MeasurementVersion
from treecablecalc.classes.series import Series
from treecablecalc.classes.system import System
from treecablecalc.classes.data_tcc import DataTCC
from treecablecalc.classes.cable_calculator import CableCalculator

# Beispiel für die Verwendung
if __name__ == "__main__":
    # Main
    main_path = Path(r"C:\kyellsen\005_Projekte\2023_TreeCableCalc")
    analyse_name = r"2023_Kronensicherung_Stuttgart_2024-06-15"
    data_path = main_path / "021_Daten_Clean"  # Für alle Daten-Importe des Projektes gemeinsam
    working_directory = main_path / "030_Analysen" / analyse_name / "working_directory"  # Für alle Daten-Exporte des Projektes gemeinsam
    source_db_name = "TreeCableCalc_Stuttgart_2024-05-29.db"
    source_db = data_path / source_db_name

    CONFIG, LOG_MANAGER, DATA_MANAGER, DATABASE_MANAGER, PLOT_MANAGER = tcc.setup(
        working_directory=str(working_directory), log_level="info")
    DATABASE_MANAGER.duplicate(database_path=str(source_db))
    DATABASE_MANAGER.connect(db_name=str(source_db_name))
    measurement_list: List[Measurement] = DATABASE_MANAGER.load(class_name=Measurement)
    measurement_list = [m for m in measurement_list if
                        m.execute != "dont_use" and m.series.description == "2023-01-10_HAWK"]
    measurement_list = measurement_list[0:15]

    plot_flag = True

    # for m in measurement_list:
    #     m.load_with_features(selection_mode="inc_preload", selection_until="first_drop",
    #                          measurement_version_name="version_tcc", update_existing=False,
    #                          filter_data=True, null_offset_f=True, fit_model=True, plot_filter=plot_flag, plot_extrema=plot_flag,
    #                          plot_selection=plot_flag, plot_f_vs_e=plot_flag, plot_fit_model=plot_flag)
    #
    # df_list = []
    # for m in measurement_list:
    #     df = m.get_param_df(use_interp1d=False, use_poly1d=True,
    #                         measurement_version_names=["version_tcc"])  # None == ALL
    #     df_list.append(df)
    #
    # df = pd.concat(df_list, ignore_index=True)

    system_list: List[System] = DATABASE_MANAGER.load(class_name=System, ids=[3, 4])

    filter_query = "series_id == 2 and execute != 'dont_select' and expansion_insert_count == 0"
    #
    sys_v_list = []
    for sys in system_list:
        new_sys_v = sys.create_system_version_plus_measurement_versions(
            version_name="inc_preload_until_first_drop_without_exp",
            filter_query=filter_query,
            auto_commit=True,
            selection_mode="inc_preload",
            selection_until="first_drop",
            update_existing=False,
            filter_data=True,
            null_offset_f=True,
            fit_model=True,
            plot_filter=plot_flag,
            plot_extrema=plot_flag,
            plot_selection=plot_flag,
            plot_f_vs_e=plot_flag,
            plot_fit_model=plot_flag
        )
        if new_sys_v:
            sys_v_list.append(new_sys_v)

    for sys in system_list:
        new_sys_v = sys.create_system_version_plus_measurement_versions(
            version_name="exc_preload_until_first_drop_without_exp",
            filter_query=filter_query,
            auto_commit=True,
            selection_mode="exc_preload",
            selection_until="first_drop",
            update_existing=False,
            filter_data=True,
            null_offset_f=True,
            fit_model=True,
            plot_filter=plot_flag,
            plot_extrema=plot_flag,
            plot_selection=plot_flag,
            plot_f_vs_e=plot_flag,
            plot_fit_model=plot_flag
        )
        if new_sys_v:
            sys_v_list.append(new_sys_v)

    filter_query = "series_id == 2 and execute != 'dont_select' and expansion_insert_count == 2"
    # Wenn mehrere Bedingungen vorhanden sind, können sie mit AND/OR kombiniert werden

    for sys in system_list:
        new_sys_v = sys.create_system_version_plus_measurement_versions(
            version_name="inc_preload_until_first_drop_with_exp",
            filter_query=filter_query,
            auto_commit=True,
            selection_mode="inc_preload",
            selection_until="first_drop",
            update_existing=False,
            filter_data=True,
            null_offset_f=True,
            fit_model=True,
            plot_filter=plot_flag,
            plot_extrema=plot_flag,
            plot_selection=plot_flag,
            plot_f_vs_e=plot_flag,
            plot_fit_model=plot_flag
        )
        if new_sys_v:
            sys_v_list.append(new_sys_v)

    for sys in system_list:
        new_sys_v = sys.create_system_version_plus_measurement_versions(
            version_name="exc_preload_until_first_drop_with_exp",
            filter_query=filter_query,
            auto_commit=True,
            selection_mode="exc_preload",
            selection_until="first_drop",
            update_existing=False,
            filter_data=True,
            null_offset_f=True,
            fit_model=True,
            plot_filter=plot_flag,
            plot_extrema=plot_flag,
            plot_selection=plot_flag,
            plot_f_vs_e=plot_flag,
            plot_fit_model=plot_flag
        )
        if new_sys_v:
            sys_v_list.append(new_sys_v)

    for sys in sys_v_list:
        sys.calc_params_from_measurement_versions("poly1d")

    cc = CableCalculator.create_with_system_version(
        ks_type='example_type',
        stem_diameter_1=50,
        stem_diameter_2=60,
        stem_damaged='no',
        system_identifier="cobra 4t statisch",
        distance_horizontal=4.5,
        rope_length=5,
        force=5)

    # Plotten Sie die Kettenkurve
    ccv_list = cc.cable_calculator_version

    for ccv in ccv_list:
        ccv.plot_catenary_curve()

    test_df_2 = cc.calculate_values_for_all_versions(force=10)
    # for ccv in cc.cable_calculator_version:
    #     ccv.calc_range_of_motion(force=10)
    #ROM = cc4.calc_for_all(10)
    #print(ROM)

    from treecablecalc.plotting.plot_multi_measurement_versions import plt_multi_measurement_versions_data
    #
    # filtered_sys_versions = [sv for sv in sys_v_list if sv.system_id in [3, 4]]
    #
    # # Extract all MeasurementVersion instances from the filtered SystemVersion instances
    # mv_list_selected = []
    # for sv in filtered_sys_versions:
    #     mv_list_selected.extend(sv.measurement_version)
    #
    # if mv_list_selected:
    #     plt_multi_measurement_versions_data(mv_list_selected)

    # def filter_and_plot_by_system_ids(system_versions: List, system_ids: List[int],
    #                                   label_attributes: Dict[str, str] = None):
    #     """
    #     Filters SystemVersion instances by a list of system_id, extracts their MeasurementVersion instances,
    #     and plots the data.
    #
    #     Args:
    #         system_versions (List[SystemVersion]): List of SystemVersion instances.
    #         system_ids (List[int]): List of system_id values to filter by.
    #         label_attributes (Dict[str, str]): Dictionary describing which attributes to include in the label string and their display names.
    #     """
    #     # Filter SystemVersion instances by the given system_ids
    #     filtered_sys_versions = [sv for sv in system_versions if sv.system_id in system_ids]
    #
    #     # Extract all MeasurementVersion instances from the filtered SystemVersion instances
    #     mv_list_selected = []
    #     for sv in filtered_sys_versions:
    #         mv_list_selected.extend(sv.measurement_version)
    #
    #     # Plot the data if there are any MeasurementVersion instances
    #     if mv_list_selected:
    #         plt_multi_measurement_versions_data(mv_list_selected, label_attributes)
    #     else:
    #         print(f"No MeasurementVersion instances found for system_ids {system_ids}")
    #
    #
    # # Example usage with a list of SystemVersion instances and system_id groups
    # system_id_groups = [[1, 2], [3, 4], [5, 6], [8, 9], [10, 11], [12, 13]]
    # label_attributes = {
    #     "system_name": "",
    #     "selection_mode": "",
    #     "selection_until": "",
    #     "expansion_insert_count": "Exp. Ins."
    # }
    #
    # # Iterate over each group of system_ids and call the filter_and_plot_by_system_ids function
    # for system_ids in system_id_groups:
    #     filter_and_plot_by_system_ids(sys_v_list, system_ids, label_attributes)
