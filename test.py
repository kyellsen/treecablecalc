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
from treecablecalc.classes.system_version import SystemVersion
from treecablecalc.classes.data_tcc import DataTCC
from treecablecalc.classes.cable_calculator import CableCalculator
from treecablecalc.classes.cable_calculator_version import CableCalculatorVersion

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
        working_directory=str(working_directory), log_level="debug")
    DATABASE_MANAGER.duplicate(database_path=str(source_db))
    DATABASE_MANAGER.connect(db_name=str(source_db_name))
    measurement_list: List[Measurement] = DATABASE_MANAGER.load(class_name=Measurement)
    measurement_list = [m for m in measurement_list if
                        m.execute != "dont_use" and m.series.description == "2023-01-10_HAWK"]
    measurement_list = measurement_list[0:15]

    plot_flag = False
    updated_existing_flag = False

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

    system_list: List[System] = DATABASE_MANAGER.load(class_name=System)  # specific ids as List

    filter_query = "series_id == 2 and execute != 'dont_select'"  # and expansion_insert_count == 0"
    #
    sys_v_list = []
    for sys in system_list:
        new_sys_v = sys.create_system_version_plus_measurement_versions(
            version_name="inc_preload_until_first_drop_without_exp",
            filter_query=filter_query,
            auto_commit=True,
            selection_mode="inc_preload",
            selection_until="first_drop",
            update_existing=updated_existing_flag,
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
            update_existing=updated_existing_flag,
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

    # filter_query = "series_id == 2 and execute != 'dont_select' and expansion_insert_count == 2"
    # # Wenn mehrere Bedingungen vorhanden sind, können sie mit AND/OR kombiniert werden
    #
    # for sys in system_list:
    #     new_sys_v = sys.create_system_version_plus_measurement_versions(
    #         version_name="inc_preload_until_first_drop_with_exp",
    #         filter_query=filter_query,
    #         auto_commit=True,
    #         selection_mode="inc_preload",
    #         selection_until="first_drop",
    #         update_existing=updated_existing_flag,
    #         filter_data=True,
    #         null_offset_f=True,
    #         fit_model=True,
    #         plot_filter=plot_flag,
    #         plot_extrema=plot_flag,
    #         plot_selection=plot_flag,
    #         plot_f_vs_e=plot_flag,
    #         plot_fit_model=plot_flag
    #     )
    #     if new_sys_v:
    #         sys_v_list.append(new_sys_v)
    #
    # for sys in system_list:
    #     new_sys_v = sys.create_system_version_plus_measurement_versions(
    #         version_name="exc_preload_until_first_drop_with_exp",
    #         filter_query=filter_query,
    #         auto_commit=True,
    #         selection_mode="exc_preload",
    #         selection_until="first_drop",
    #         update_existing=updated_existing_flag,
    #         filter_data=True,
    #         null_offset_f=True,
    #         fit_model=True,
    #         plot_filter=plot_flag,
    #         plot_extrema=plot_flag,
    #         plot_selection=plot_flag,
    #         plot_f_vs_e=plot_flag,
    #         plot_fit_model=plot_flag
    #     )
    #     if new_sys_v:
    #         sys_v_list.append(new_sys_v)

    for sys in sys_v_list:
        sys.calc_params_from_measurement_versions("poly1d")

    ####

    # Example usage with a list of SystemVersion instances and system_id groups
    system_id_groups = [[1, 2], [3, 4], [5, 6], [8, 9], [10, 11], [12, 13]]

    # Iterate over each group of system_ids and call the plot_measurement_versions_by_system_ids function
    for system_ids in system_id_groups:
        System.plot_measurement_versions_by_system_ids(sys_v_list, system_ids)
    ###

    cc = CableCalculator.create_with_system_version(
        ks_type='example_type',
        stem_diameter_1=100,
        stem_diameter_2=100,
        stem_damaged='no',
        system_identifier="cobra 4t dynamisch",
        distance_horizontal=4.5,
        rope_length=5,
        force=10)

    # Plotten der Curve
    ccv_list = cc.cable_calculator_version

    for ccv in ccv_list:
        ccv.plot_catenary_curve()

    test_df_2 = cc.calculate_values_for_all_versions(force=10)

    import random
    random_list = random.sample(sys_v_list, 20)
    SystemVersion.plot_multi_cable_models(random_list, 8)

    SystemVersion.plot_difference_in_cable_models(sys_v_list[0], sys_v_list[10])
    SystemVersion.plot_difference_in_cable_models(sys_v_list[1], sys_v_list[11])
    SystemVersion.plot_difference_in_cable_models(sys_v_list[16], sys_v_list[7])
    SystemVersion.plot_difference_in_cable_models(sys_v_list[3], sys_v_list[9])
    SystemVersion.plot_difference_in_cable_models(sys_v_list[2], sys_v_list[14])
    SystemVersion.plot_difference_in_cable_models(sys_v_list[12], sys_v_list[5])
    SystemVersion.plot_difference_in_cable_models(sys_v_list[18], sys_v_list[21])
    SystemVersion.plot_difference_in_cable_models(sys_v_list[22], sys_v_list[1])
    SystemVersion.plot_difference_in_cable_models(sys_v_list[0], sys_v_list[25])
