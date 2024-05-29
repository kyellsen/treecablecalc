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


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Main
    main_path = Path(r"C:\kyellsen\005_Projekte\2023_TreeCableCalc")
    analyse_name = r"2023_Kronensicherung_Stuttgart_2024-05-27"
    data_path = main_path / "021_Daten_Clean"  # Für alle Daten-Importe des Projektes gemeinsam
    working_directory = main_path / "030_Analysen" / analyse_name / "working_directory"  # Für alle Daten-Exporte des Projektes gemeinsam
    source_db_name = "TreeCableCalc_Stuttgart_2023-12-08.db"
    source_db = data_path / source_db_name

    CONFIG, LOG_MANAGER, DATA_MANAGER, DATABASE_MANAGER, PLOT_MANAGER = tcc.setup(working_directory=str(working_directory), log_level="info")
    DATABASE_MANAGER.duplicate(database_path=str(source_db))
    DATABASE_MANAGER.connect(db_name=str(source_db_name))
    m_list: List[Measurement] = DATABASE_MANAGER.load(class_name=Measurement)
    m_list = [m for m in m_list if m.execute != "dont_use" and m.series.description == "2023-01-10_HAWK"]

    df_list = []
    for m in m_list:
        m.load_preconfigured()
        df = m.get_param_df()
        df_list.append(df)

    big_df = pd.concat(df_list, ignore_index=True)
