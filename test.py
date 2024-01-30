import treecablecalc as tcc
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from kj_logger import get_logger

from treecablecalc.classes.measurement import Measurement


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Main
    main_path = Path(r"C:\kyellsen\005_Projekte\2023_TreeCableCalc")
    analyse_name = r"2023_Kronensicherung_Stuttgart_2023-12-08"
    data_path = main_path / "021_Daten_Clean"  # Für alle Daten-Importe des Projektes gemeinsam
    working_directory = main_path / "030_Analysen" / analyse_name / "working_directory"  # Für alle Daten-Exporte des Projektes gemeinsam
    db_name = "TreeCableCalc_Stuttgart_2023-12-08.db"
    source_db = data_path / db_name

    CONFIG, LOG_MANAGER, DATA_MANAGER, DATABASE_MANAGER, PLOT_MANAGER = tcc.setup(working_directory=str(working_directory))
    DATABASE_MANAGER.duplicate(database_path=str(source_db))
    DATABASE_MANAGER.connect(db_name=str("test.db"))
    ms = Measurement()




