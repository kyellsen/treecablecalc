from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd

from kj_core.core_config import CoreConfig

from kj_logger import get_logger

logger = get_logger(__name__)


class Config(CoreConfig):
    """
    Configuration class for the package, extending the core configuration.
    Provides package-specific settings and default values.
    """
    # Override default working directory specific
    package_name = "treecablecalc"
    package_name_short = "tcc"
    # Override default working directory specific
    default_working_directory = r"C:\kyellsen\006_Packages\treecablecalc\working_directory_tms"

    def __init__(self, working_directory: Optional[str] = None):
        """
        Initializes the configuration settings, building upon the core configuration.
        """
        super().__init__(f"{working_directory}/{self.package_name_short}")
        logger.info(f"{self} initialized! Code: 002")

    class Measurement:
        pass

    class MeasurementVersion:
        measurement_version_name_default = "raw"

    class DataTCC:
        data_directory = "data_tcc"

        columns_to_use = ["Weg_Time [s]", "Weg [mm]", "Kraft [kN]"]  # Don´t use "Kraft_Time [s]"
        time_column = "Weg_Time [s]"  # "Time" is the index! "Kraft_Time [s]" and "Weg_Time [s]" are identical. Read only one.
        dtype_dict = {"Weg [mm]": np.float64, "Kraft [kN]": np.float64}

        rename_dict = {"Weg [mm]": "x", "Kraft [kN]": "f"}

        column_names = ["x", "f"]
        column_names_plotting = ["Time [s]", "Distanz [mm]", "Force [kN]"]
