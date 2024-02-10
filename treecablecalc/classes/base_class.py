from typing import Optional

from kj_logger import get_logger
from kj_core.classes.core_base_class import CoreBaseClass

import treecablecalc

logger = get_logger(__name__)


class BaseClass(CoreBaseClass):
    __abstract__ = True

    def __init__(self):
        super().__init__()

    def get_child_attr_name(self) -> Optional[str]:
        """
        Get the attribute name of the children based on the class name.
        Should be overridden in specific packages due to the specific hierarchie.

        Returns
        -------
        str or None
            The attribute name if the class name is found, otherwise None.
        """
        mapping = {
            "Project": "series",
            "Series": "measurement",
            "Measurement": "measurement_version",
            "MeasurementVersion": "data_tcc"
        }

        # Store the attribute name corresponding to the class in a variable
        child_attr_name = mapping.get(self.__class__.__name__)

        return child_attr_name

    @classmethod
    def get_config(cls):
        return treecablecalc.CONFIG

    @classmethod
    def get_data_manager(cls):
        return treecablecalc.DATA_MANAGER

    @classmethod
    def get_database_manager(cls):
        return treecablecalc.DATABASE_MANAGER

    @classmethod
    def get_plot_manager(cls):
        return treecablecalc.PLOT_MANAGER
