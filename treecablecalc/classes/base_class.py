from kj_core import get_logger
from kj_core.classes.core_base_class import CoreBaseClass

import treecablecalc

logger = get_logger(__name__)


class BaseClass(CoreBaseClass):
    __abstract__ = True
    _config = None
    _data_manager = None
    _database_manager = None
    _plot_manager = None

    def __init__(self):
        super().__init__()

    @property
    def CONFIG(self):
        if self._config is None:
            self._config = treecablecalc.CONFIG
        return self._config

    @property
    def DATA_MANAGER(self):
        if self._data_manager is None:
            self._data_manager = treecablecalc.DATA_MANAGER
        return self._data_manager

    @property
    def DATABASE_MANAGER(self):
        if self._database_manager is None:
            self._database_manager = treecablecalc.DATABASE_MANAGER
        return self._database_manager

    @property
    def PLOT_MANAGER(self):
        if self._plot_manager is None:
            self._plot_manager = treecablecalc.PLOT_MANAGER
        return self._plot_manager
