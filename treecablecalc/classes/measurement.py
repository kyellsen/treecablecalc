import pandas as pd

from ..common_imports.imports_classes import *

from .series import Series
from .system import System
from .measurement_version import MeasurementVersion

logger = get_logger(__name__)


class Measurement(BaseClass):
    """
    This class represents a measurement in the system.
    """
    __tablename__ = 'Measurement'

    measurement_id = Column(Integer, primary_key=True, autoincrement=True, unique=True, nullable=False)
    series_id = Column(Integer, ForeignKey("Series.series_id"), nullable=False)
    sample_id = Column(Integer, nullable=False)
    system_id = Column(Integer, ForeignKey("System.system_id"), nullable=False)
    file_name_csv = Column(String, nullable=False)
    pre_tension_load = Column(Float)
    prio = Column(Integer)
    cable_m = Column(Float)
    expansion_insert_count = Column(Integer)
    anti_abrasion_hose_m = Column(Float)
    material_add_count = Column(Integer)
    failure_loc = Column(String)
    left_head = Column(Float)
    peak_height = Column(Integer)
    peak_prominence = Column(Integer)
    peak_distance = Column(Integer)
    peak_width = Column(Integer)
    valley_height = Column(Integer)
    valley_prominence = Column(Integer)
    valley_distance = Column(Integer)
    valley_width = Column(Integer)
    shock_absorber_l1 = Column(Float)
    shock_absorber_l2 = Column(Float)
    note = Column(String)
    execute = Column(String)
    date = Column(DateTime)

    measurement_version = relationship(MeasurementVersion, backref="measurement", lazy="joined",
                                       cascade='all, delete-orphan',
                                       order_by='MeasurementVersion.measurement_version_id')

    def __init__(self, measurement_id: int, series_id: int, sample_id: int, system_id: int,
                 file_name_csv: str, pre_tension_load: float = None, prio: int = None,
                 cable_m: float = None, expansion_insert_count: int = None, anti_abrasion_hose_m: float = None,
                 material_add_count: int = None, failure_loc: str = None, left_head: float = None,
                 peak_prominence: int = None, valley_prominence: int = None, shock_absorber_l1: float = None,
                 shock_absorber_l2: float = None, note: str = None, execute: str = None, date: datetime = None):
        super().__init__()
        self.measurement_id = measurement_id
        self.series_id = series_id
        self.sample_id = sample_id
        self.system_id = system_id
        self.file_name_csv = file_name_csv
        self.pre_tension_load = pre_tension_load
        self.prio = prio
        self.cable_m = cable_m
        self.expansion_insert_count = expansion_insert_count
        self.anti_abrasion_hose_m = anti_abrasion_hose_m
        self.material_add_count = material_add_count
        self.failure_loc = failure_loc
        self.left_head = left_head
        self.peak_prominence = peak_prominence
        self.valley_prominence = valley_prominence
        self.shock_absorber_l1 = shock_absorber_l1
        self.shock_absorber_l2 = shock_absorber_l2
        self.note = note
        self.execute = execute
        self.date = date

    def __str__(self) -> str:
        """
        Represents the Measurement instance as a string.

        :return: A string representation of the Measurement instance.
        """
        return f"Measurement(id={self.measurement_id}, series_id={self.series_id})"

    @property
    def csv_filepath(self) -> str:
        """
        Constructs and returns the file path for a CSV file by combining the series filepath_csv
        and the file_name_csv. Validates the existence of necessary attributes, the ability to
        construct the path, and the presence of the TCC file.

        Raises:
            ValueError: If any required attribute is missing or if the file path cannot be constructed.
            FileNotFoundError: If the constructed file path does not point to an existing file.

        Returns:
            Path: The constructed file path for the CSV file.
        """
        # Ensure necessary attributes are present
        if not getattr(self, 'series', None) or not getattr(self, 'file_name_csv', None):
            raise ValueError("Missing required attributes 'series' or 'file_name_csv'.")

        # Attempt to construct the file path
        try:
            filepath = Path(self.series.filepath_csv) / self.file_name_csv
        except Exception as e:
            raise ValueError(f"Failed to construct file path: {e}")

        # Validate the constructed file path points to an existing file
        if not filepath.is_file():
            raise FileNotFoundError(f"No file found at the constructed path: '{filepath}'.")

        return str(filepath)

    @dec_runtime
    def load_from_csv(self, measurement_version_name: str = None,
                      update_existing: bool = True, load_dont_use: bool = False) -> Optional[MeasurementVersion]:
        """
        Loads data from a CSV file into a MeasurementVersion instance.

        Attempts to find an existing MeasurementVersion based on the provided name or a default.
        If found and `update_existing` is False, returns the found instance without changes.
        If not found, creates a new MeasurementVersion from the CSV.
        If found and `update_existing` is True, updates the existing MeasurementVersion from the CSV.

        Args:
            measurement_version_name (str, optional): Name of the MeasurementVersion. Defaults to None, which uses the default name from config.
            update_existing (bool): Whether to update an existing MeasurementVersion with the same name. Defaults to True.
            load_dont_use (bool):
        Returns:
            Optional[MeasurementVersion]: The updated, newly created, or found MeasurementVersion instance, or None if an error occurs.
        """
        # Überprüfung, ob load_dont_use False ist und self.execute 'dont_use' entspricht.
        if not load_dont_use and self.execute == "dont_use":
            logger.warning(f"Aborting operation for '{self}' as 'execute' is set to 'dont_use'.")
            return None

        logger.info(f"Start loading TCC data from CSV for '{self}'")
        try:
            mv_name = measurement_version_name or self.get_config().MeasurementVersion.measurement_version_name_default

            m_v_present: MeasurementVersion = (self.get_database_manager().session.query(MeasurementVersion)
                                               .filter(MeasurementVersion.measurement_id == self.measurement_id,
                                                       MeasurementVersion.measurement_version_name == mv_name)
                                               .first())

        except Exception as e:
            logger.error(
                f"Failed to retrieve MeasurementVersion '{measurement_version_name}' for Measurement ID '{self.measurement_id}'. Error: {e}")
            return None

        if m_v_present and not update_existing:
            # Fall 1: Ein vorhandenes Objekt existiert und soll nicht aktualisiert werden.
            # Gib das vorhandene Objekt zurück.
            logger.warning(f"Existing measurement_version '{mv_name}' not updated: '{m_v_present}'")
            return m_v_present

        elif not m_v_present:
            # Fall 2: Kein vorhandenes Objekt existiert.
            # Erstelle ein neues Objekt und gib dieses zurück.
            try:
                mv_new = MeasurementVersion.create_from_csv(self.csv_filepath, self.measurement_id, mv_name)
                DATABASE_MANAGER = self.get_database_manager()
                self.measurement_version.append(mv_new)
                DATABASE_MANAGER.commit()
                logger.info(f"New measurement_version '{mv_name}' created: '{mv_new}'")
                return mv_new
            except Exception as e:
                logger.error(f"Failed to create MeasurementVersion '{mv_name}' for '{self}', error: {e}")

        elif m_v_present and update_existing:
            # Fall 3: Ein vorhandenes Objekt existiert und soll aktualisiert werden.
            # Aktualisiere das vorhandene Objekt und gib es zurück.
            try:
                mv_updated = m_v_present.update_from_csv(self.csv_filepath)
                DATABASE_MANAGER = self.get_database_manager()
                self.measurement_version.append(mv_updated)
                DATABASE_MANAGER.commit()
                logger.info(f"Existing measurement_version '{mv_name}' updated: '{mv_updated}'")
                return mv_updated
            except Exception as e:
                logger.error(f"Failed to update MeasurementVersion '{mv_name}' for '{self}', error: {e}")
        return None

    def load_with_features(self, selection_mode: str = "default", selection_until: str = "end",
                           measurement_version_name: str = None, update_existing=True,
                           filter=True, null_offset_f=True, plot_filter=False, plot_extrema=False, plot_selection=False,
                           plot_f_vs_e=False, plot_fit_model=False) -> Optional[MeasurementVersion]:
        if measurement_version_name is None:
            measurement_version_name = f"{selection_mode}_until_{selection_until}"

        config = self.get_config().MeasurementVersion
        if selection_mode not in config.valide_selection_mode:
            raise ValueError
        if selection_until not in config.valide_selection_until:
            raise ValueError

        mv: MeasurementVersion = self.load_from_csv(measurement_version_name, update_existing, load_dont_use=False)
        if filter:
            mv.filter(plot=plot_filter, inplace=True, auto_commit=True)
        if null_offset_f:
            mv.null_offset_f(inplace=True, auto_commit=True)

        mv.calc_features(set_raw=True, inplace=True, auto_commit=True)

        if selection_mode in ["inc_preload", "exc_preload"]:
            mv.calc_extrema(plot=plot_extrema, inplace=True)

            if selection_mode == "inc_preload":
                mv.select_inc_preload(selection_until=selection_until,
                                      plot=plot_selection, inplace=True, auto_commit=True)
            if selection_mode == "exc_preload":
                mv.select_exc_preload(selection_until=selection_until, close_gap=True,
                                      plot=plot_selection, inplace=True, auto_commit=True)
            # Again calc_features after selection
            mv.calc_features(set_raw=False, inplace=True, auto_commit=True)

        if plot_f_vs_e:
            mv.plot_f_vs_e(plot_raw=True)

        if selection_until == "first_drop":
            mv.fit_model(plot=plot_fit_model, inplace=True, auto_commit=True)

        return mv

    def load_preconfigured(self):
        self.load_with_features(selection_mode="default", selection_until="end")
        self.load_with_features(selection_mode="inc_preload", selection_until="first_drop")
        self.load_with_features(selection_mode="exc_preload", selection_until="first_drop")
        self.load_with_features(selection_mode="inc_preload", selection_until="f_max")
        self.load_with_features(selection_mode="exc_preload", selection_until="f_max")

    def get_param_df(self):
        dict_list_interp1 = [mv.get_params_dict(method="interp1d") for mv in self.measurement_version]
        dict_list_poly1d = [mv.get_params_dict(method="poly1d") for mv in self.measurement_version]
        dict_list = dict_list_interp1 + dict_list_poly1d
        dict_df = pd.DataFrame(dict_list)
        return dict_df
