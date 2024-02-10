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
    peak_prominence = Column(Integer)
    valley_prominence = Column(Integer)
    shock_absorber_l1 = Column(Float)
    shock_absorber_l2 = Column(Float)
    note = Column(String)
    date = Column(DateTime)

    measurement_version = relationship(MeasurementVersion, backref="measurement", lazy="joined",
                                       cascade='all, delete-orphan',
                                       order_by='MeasurementVersion.measurement_version_id')

    def __init__(self, measurement_id: int, series_id: int, sample_id: int, system_id: int,
                 file_name_csv: str, pre_tension_load: float = None, prio: int = None,
                 cable_m: float = None, expansion_insert_count: int = None, anti_abrasion_hose_m: float = None,
                 material_add_count: int = None, failure_loc: str = None, left_head: float = None,
                 peak_prominence: int = None, valley_prominence: int = None, shock_absorber_l1: float = None,
                 shock_absorber_l2: float = None, note: str = None, date: datetime = None):
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
                      update_existing: bool = True) -> Optional[MeasurementVersion]:
        """
        Loads data from a CSV file into a MeasurementVersion instance.

        Attempts to find an existing MeasurementVersion based on the provided name or a default.
        If found and `update_existing` is False, returns the found instance without changes.
        If not found, creates a new MeasurementVersion from the CSV.
        If found and `update_existing` is True, updates the existing MeasurementVersion from the CSV.

        Args:
            measurement_version_name (str, optional): Name of the MeasurementVersion. Defaults to None, which uses the default name from config.
            update_existing (bool): Whether to update an existing MeasurementVersion with the same name. Defaults to True.

        Returns:
            Optional[MeasurementVersion]: The updated, newly created, or found MeasurementVersion instance, or None if an error occurs.
        """

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
