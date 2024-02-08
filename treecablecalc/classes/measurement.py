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
