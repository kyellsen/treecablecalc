from ..common_imports.imports_classes import *

logger = get_logger(__name__)

from .data_tcc import DataTCC

class MeasurementVersion(BaseClass):
    """
    This class represents a measurement in the system.
    """
    __tablename__ = 'MeasurementVersion'

    measurement_version_id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    measurement_version_name = Column(String)
    measurement_id = Column(Integer, ForeignKey('Measurement.measurement_id', onupdate='CASCADE'), nullable=False)

    data_tcc = relationship("DataTCC", backref="measurement_version", uselist=False, cascade='all, delete-orphan')

    def __init__(self, measurement_version_id=None, measurement_version_name=None, measurement_id=None, data_tcc_id: int = None):
        super().__init__()
        self.measurement_version_id = measurement_version_id
        self.measurement_version_name = measurement_version_name
        self.measurement_id = measurement_id
        self.data_tcc_id = data_tcc_id
