import pandas as pd

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

    def __init__(self, measurement_version_id=None, measurement_version_name=None, measurement_id=None,
                 data_tcc_id: int = None):
        super().__init__()
        self.measurement_version_id = measurement_version_id
        self.measurement_version_name = measurement_version_name
        self.measurement_id = measurement_id
        self.data_tcc_id = data_tcc_id

    @classmethod
    def create_from_csv(cls, csv_filepath: str, measurement_id: int, measurement_version_name: str = None) -> Optional['MeasurementVersion']:
        """
        Loads TCC Data from a CSV file.

        :param csv_filepath: Path to the CSV file.
        :param measurement_id: ID of the measurement to which the data belongs.
        :param measurement_version_name: Version Name of the data.
        :return: MeasurementVersion object.
        """
        config = cls.get_config()
        obj = cls(measurement_id=measurement_id, measurement_version_name=measurement_version_name)

        data_directory = config.data_directory
        folder: str = config.DataTCC.data_directory
        filename: str = cls.get_data_manager().get_new_filename(measurement_id,
                                                                prefix=f"tcc_{measurement_version_name}",
                                                                file_extension="feather")

        data_filepath = str(data_directory / folder / filename)

        data_tcc = DataTCC.create_from_csv(csv_filepath, data_filepath, obj.measurement_version_id)

        obj.data_tcc = data_tcc

        session = cls.get_database_manager().session
        session.add(obj)
        logger.info(f"Created new '{obj}'")
        return obj

    def update_from_csv(self, csv_filepath: str) -> Optional['MeasurementVersion']:
        self.data_tcc = self.data_tcc.update_from_csv(csv_filepath)
        logger.info(f"Updated new '{self}'")
        return self

