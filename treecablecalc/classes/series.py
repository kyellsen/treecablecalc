from ..common_imports.imports_classes import *

logger = get_logger(__name__)


class Series(BaseClass):
    """
    This class represents a series of measurements.
    """
    __tablename__ = 'Series'

    series_id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    description = Column(String)
    filepath_csv = Column(String, unique=True)

    measurement = relationship("Measurement", backref="series", lazy="joined", cascade='all, delete-orphan',
                               order_by='Measurement.measurement_id')

    def __init__(self, description: str = None, filepath_csv: str = None):
        super().__init__()
        self.description = description
        self.filepath_csv = filepath_csv

    def __str__(self) -> str:
        """
        Represents the Series instance as a string.

        :return: A string representation of the Series instance.
        """
        return f"Series(series_id={self.series_id}, description={self.description}, filepath_csv={self.filepath_csv})"

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the Series instance for debugging.

        :return: A detailed string representation of the Series instance.
        """
        return (f"<Series(series_id={self.series_id}, description={self.description}, "
                f"filepath_csv={self.filepath_csv})>")
