from kj_core.classes.core_data_class import CoreDataClass
from kj_core.df_utils.validate import validate_df

from ..common_imports.imports_classes import *

logger = get_logger(__name__)


class DataTCC(CoreDataClass, BaseClass):
    __tablename__ = 'DataTCC'
    data_id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    data_filepath = Column(String, unique=True)
    data_changed = Column(Boolean)
    measurement_version_id = Column(Integer,
                                    ForeignKey('MeasurementVersion.measurement_version_id', onupdate='CASCADE'),
                                    nullable=False)
    tempdrift_method = Column(String)

    def __init__(self, data_id: int = None, data: pd.DataFrame = None, data_filepath: str = None, data_changed: bool = False, datetime_added=None,
                 datetime_last_edit=None, measurement_version_id: int = None):
        CoreDataClass.__init__(self, data_id=data_id, data=data, data_filepath=data_filepath, data_changed=data_changed,
                               datetime_added=datetime_added, datetime_last_edit=datetime_last_edit)

        self.measurement_version_id = measurement_version_id

    @classmethod
    def create_from_csv(cls, csv_filepath: str, data_filepath: str, measurement_version_id: int) -> Optional['DataTCC']:

        data: pd.DataFrame = cls.read_data_csv(csv_filepath)
        obj = cls(data=data, data_filepath=data_filepath, measurement_version_id=measurement_version_id)
        logger.info(f"Created new '{obj}'")
        return obj

    def update_from_csv(self, csv_filepath: str) -> Optional['DataTCC']:
        self.data = self.read_data_csv(csv_filepath)
        logger.info(f"Updated new '{self}'")

        return self

    @classmethod
    @dec_runtime
    def read_data_csv(cls, filepath: str) -> Optional[pd.DataFrame]:
        """
        Reads data from a CSV file.

        :param filepath: Path to the CSV file.
        :return: DataFrame with the read data.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"search_path {filepath} does not exist.")
            return

        config = cls.get_config().DataTCC
        time_column: str = config.time_column
        columns_to_use: List = config.columns_to_use
        dtype_dict: Dict = config.dtype_dict
        rename_dict: Dict = config.rename_dict
        try:
            df = pd.read_csv(
                filepath,
                decimal=",",
                sep=";",
                usecols=columns_to_use,
                dtype=dtype_dict  # Datentypen für die Spalten
            )

            # Konvertiere die Zeitdaten in datetime
            df['time'] = pd.to_datetime(df[time_column].astype(float), unit='s', origin=pd.Timestamp('2000-01-01'))
            # Entferne die ursprüngliche Zeit-Spalte
            df.drop(columns=[time_column], inplace=True)

            # Setze die umgewandelte Zeit-Spalte als Index
            df.set_index('time', inplace=True)

            # Umbenennung der Spalten
            df.rename(columns=rename_dict, inplace=True)

        except pd.errors.ParserError as e:
            logger.error(f"Error while reading the file {filepath.stem}. Please check the file format.")
            raise e
        except Exception as e:
            logger.error(f"Unusual error while loading {filepath.stem}: {e}")
            raise e
        return df

    def validate_data(self) -> bool:
        """
        Checks if the DataFrame data is valid and contains the required columns.

        Returns:
            bool: True if the DataFrame is valid, False otherwise.
        """
        try:
            validate_df(df=self.data, columns=self.get_config().DataTCC.column_names)
            logger.debug(f"Data validation for '{self}' correct!")
            return True
        except Exception as e:
            logger.error(f"Error during validation of the DataFrame: {e}")
            return False
