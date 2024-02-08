from kj_core.classes.core_data_class import CoreDataClass

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
