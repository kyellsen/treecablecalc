from ..common_imports.imports_classes import *

logger = get_logger(__name__)


class ElongationProperties(BaseClass):
    """
    This class represents elongation properties.
    """
    __tablename__ = 'ElongationProperties'

    elongation_properties_id = Column(Integer, primary_key=True, autoincrement=True)
    elongation_properties = Column(String)

    cable = relationship("Cable", backref="elongation_properties", lazy="joined", cascade='all, delete-orphan',
                         order_by='Cable.cable_id')
    system = relationship("System", backref="elongation_properties", lazy="joined", cascade='all, delete-orphan',
                          order_by='System.system_id')

    def __init__(self, elongation_properties: str = None):
        super().__init__()
        self.elongation_properties = elongation_properties
