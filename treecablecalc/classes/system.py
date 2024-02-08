from ..common_imports.imports_classes import *

from .cable import Cable
from .sling import Sling
from .expansion_insert import ExpansionInsert
from .shock_absorber import ShockAbsorber
from .anti_abrasion_hose import AntiAbrasionHose
from .material_add import MaterialAdd
from .elongation_properties import ElongationProperties

logger = get_logger(__name__)


class System(BaseClass):
    """
    This class represents a system.
    """
    __tablename__ = 'System'

    system_id = Column(Integer, primary_key=True, autoincrement=True)
    system = Column(String)
    cable_id = Column(Integer, ForeignKey('Cable.cable_id'), nullable=True)
    sling_id = Column(Integer, ForeignKey('Sling.sling_id'), nullable=True)
    sling_count = Column(Integer)
    expansion_insert_id = Column(Integer, ForeignKey('ExpansionInsert.expansion_insert_id'), nullable=True)
    shock_absorber_id = Column(Integer, ForeignKey('ShockAbsorber.shock_absorber_id'), nullable=True)
    shock_absorber_count = Column(Integer)
    anti_abrasion_hose_id = Column(Integer, ForeignKey('AntiAbrasionHose.anti_abrasion_hose_id'), nullable=True)
    material_add_id = Column(Integer, ForeignKey('MaterialAdd.material_add_id'), nullable=True)
    elongation_properties_id = Column(Integer, ForeignKey('ElongationProperties.elongation_properties_id'),
                                      nullable=True)

    measurement = relationship("Measurement", backref="system", lazy="joined", cascade='all, delete-orphan',
                               order_by='Measurement.measurement_id')

    def __init__(self, system_id: int = None, system: str = None, sling_count: int = None, shock_absorber_count: int = None):
        super().__init__()
        self.system_id = system_id
        self.system = system
        self.sling_count = sling_count
        self.shock_absorber_count = shock_absorber_count
