from ..common_imports.imports_classes import *

logger = get_logger(__name__)

class MaterialType(BaseClass):
    """
    This class represents a material type.
    """
    __tablename__ = 'MaterialType'

    material_type_id = Column(Integer, primary_key=True, autoincrement=True)
    material_type = Column(String)

    anti_abrasion_hose = relationship("AntiAbrasionHose", backref="material_typ", lazy="joined", cascade='all, delete-orphan',
                                      order_by='AntiAbrasionHose.anti_abrasion_hose_id')
    cable = relationship("Cable", backref="material_type", lazy="joined", cascade='all, delete-orphan',
                         order_by='Cable.cable_id')
    expansion_insert = relationship("ExpansionInsert", backref="material_typ", lazy="joined", cascade='all, delete-orphan',
                         order_by='ExpansionInsert.expansion_insert_id')
    material_add = relationship("MaterialAdd", backref="material_typ", lazy="joined", cascade='all, delete-orphan',
                         order_by='MaterialAdd.material_add_id')
    shock_absorber = relationship("ShockAbsorber", backref="material_typ", lazy="joined", cascade='all, delete-orphan',
                                order_by='ShockAbsorber.shock_absorber_id')
    sling = relationship("Sling", backref="material_typ", lazy="joined", cascade='all, delete-orphan',
                         order_by='Sling.sling_id')

    def __init__(self, material_type: str = None):
        super().__init__()
        self.material_type = material_type
