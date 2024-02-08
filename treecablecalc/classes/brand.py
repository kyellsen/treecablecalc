from ..common_imports.imports_classes import *

logger = get_logger(__name__)

class Brand(BaseClass):
    """
    This class represents a brand.
    """
    __tablename__ = 'Brand'

    brand_id = Column(Integer, primary_key=True, autoincrement=True)
    brand_short = Column(String)
    brand_long = Column(String)
    domain = Column(String)

    anti_abrasion_hose = relationship("AntiAbrasionHose", backref="brand", lazy="joined", cascade='all, delete-orphan',
                                      order_by='AntiAbrasionHose.anti_abrasion_hose_id')
    cable = relationship("Cable", backref="brand", lazy="joined", cascade='all, delete-orphan',
                         order_by='Cable.cable_id')
    expansion_insert = relationship("ExpansionInsert", backref="brand", lazy="joined", cascade='all, delete-orphan',
                         order_by='ExpansionInsert.expansion_insert_id')
    material_add = relationship("MaterialAdd", backref="brand", lazy="joined", cascade='all, delete-orphan',
                         order_by='MaterialAdd.material_add_id')
    shock_absorber = relationship("ShockAbsorber", backref="brand", lazy="joined", cascade='all, delete-orphan',
                                order_by='ShockAbsorber.shock_absorber_id')
    sling = relationship("Sling", backref="brand", lazy="joined", cascade='all, delete-orphan',
                         order_by='Sling.sling_id')

    def __init__(self, brand_short: str = None, brand_long: str = None, domain: str = None):
        super().__init__()
        self.brand_short = brand_short
        self.brand_long = brand_long
        self.domain = domain
