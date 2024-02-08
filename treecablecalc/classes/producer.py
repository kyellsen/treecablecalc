from ..common_imports.imports_classes import *

logger = get_logger(__name__)


class Producer(BaseClass):
    """
    This class represents a producer.
    """
    __tablename__ = 'Producer'

    producer_id = Column(Integer, primary_key=True, autoincrement=True)
    producer = Column(String)

    anti_abrasion_hose = relationship("AntiAbrasionHose", backref="producer", lazy="joined",
                                      cascade='all, delete-orphan',
                                      order_by='AntiAbrasionHose.anti_abrasion_hose_id')
    cable = relationship("Cable", backref="producer", lazy="joined", cascade='all, delete-orphan',
                         order_by='Cable.cable_id')
    expansion_insert = relationship("ExpansionInsert", backref="producer", lazy="joined", cascade='all, delete-orphan',
                                    order_by='ExpansionInsert.expansion_insert_id')
    material_add = relationship("MaterialAdd", backref="producer", lazy="joined", cascade='all, delete-orphan',
                                order_by='MaterialAdd.material_add_id')
    shock_absorber = relationship("ShockAbsorber", backref="producer", lazy="joined", cascade='all, delete-orphan',
                                order_by='ShockAbsorber.shock_absorber_id')
    sling = relationship("Sling", backref="producer", lazy="joined", cascade='all, delete-orphan',
                         order_by='Sling.sling_id')

    def __init__(self, producer: str = None):
        super().__init__()
        self.producer = producer
