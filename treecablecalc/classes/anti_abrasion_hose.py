from ..common_imports.imports_classes import *

from .material_type import MaterialType
from .producer import Producer
from .brand import Brand
from .color import Color

logger = get_logger(__name__)


class AntiAbrasionHose(BaseClass):
    """
    This class represents an anti-abrasion hose.
    """
    __tablename__ = 'AntiAbrasionHose'

    material_type_id = Column(Integer, ForeignKey('MaterialType.material_type_id'), nullable=True)
    anti_abrasion_hose_id = Column(Integer, primary_key=True, autoincrement=True)
    producer_id = Column(Integer, ForeignKey('Producer.producer_id'), nullable=True)
    brand_id = Column(Integer, ForeignKey('Brand.brand_id'), nullable=True)
    anti_abrasion_hose = Column(String)
    color_id = Column(Integer, ForeignKey('Color.color_id'), nullable=True)
    load_ztv = Column(Integer)
    date = Column(DateTime)

    system = relationship("System", backref="anti_abrasion_hose", lazy="joined", cascade='all, delete-orphan',
                          order_by='System.system_id')

    def __init__(self, anti_abrasion_hose: str = None, load_ztv: int = None, date: datetime = None):
        super().__init__()
        self.anti_abrasion_hose = anti_abrasion_hose
        self.load_ztv = load_ztv
        self.date = date
