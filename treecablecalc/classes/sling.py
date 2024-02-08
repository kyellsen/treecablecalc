from ..common_imports.imports_classes import *

from .material_type import MaterialType
from .producer import Producer
from .brand import Brand
from .color import Color
from .fiber_type import FiberType

logger = get_logger(__name__)

class Sling(BaseClass):
    """
    This class represents a sling.
    """
    __tablename__ = 'Sling'

    material_type_id = Column(Integer, ForeignKey('MaterialType.material_type_id'), nullable=False)
    sling_id = Column(Integer, primary_key=True, autoincrement=True)
    producer_id = Column(Integer, ForeignKey('Producer.producer_id'), nullable=False)
    brand_id = Column(Integer, ForeignKey('Brand.brand_id'), nullable=False)
    sling = Column(String)
    color_id = Column(Integer, ForeignKey('Color.color_id'), nullable=False)
    load_ztv = Column(Integer)
    mbl_producer = Column(Float)
    fiber_type_id = Column(Integer, ForeignKey('FiberType.fiber_type_id'), nullable=False)
    datum = Column(DateTime)

    system = relationship("System", backref="sling", lazy="joined", cascade='all, delete-orphan',
                          order_by='System.system_id')

    def __init__(self, sling: str = None, load_ztv: int = None, mbl_producer: float = None, datum: datetime = None):
        super().__init__()
        self.sling = sling
        self.load_ztv = load_ztv
        self.mbl_producer = mbl_producer
        self.datum = datum
