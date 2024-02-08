from ..common_imports.imports_classes import *

from .material_type import MaterialType
from .producer import Producer
from .brand import Brand

logger = get_logger(__name__)

class ShockAbsorber(BaseClass):
    """
    This class represents a shock absorber.
    """
    __tablename__ = 'ShockAbsorber'

    material_type_id = Column(Integer, ForeignKey('MaterialType.material_type_id'), nullable=False)
    shock_absorber_id = Column(Integer, primary_key=True, autoincrement=True)
    producer_id = Column(Integer, ForeignKey('Producer.producer_id'), nullable=False)
    brand_id = Column(Integer, ForeignKey('Brand.brand_id'), nullable=False)
    shock_absorber = Column(String)
    load_ztv = Column(String)
    date = Column(DateTime)

    system = relationship("System", backref="shock_absorber", lazy="joined", cascade='all, delete-orphan',
                          order_by='System.system_id')

    def __init__(self, shock_absorber: str = None, load_ztv: str = None, date: datetime = None):
        super().__init__()
        self.shock_absorber = shock_absorber
        self.load_ztv = load_ztv
        self.date = date
