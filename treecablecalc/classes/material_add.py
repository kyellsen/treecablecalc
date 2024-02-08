from ..common_imports.imports_classes import *

from .material_type import MaterialType
from .producer import Producer
from .brand import Brand

logger = get_logger(__name__)

class MaterialAdd(BaseClass):
    """
    This class represents an additional material.
    """
    __tablename__ = 'MaterialAdd'

    material_type_id = Column(Integer, ForeignKey('MaterialType.material_type_id'), nullable=False)
    material_add_id = Column(Integer, primary_key=True, autoincrement=True)
    producer_id = Column(Integer, ForeignKey('Producer.producer_id'), nullable=False)
    brand_id = Column(Integer, ForeignKey('Brand.brand_id'), nullable=False)
    material_add = Column(String)
    load_ztv = Column(Integer)
    date = Column(DateTime)

    system = relationship("System", backref="material_add", lazy="joined", cascade='all, delete-orphan',
                          order_by='System.system_id')

    def __init__(self, material_add: str = None, load_ztv: int = None, date: datetime = None):
        super().__init__()
        self.material_add = material_add
        self.load_ztv = load_ztv
        self.date = date
