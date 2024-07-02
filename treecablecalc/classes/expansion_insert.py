from ..common_imports.imports_classes import *

from .material_type import MaterialType
from .producer import Producer
from .brand import Brand

logger = get_logger(__name__)

class ExpansionInsert(BaseClass):
    """
    This class represents an expansion insert.
    """
    __tablename__ = 'ExpansionInsert'

    material_type_id = Column(Integer, ForeignKey('MaterialType.material_type_id'), nullable=True)
    expansion_insert_id = Column(Integer, primary_key=True, autoincrement=True)
    producer_id = Column(Integer, ForeignKey('Producer.producer_id'), nullable=True)
    brand_id = Column(Integer, ForeignKey('Brand.brand_id'), nullable=True)
    expansion_insert = Column(String)
    load_ztv = Column(Integer)
    date = Column(DateTime)

    system = relationship("System", backref="expansion_insert", lazy="joined", cascade='all, delete-orphan',
                          order_by='System.system_id')

    def __init__(self, expansion_insert: str = None, load_ztv: int = None, date: datetime = None):
        super().__init__()
        self.expansion_insert = expansion_insert
        self.load_ztv = load_ztv
        self.date = date
