from ..common_imports.imports_classes import *

from .material_type import MaterialType
from .producer import Producer
from .brand import Brand
from .color import Color
from .fiber_type import FiberType
from .elongation_properties import ElongationProperties

logger = get_logger(__name__)

class Cable(BaseClass):
    """
    This class represents a cable.
    """
    __tablename__ = 'Cable'

    material_type_id = Column(Integer, ForeignKey('MaterialType.material_type_id'), nullable=True)
    cable_id = Column(Integer, primary_key=True, autoincrement=True)
    producer_id = Column(Integer, ForeignKey('Producer.producer_id'), nullable=True)
    brand_id = Column(Integer, ForeignKey('Brand.brand_id'), nullable=True)
    cable = Column(String)
    color_id = Column(Integer, ForeignKey('Color.color_id'), nullable=True)
    picture_filename = Column(String)
    load_ztv = Column(Integer)
    pre_tension_load = Column(Integer)
    mbl_producer_cable_system = Column(Float)
    mbl_producer_cable_system_shock_absorber = Column(Float)
    diameter = Column(Integer)
    fiber_type_id = Column(Integer, ForeignKey('FiberType.fiber_type_id'), nullable=True)
    elongation_properties_id = Column(Integer, ForeignKey('ElongationProperties.elongation_properties_id'), nullable=True)
    lifespan = Column(Integer)
    ztv_compliant = Column(Boolean)
    splicing_tool_necessary = Column(Boolean)
    date = Column(DateTime)
    date_last_modification = Column(String)

    system = relationship("System", backref="cable", lazy="joined", cascade='all, delete-orphan',
                          order_by='System.system_id')

    def __init__(self, cable: str = None, picture_filename: str = None, date: datetime = None, date_last_modification: str = None):
        super().__init__()
        self.cable = cable
        self.picture_filename = picture_filename
        self.date = date
        self.date_last_modification = date_last_modification
