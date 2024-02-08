from ..common_imports.imports_classes import *

logger = get_logger(__name__)

class FiberType(BaseClass):
    """
    This class represents a fiber type in the system.
    """
    __tablename__ = 'FiberType'

    fiber_type_id = Column(Integer, primary_key=True, autoincrement=True)
    fiber_type = Column(String)
    fiber_type_long = Column(String)

    cable = relationship("Cable", backref="fiber_type", lazy="joined", cascade='all, delete-orphan',
                         order_by='Cable.cable_id')
    sling = relationship("Sling", backref="fiber_type", lazy="joined", cascade='all, delete-orphan',
                         order_by='Sling.sling_id')

    def __init__(self, fiber_type: str = None, fiber_type_long: str = None):
        super().__init__()
        self.fiber_type = fiber_type
        self.fiber_type_long = fiber_type_long
