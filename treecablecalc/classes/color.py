from ..common_imports.imports_classes import *

logger = get_logger(__name__)

class Color(BaseClass):
    """
    This class represents a color.
    """
    __tablename__ = 'Color'

    color_id = Column(Integer, primary_key=True, autoincrement=True)
    color = Column(String)
    farbe = Column(String)  # German for color

    anti_abrasion_hose = relationship("AntiAbrasionHose", backref="color", lazy="joined", cascade='all, delete-orphan',
                                      order_by='AntiAbrasionHose.anti_abrasion_hose_id')
    cable = relationship("Cable", backref="color", lazy="joined", cascade='all, delete-orphan',
                         order_by='Cable.cable_id')
    sling = relationship("Sling", backref="color", lazy="joined", cascade='all, delete-orphan',
                         order_by='Sling.sling_id')

    def __init__(self, color: str = None, farbe: str = None):
        super().__init__()
        self.color = color
        self.farbe = farbe
