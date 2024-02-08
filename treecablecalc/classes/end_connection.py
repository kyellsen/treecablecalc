from ..common_imports.imports_classes import *

logger = get_logger(__name__)

class EndConnection(BaseClass):
    """
    This class represents an end connection.
    """
    __tablename__ = 'EndConnection'

    end_connection_id = Column(Integer, primary_key=True, autoincrement=True)
    end_connection = Column(String, unique=True, nullable=False)

    def __init__(self, end_connection: str = None):
        super().__init__()
        self.end_connection = end_connection
