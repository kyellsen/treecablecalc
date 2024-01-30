from pathlib import Path
import numpy as np
import pandas as pd

from sqlalchemy import Column, Integer
from .base_class import BaseClass

from kj_logger import get_logger

logger = get_logger(__name__)


class Measurement(BaseClass):
    __tablename__ = 'measurements'  # Eindeutiger Name für die Tabelle

    id = Column(Integer, primary_key=True)  # Primärschlüsselspalte

    def __init__(self):
        super().__init__()
        # Hier können weitere Initialisierungen hinzugefügt werden

