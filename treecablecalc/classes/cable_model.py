import pickle
from ..common_imports.imports_classes import *
from sqlalchemy import LargeBinary


class CableModel(BaseClass):
    __tablename__ = 'CableModel'

    cable_model_id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    measurement_version_id = Column(Integer,
                                    ForeignKey('MeasurementVersion.measurement_version_id', onupdate='CASCADE'))
    measurement_id = Column(Integer, ForeignKey('Measurement.measurement_id', onupdate='CASCADE'))

    # Ändern Sie den Datentyp von 'model' in LargeBinary für die Serialisierung
    _model_data = Column('model', LargeBinary)
    degree = Column(Integer)
    quality_r2 = Column(Float)
    lower_bound = Column(Float)
    upper_bound = Column(Float)

    def __init__(self, model: np.poly1d = None, degree: int = None, quality: float = None,
                 lower_bound: float = None, upper_bound: float = None):
        super().__init__()
        self.model = model  # Verwenden der Property für die Serialisierung
        self.degree = degree
        self.quality_r2 = quality
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def model(self) -> np.poly1d:
        """Deserialisiert das Pipeline-Objekt aus dem gespeicherten Binärdaten."""
        if self._model_data is not None:
            return pickle.loads(self._model_data)
        raise ValueError(f"Fail in pickle.loads(self._model_data)")

    @model.setter
    def model(self, model: np.poly1d):
        """Serialisiert das Pipeline-Objekt für die Speicherung in der Datenbank."""
        self._model_data = pickle.dumps(model)

    def __str__(self):
        return (f"CableModel(degree={self.degree}, "
                f"quality_r2={self.quality_r2:.4f} (R^2), "
                f"lower_bound={self.lower_bound:.2f}, "
                f"upper_bound={self.upper_bound:.2f})")
