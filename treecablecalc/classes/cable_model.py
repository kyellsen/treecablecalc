import pickle
from ..common_imports.imports_classes import *
from sqlalchemy import LargeBinary


class CableModel(BaseClass):
    __tablename__ = 'CableModel'

    cable_model_id = Column(Integer, primary_key=True, autoincrement=True, unique=True)

    # Ändern Sie den Datentyp von 'model' in LargeBinary für die Serialisierung
    _model_data = Column('model', LargeBinary)
    quality_r2 = Column(Float)
    lower_bound = Column(Float)
    upper_bound = Column(Float)

    measurement_version = relationship("MeasurementVersion", backref="cable_model", lazy="joined", uselist=False, cascade='all, delete-orphan')
    system_version = relationship("SystemVersion", backref="cable_model", lazy="joined", uselist=False, cascade='all, delete-orphan')

    def __init__(self, model: np.poly1d = None, quality: float = None,
                 lower_bound: float = None, upper_bound: float = None):
        super().__init__()
        self.model = model  # Verwenden der Property für die Serialisierung
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

    def __str__(self) -> str:
        """
        Returns a string representation of the CableModel object, ensuring that None values
        in the attributes are handled gracefully.

        Returns:
            str: A string representing the CableModel object with formatted attribute values.
        """
        quality_r2_str = f"{self.quality_r2:.4f}" if self.quality_r2 is not None else "None"
        lower_bound_str = f"{self.lower_bound:.2f}" if self.lower_bound is not None else "None"
        upper_bound_str = f"{self.upper_bound:.2f}" if self.upper_bound is not None else "None"

        return (f"CableModel quality_r2={quality_r2_str} (R^2), "
                f"lower_bound={lower_bound_str}, "
                f"upper_bound={upper_bound_str})")
