from typing import List, Optional, Union

from scipy.optimize import fsolve

from ..common_imports.imports_classes import *

from .system_version import SystemVersion
from .system import System
from .cable_calculator_version import CableCalculatorVersion

from ..plotting.plot_catenary_curve import plot_catenary_curve

logger = get_logger(__name__)


class CableCalculator(BaseClass):
    """
    This class performs calculations based on SystemVersion instances.
    """
    __tablename__ = 'CableCalculator'

    cable_calculator_id = Column(Integer, primary_key=True, autoincrement=True)
    ks_type = Column(String)
    _stem_diameter_1 = Column(Float)
    _stem_diameter_2 = Column(Float)
    _stem_damaged = Column(String)
    _distance_horizontal = Column(Float)
    _rope_length = Column(Float)
    system_id = Column(Integer, ForeignKey('System.system_id'))

    system = relationship("System", lazy="joined")
    cable_calculator_version = relationship(CableCalculatorVersion, backref="cable_calculator", lazy="joined",
                                            cascade='all, delete-orphan',
                                            order_by='CableCalculatorVersion.cable_calculator_version_id')

    _valid_stem_damaged_values = {"no", "stem_a", "stem_b"}

    def __init__(self, ks_type: str = None, stem_diameter_1: float = None,
                 stem_diameter_2: float = None, stem_damaged: str = None, system: System = None,
                 distance_horizontal: float = None, rope_length: float = None):
        """
        Initialize the CableCalculator with a database session and attributes.

        Args:
            ks_type (str): The type of KS.
            stem_diameter_1 (float): Diameter of the first stem.
            stem_diameter_2 (float): Diameter of the second stem.
            stem_damaged (bool): Indicates if the stem is damaged.
            system (System): The associated system instance.
            distance_horizontal (float): Horizontal distance in meters.
            rope_length (float): Length of the rope in meters.
        """
        super().__init__()
        self.ks_type = ks_type
        self.stem_diameter_1 = stem_diameter_1
        self.stem_diameter_2 = stem_diameter_2
        self.stem_damaged = stem_damaged
        self.system = system
        self.system_id = system.system_id if system else None
        self.distance_horizontal = distance_horizontal
        self.rope_length = rope_length

    def __str__(self) -> str:
        """
        Represents the CableCalculator instance as a string.

        :return: A string representation of the CableCalculator instance.
        """
        return f"CableCalculator(cable_calculator_id={self.cable_calculator_id}, system_id={self.system_id})"

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the CableCalculator instance for debugging.

        :return: A detailed string representation of the CableCalculator instance.
        """
        return f"<CableCalculator(cable_calculator_id={self.cable_calculator_id}, system_id={self.system_id})>"

    @property
    def stem_diameter_1(self) -> Optional[float]:
        return self._stem_diameter_1

    @stem_diameter_1.setter
    def stem_diameter_1(self, value: Optional[float]) -> None:
        if value is not None and not (10 <= value <= 200):
            logger.error("Stem diameter 1 must be between 10 and 200 mm.")
            raise ValueError("Stem diameter 1 must be between 10 and 200 mm.")
        self._stem_diameter_1 = value

    @property
    def stem_diameter_2(self) -> Optional[float]:
        return self._stem_diameter_2

    @stem_diameter_2.setter
    def stem_diameter_2(self, value: Optional[float]) -> None:
        if value is not None and not (10 <= value <= 200):
            logger.error("Stem diameter 2 must be between 10 and 200 mm.")
            raise ValueError("Stem diameter 2 must be between 10 und 200 mm.")
        self._stem_diameter_2 = value

    @property
    def distance_horizontal(self) -> Optional[float]:
        return self._distance_horizontal

    @distance_horizontal.setter
    def distance_horizontal(self, value: Optional[float]) -> None:
        if value is not None and not (0.5 <= value <= 20):
            logger.error("Horizontal distance must be between 0.5 and 20 meters.")
            raise ValueError("Horizontal distance must be between 0.5 and 20 meters.")
        self._distance_horizontal = value
        if self._rope_length is not None:
            self._validate_rope_length()

    @property
    def rope_length(self) -> Optional[float]:
        return self._rope_length

    @rope_length.setter
    def rope_length(self, value: Optional[float]) -> None:
        self._rope_length = value
        if self._distance_horizontal is not None:
            self._validate_rope_length()

    def _validate_rope_length(self) -> None:
        if not (self._distance_horizontal <= self._rope_length <= 1.25 * self._distance_horizontal):
            logger.error(
                "Rope length must be equal to or greater than the horizontal distance and at most 1.25 times the horizontal distance.")
            raise ValueError("Invalid rope length.")

    @property
    def stem_damaged(self) -> Optional[str]:
        return self._stem_damaged

    @stem_damaged.setter
    def stem_damaged(self, value: Optional[str]) -> None:
        if value is not None and value not in self._valid_stem_damaged_values:
            logger.error(f"Invalid value for stem_damaged: {value}. Must be one of {self._valid_stem_damaged_values}.")
            raise ValueError(
                f"Invalid value for stem_damaged: {value}. Must be one of {self._valid_stem_damaged_values}.")
        self._stem_damaged = value

    @property
    def distance_absolute(self) -> Optional[float]:
        """
        Calculate and return the absolute distance using the Pythagorean theorem.

        Returns:
            Optional[float]: The absolute distance in meters, or None if necessary attributes are not set.
        """
        if self._distance_horizontal is None:
            logger.error("Horizontal distance is not set.")
            return None
        return self._distance_horizontal

    @property
    def slack_absolute(self) -> Optional[float]:
        """
        Calculate and return the slack absolute by subtracting the absolute distance from the rope length.

        Returns:
            Optional[float]: The slack absolute in meters, or None if necessary attributes are not set.
        """
        if self._rope_length is None or self.distance_absolute is None:
            logger.error("Rope length or distance absolute is not set.")
            return None
        return self._rope_length - self.distance_absolute

    @classmethod
    def create_with_system_version(cls, system_identifier: Union[int, str], ks_type: Optional[str] = None,
                                   stem_diameter_1: Optional[float] = None, stem_diameter_2: Optional[float] = None,
                                   stem_damaged: Optional[str] = None, distance_horizontal: Optional[float] = None,
                                   rope_length: Optional[float] = None, force: Optional[float] = None,
                                   auto_commit: bool = True) -> Optional['CableCalculator']:
        """
        Class method to create a new instance of CableCalculator based on a system identifier.

        :param system_identifier: Either the system_id (int) or the system_name (str).
        :param ks_type: Optional KS type.
        :param stem_diameter_1: Optional diameter of the first stem.
        :param stem_diameter_2: Optional diameter of the second stem.
        :param stem_damaged: Optional indicator if the stem is damaged.
        :param distance_horizontal: Optional horizontal distance in meters.
        :param rope_length: Optional length of the rope in meters.
        :param force: Optional force in kN.
        :param auto_commit: If True, commits changes to the database. Defaults to True.
        :return: A new instance of CableCalculator or None if the system versions are not found.
        """
        session = cls.get_database_manager().session
        if isinstance(system_identifier, int):
            system = session.query(System).filter_by(system_id=system_identifier).one_or_none()
        else:
            system = session.query(System).filter_by(system=system_identifier).one_or_none()

        if system is None:
            logger.error(f"System identifier '{system_identifier}' could not be resolved to a system.")
            return None

        # Create CableCalculator instance
        try:
            new_cable_calculator = cls(
                ks_type=ks_type,
                stem_diameter_1=stem_diameter_1,
                stem_diameter_2=stem_diameter_2,
                stem_damaged=stem_damaged,
                system=system,
                distance_horizontal=distance_horizontal,
                rope_length=rope_length
            )
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return None

        # Add the new system version to the session first
        session.add(new_cable_calculator)
        session.flush()  # Ensure the CableCalculator gets an ID

        # Retrieve system versions
        system_version_list: List[SystemVersion] = new_cable_calculator.get_system_version()

        if not system_version_list:
            logger.warning(f"No SystemVersion instances found for system_id {system.system_id}.")
            return None

        # Create CableCalculatorVersion instances
        for system_version in system_version_list:
            new_cable_calculator_version = CableCalculatorVersion(
                cable_calculator=new_cable_calculator,
                system_version=system_version,
                force=force
            )
            session.add(new_cable_calculator_version)
            new_cable_calculator.cable_calculator_version.append(new_cable_calculator_version)

        if auto_commit:
            cls.get_database_manager().commit()

        return new_cable_calculator

    def get_system_version(self) -> List[SystemVersion]:
        """
        Retrieve all SystemVersion instances for the associated system and return them as a list.

        :return: List of SystemVersion instances.
        """
        if self.system is None:
            logger.warning("No associated system found.")
            return []

        system_versions = self.system.system_version

        if not system_versions:
            logger.warning(f"No SystemVersion instances found for system {self.system}.")
            return []

        logger.info(
            f"Successfully retrieved {len(system_versions)} SystemVersion instances for system {self.system}.")
        return system_versions

    def calculate_values_for_all_versions(self, force: Optional[float]) -> pd.DataFrame:
        """
        Calculate values for all CableCalculatorVersion instances and return them in a pandas DataFrame.

        Args:
            force (float): The force applied to the rope in kN.

        Returns:
            pd.DataFrame: DataFrame containing the calculated values for each CableCalculatorVersion.
        """
        columns = [
            'cable_calculator_version_id',
            'e_percent_by_f',
            'e_absolute_by_f',
            'range_of_motion',
            'selection_mode',
            'expansion_insert_count',
            'shock_absorber_count',
            'slack_absolute',
            'sag_vertical'
        ]

        data = {column: [] for column in columns}

        for version in self.cable_calculator_version:
            # Set force if force is provided
            if force:
                version.force = force

            values = {
                'selection_mode': version.selection_mode,
                'expansion_insert_count': version.expansion_insert_count,
                'shock_absorber_count': version.shock_absorber_count,
                'cable_calculator_version_id': version.cable_calculator_version_id,
                'e_percent_by_f': version.e_percent_by_f,
                'e_absolute_by_f': version.e_absolute_by_f,
                'range_of_motion': version.range_of_motion,
                'slack_absolute': self.slack_absolute,
                'sag_vertical': self.sag_vertical
            }

            for column in columns:
                data[column].append(values[column])

        df = pd.DataFrame(data)
        logger.info("Calculated values for all CableCalculatorVersion instances and stored in DataFrame.")
        return df

    def calculate_catenary_curve(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate the catenary curve based on horizontal distance and rope length.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: x and y values of the catenary curve, or None if parameters are not set.
        """
        if self.distance_horizontal is None or self.rope_length is None:
            logger.error("Necessary parameters are not set.")
            return None

        if self.rope_length < self.distance_horizontal:
            logger.warning("The rope is too short to span the distance.")
            return None

        h = self.distance_horizontal
        s = self.rope_length

        def equations(a):
            return 2 * a * np.sinh(h / (2 * a)) - s

        # Initial guess
        a_initial_guess = h / 2

        # Solve the equation
        a = fsolve(equations, a_initial_guess)[0]

        # Calculate the y-values of the catenary curve
        x = np.linspace(0, h, 100)
        y = a * (np.cosh((x - h / 2) / a) - np.cosh(h / (2 * a)))

        return x, y

    @property
    def sag_vertical(self) -> Optional[float]:
        """
        Calculate and return the vertical sag of the rope.

        Returns:
            Optional[float]: The vertical sag in meters, or None if necessary parameters are not set.
        """
        curve = self.calculate_catenary_curve()
        if curve is None:
            return None

        x, y = curve
        return np.max(-y)

    def plot_catenary_curve(self):
        """
        Plots the catenary curve using the external plotting function and instance attributes.
        """
        # Call the external plotting function with instance attributes
        fig = plot_catenary_curve(
            curve=self.calculate_catenary_curve(),
            distance_horizontal=self.distance_horizontal,
            rope_length=self.rope_length,
            slack_absolute=self.slack_absolute,
            sag_vertical=self.sag_vertical,
            system_name=self.system.system
        )

        plot_manager = self.get_plot_manager()
        filename = f'system_id_{self.system.system_id}_{self.system.system}'
        subdir = f"{self.system_version_name}/cable_calculator_plot_catenary_curve"
        plot_manager.save_plot(fig, filename, subdir)
