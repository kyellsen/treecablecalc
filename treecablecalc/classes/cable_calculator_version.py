from ..common_imports.imports_classes import *
from .system_version import SystemVersion
from ..plotting.plot_catenary_curve import plot_catenary_curve

logger = get_logger(__name__)


class CableCalculatorVersion(BaseClass):
    """
    This class performs calculations for a specific SystemVersion instance.
    """
    __tablename__ = 'CableCalculatorVersion'

    cable_calculator_version_id = Column(Integer, primary_key=True, autoincrement=True)
    cable_calculator_id = Column(Integer, ForeignKey('CableCalculator.cable_calculator_id'))
    system_version_id = Column(Integer, ForeignKey('SystemVersion.system_version_id'))
    _force = Column(Float, nullable=True)  # Force as a database attribute

    system_version = relationship('SystemVersion', backref="cable_calculator_version", lazy="joined",
                                  order_by='SystemVersion.system_version_id', uselist=False)

    def __init__(self, cable_calculator, system_version: SystemVersion, force: Optional[float] = None):
        """
        Initialize the CableCalculatorVersion with a specific SystemVersion instance.

        Args:
            cable_calculator (CableCalculator): The associated CableCalculator instance.
            system_version (SystemVersion): The system version instance.
            force (float, optional): The force applied to the rope in kN.
        """
        super().__init__()
        self.cable_calculator = cable_calculator
        self.system_version = system_version
        self.force = force

    def __str__(self) -> str:
        """
        Represents the CableCalculatorVersion instance as a string.

        :return: A string representation of the CableCalculatorVersion instance.
        """
        return f"CableCalculatorVersion(cable_calculator_version_id={self.cable_calculator_version_id}, system_version_id={self.system_version_id})"

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the CableCalculatorVersion instance for debugging.

        :return: A detailed string representation of the CableCalculatorVersion instance.
        """
        return f"<CableCalculatorVersion(cable_calculator_version_id={self.cable_calculator_version_id}, system_version_id={self.system_version_id})>"

    @property
    def force(self) -> Optional[float]:
        """
        Get the force applied to the rope.

        Returns:
            Optional[float]: The force applied to the rope in kN.
        """
        return self._force

    @force.setter
    def force(self, value: Optional[float]) -> None:
        """
        Set the force applied to the rope.

        Args:
            value (Optional[float]): The force to be applied in kN.
        """
        if value is not None and (value < 0 or value > 100):
            value = None
            logger.error("Force must be between 0 and 100 kN.")

        if value is None:
            self._force = None
            logger.debug("Force set to None.")
        else:
            self._force = value
            logger.debug(f"Force set to {self._force} kN")

    @property
    def selection_mode(self) -> Optional[str]:
        """
        Get the selection_mode from the associated SystemVersion instance.

        Returns:
            Optional[str]: The selection mode, or None if it cannot be retrieved.
        """
        try:
            return self.system_version.selection_mode
        except AttributeError:
            logger.warning("Selection mode could not be retrieved from SystemVersion.")
            return None

    @property
    def expansion_insert_count(self) -> Optional[float]:
        """
        Get the expansion_insert_count from the associated SystemVersion instance.

        Returns:
            Optional[float]: The expansion insert count, or None if it cannot be retrieved.
        """
        try:
            return self.system_version.expansion_insert_count
        except AttributeError:
            logger.warning("Expansion insert count could not be retrieved from SystemVersion.")
            return None

    @property
    def shock_absorber_count(self) -> Optional[float]:
        """
        Get the shock_absorber_count from the associated SystemVersion instance.

        Returns:
            Optional[float]: The shock absorber count, or None if it cannot be retrieved.
        """
        try:
            return self.system_version.shock_absorber_count
        except AttributeError:
            logger.warning("Shock absorber count could not be retrieved from SystemVersion.")
            return None

    @property
    def e_percent_by_f(self) -> Optional[float]:
        """
        Calculate the elongation percentage for the set force.

        Returns:
            Optional[float]: The elongation percentage.
        """
        if self.force is None:
            return None

        e_percent = self.system_version.get_e_by_f_poly1d(force=self.force)
        if e_percent is None:
            logger.warning("Elongation percentage could not be calculated.")
            return None

        logger.debug(f"{self} - force: {self.force}, e_percent: {e_percent}")
        return e_percent

    @property
    def e_absolute_by_f(self) -> Optional[float]:
        """
        Calculate the absolute elongation for the set force.

        Returns:
            Optional[float]: The absolute elongation.
        """
        if self.force is None:
            return None

        if self.cable_calculator.rope_length is None:
            logger.warning("Rope length is not set.")
            return None

        e_percent_by_f = self.e_percent_by_f
        if e_percent_by_f is None:
            logger.warning("Elongation percentage is None.")
            return None

        e_absolut_by_f = self.cable_calculator.rope_length * (e_percent_by_f / 100)
        logger.debug(f"{self} - rope_length: {self.cable_calculator.rope_length}, e_absolute: {e_absolut_by_f}")
        return e_absolut_by_f

    @property
    def range_of_motion(self) -> Optional[float]:
        """
        Calculate the range of motion for the set force.

        Returns:
            Optional[float]: The range of motion.
        """
        if self.force is None:
            return None

        e_absolut_by_f = self.e_absolute_by_f
        if e_absolut_by_f is None:
            logger.warning("Absolute elongation is None.")
            return None

        range_of_motion = e_absolut_by_f + self.cable_calculator.slack_absolute
        logger.debug(f"{self} - force: {self.force}, range_of_motion: {range_of_motion}")
        return range_of_motion

    def plot_catenary_curve(self):
        """
        Plots the catenary curve using the external plotting function and instance attributes.
        """
        # Call the external plotting function with instance attributes
        fig = plot_catenary_curve(
            curve=self.cable_calculator.calculate_catenary_curve(),
            distance_horizontal=self.cable_calculator.distance_horizontal,
            rope_length=self.cable_calculator.rope_length,
            slack_absolute=self.cable_calculator.slack_absolute,
            sag_vertical=self.cable_calculator.sag_vertical,
            force=self.force,
            e_percent_by_f=self.e_percent_by_f,
            e_absolute_by_f=self.e_absolute_by_f,
            range_of_motion=self.range_of_motion,
            system_name=self.cable_calculator.system.system,
            system_version_name=self.system_version.system_version_name
        )

        plot_manager = self.get_plot_manager()
        filename = f'system_id_{self.cable_calculator.system.system_id}_{self.cable_calculator.system.system}_{self.system_version.system_version_name}'
        subdir = f"{self.system_version.system_version_name}/cable_calculator_version_plot_catenary_curve"
        plot_manager.save_plot(fig, filename, subdir)
