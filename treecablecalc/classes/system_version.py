import json

from ..common_imports.imports_classes import *
from .cable_model import CableModel

from ..plotting.plot_average_poly1d import plt_average_poly1d
from ..plotting.plot_cable_models import plot_cable_models, plot_cable_models_difference

logger = get_logger(__name__)


class SystemVersion(BaseClass):
    """
    This class represents a system version.
    """
    __tablename__ = 'SystemVersion'

    system_version_id = Column(Integer, primary_key=True, autoincrement=True)
    system_version_name = Column(String)
    system_id = Column(Integer, ForeignKey("System.system_id"), nullable=False)
    cable_model_id = Column(Integer, ForeignKey('CableModel.cable_model_id', onupdate='CASCADE'), nullable=True)
    filter_query = Column(String)
    measurement_version_count = Column(Integer)

    # New columns for additional attributes
    selection_mode = Column(String)
    selection_until = Column(String)
    null_offset = Column(Float)
    filter_flag = Column(Boolean)
    e_by_f_method = Column(String)
    pre_tension_load = Column(Float)
    d_min = Column(Float)
    d_max = Column(Float)
    f_min = Column(Float)
    f_max = Column(Float)
    e_at_load_ztv = Column(Float)
    e_at_pre_tension_load = Column(Float)
    e_at_f_max = Column(Float)
    expansion_insert_count = Column(Float)
    shock_absorber_count = Column(Float)
    shock_absorber_l_delta = Column(Float)
    failure_loc = Column(String)

    measurement_version = relationship('MeasurementVersion', backref="system_version", lazy="joined",
                                       order_by='MeasurementVersion.measurement_version_id')

    def __init__(self, system_version_id: int = None, system_version_name: str = None, system_id: int = None,
                 cable_model_id: int = None, filter_query: str = None):
        super().__init__()
        self.system_version_id = system_version_id
        self.system_version_name = system_version_name
        self.system_id = system_id
        self.cable_model_id = cable_model_id
        self.filter_query = filter_query

        self.measurement_version_count = None

        # Calc from MeasurementVersions
        self.selection_mode = None
        self.selection_until = None
        self.null_offset = None
        self.filter_flag = None
        self.e_by_f_method = None

        self.pre_tension_load = None

        self.d_min = None
        self.d_max = None
        self.f_min = None
        self.f_max = None

        self.cable_model = None

        self.e_at_load_ztv = None
        self.e_at_pre_tension_load = None
        self.e_at_f_max = None

        self.expansion_insert_count = None
        self.shock_absorber_count = None
        self.shock_absorber_l_delta = None
        self.failure_loc = None

    def __str__(self) -> str:
        """
        Represents the SystemVersion instance as a string.

        :return: A string representation of the SystemVersion instance.
        """
        return f"SystemVersion(system_version_id={self.system_version_id}, system_version_name={self.system_version_name})"

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the SystemVersion instance for debugging.

        :return: A detailed string representation of the SystemVersion instance.
        """
        return f"<SystemVersion(system_version_id={self.system_version_id}, system_version_name={self.system_version_name})>"

    def check_measurement_versions_attribute(self, attribute_name: str):
        """
        Check if all instances of MeasurementVersion have the same value for the given attribute.

        :param attribute_name: The attribute to check across all MeasurementVersion instances.
        :return: The common value of the attribute if all instances share the same value, otherwise log an error.
        """
        if not self.measurement_version:
            logger.error("No MeasurementVersion instances available.")
            return None

        if not hasattr(self.measurement_version[0], attribute_name):
            logger.error(f"Attribute {attribute_name} does not exist in MeasurementVersion instances.")
            return None

        first_value = getattr(self.measurement_version[0], attribute_name)
        for mv in self.measurement_version:
            if not hasattr(mv, attribute_name) or getattr(mv, attribute_name) != first_value:
                logger.error(f"Not all MeasurementVersion instances have the same value for {attribute_name}.")
                return None

        return first_value

    def calculate_mean_for_attribute(self, attribute_name: str) -> Optional[float]:
        """
        Calculate the mean value for a numeric attribute across all MeasurementVersion instances.

        :param attribute_name: The attribute to calculate the mean for.
        :return: The mean value of the attribute if it is numeric and present, otherwise log info and return None.
        """
        if not self.measurement_version:
            logger.info("No MeasurementVersion instances available.")
            return None

        values = []
        for mv in self.measurement_version:
            if hasattr(mv, attribute_name):
                value = getattr(mv, attribute_name)
                if isinstance(value, (int, float)) and not np.isnan(value):
                    values.append(value)

        if not values:
            logger.info(f"No valid numeric values found for attribute {attribute_name}.")
            return None

        mean_value = np.mean(values)
        return mean_value

    def calc_params_from_measurement_versions(self, e_by_f_method: str = 'poly1d'):
        """
        Args:
            e_by_f_method (str): The method to use ('interp1d' or 'poly1d')

        :return:
        """
        mv_list = self.measurement_version

        if len(mv_list) == 0:
            logger.warning(f"{self}: No merge possible, mv_list empty.")
            return

        self.measurement_version_count = len(mv_list)

        self.selection_mode = self.check_measurement_versions_attribute('selection_mode')
        self.selection_until = self.check_measurement_versions_attribute('selection_until')
        self.filter_flag = self.check_measurement_versions_attribute('filter_flag')
        self.pre_tension_load = self.check_measurement_versions_attribute('pre_tension_load')
        self.null_offset = self.calculate_mean_for_attribute('null_offset')
        self.d_min = self.calculate_mean_for_attribute('d_min')
        self.d_max = self.calculate_mean_for_attribute('d_max')
        self.f_min = self.calculate_mean_for_attribute('f_min')
        self.f_max = self.calculate_mean_for_attribute('f_max')
        self.expansion_insert_count = self.calculate_mean_for_attribute('expansion_insert_count')
        self.shock_absorber_count = self.calculate_mean_for_attribute('shock_absorber_count')
        self.shock_absorber_l_delta = self.calculate_mean_for_attribute('shock_absorber_l_delta')
        self.failure_loc = json.dumps([mv.measurement.failure_loc for mv in self.measurement_version])

        self.e_by_f_method = e_by_f_method

        if e_by_f_method == 'interp1d':
            logger.critical(f"e_by_f_method 'interp1d' not implemented yet")

        elif e_by_f_method == 'poly1d':
            self.merge_cable_models(inplace=True, auto_commit=True)
            self.e_at_load_ztv = self.get_e_at_load_ztv(100)
            self.e_at_pre_tension_load = self.get_e_at_pre_tension_load()
            self.e_at_f_max = self.get_e_at_f_max()

        else:
            logger.error(f"No e_by_f_method: {e_by_f_method}, use 'interp1d' or 'poly1d'")
            return
        logger.info(
            f"{self}: Successfully calculated parameters from measurement versions using method '{e_by_f_method}'.")
        return self

    def merge_cable_models(self, plot: bool = True, inplace: bool = True, auto_commit: bool = True) -> Optional[
        CableModel]:
        """
        Merge cable models from measurement versions and optionally plot, set inplace, and commit changes.

        :param plot: Whether to plot the average polynomial (default is True)
        :param inplace: Whether to set the resulting model inplace (default is True)
        :param auto_commit: Whether to commit the changes to the database (default is True)
        :return: The averaged CableModel or None if no models were available
        """
        cable_model_list = [mv.cable_model for mv in self.measurement_version if hasattr(mv, "cable_model")]

        if len(cable_model_list) == 0:
            return None

        model_list = [cm.model for cm in cable_model_list]
        lower_bound = max(cm.lower_bound for cm in cable_model_list)
        upper_bound = min(cm.upper_bound for cm in cable_model_list)

        avg_poly1d = self.average_poly1d(model_list)
        avg_cable_model = CableModel(
            model=avg_poly1d,
            quality=None,  # Assuming quality needs to be calculated or is not required
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        if plot:
            self.plot_average_poly1d(cable_models=cable_model_list, avg_cable_model=avg_cable_model)

        if inplace:
            self.cable_model = avg_cable_model

        if auto_commit:
            self.get_database_manager().commit()

        logger.debug(f"Successfully merged cable models with lower_bound={lower_bound}, upper_bound={upper_bound}.")

        return avg_cable_model

    @classmethod
    def plot_multi_cable_models(cls, system_versions: List['SystemVersion'], force_max: float = 40):
        """
        Compare the avg_cable_model polynomials of multiple SystemVersion instances.

        :param system_versions: List of SystemVersion instances
        :return: The matplotlib Figure object
        """
        logger.debug("Starting plot_multi_cable_models")

        if not system_versions:
            logger.error("No SystemVersion instances provided for comparison.")
            return None

        cable_models = [(sv.cable_model, sv) for sv in system_versions if sv.cable_model is not None]

        if not cable_models:
            logger.warning("No valid avg_cable_model polynomials found in the provided SystemVersion instances.")
            return None

        max_degree = max(cm.model.order for cm, _ in cable_models)

        try:
            fig = plot_cable_models(cable_models, max_degree, force_max)
            logger.info("Successfully created the polynomial comparison plot.")

            # Convert list of integers to a string suitable for a filename
            system_version_ids = [sv.system_version_id for sv in system_versions if sv.cable_model is not None]
            ids_str = "_".join(map(str, system_version_ids))
            filename = f'system_version_ids_{ids_str}'
            subdir = "plot_multi_cable_models"
            plot_manager = cls.get_plot_manager()
            plot_manager.save_plot(fig, filename, subdir)
            logger.debug(f"Plot saved successfully to {subdir}/{filename}")

        except RuntimeError as e:
            logger.error(e)
            return None

        return fig

    @classmethod
    def plot_difference_in_cable_models(cls, system_version_1: 'SystemVersion', system_version_2: 'SystemVersion',
                                        force_max: float = 40, skale_ax2_ref: int = 400):
        """
        Compare the avg_cable_model polynomials of exactly two SystemVersion instances by plotting their differences.

        :param system_version_1: First SystemVersion instance
        :param system_version_2: Second SystemVersion instance
        :param force_max
        :param skale_ax2_ref
        :return: The matplotlib Figure object
        """
        logger.debug("Starting plot_difference_in_cable_models")

        if not system_version_1 or not system_version_2:
            logger.error("Both SystemVersion instances are required for this comparison.")
            return None

        max_degree = max(system_version_1.cable_model.model.order, system_version_2.cable_model.model.order)

        try:
            fig = plot_cable_models_difference(system_version_1, system_version_2, max_degree, force_max, skale_ax2_ref)
            logger.info("Successfully created the polynomial difference plot.")

            filename = f'system_version_ids_{system_version_1.system_version_id}_vs_{system_version_2.system_version_id}'
            subdir = "plot_difference_in_cable_models"
            plot_manager = cls.get_plot_manager()
            plot_manager.save_plot(fig, filename, subdir)
            logger.debug(f"Plot saved successfully to {subdir}/{filename}")

        except RuntimeError as e:
            logger.error(e)
            return None

        return fig

    @staticmethod
    def average_poly1d(poly_list: List[np.poly1d]) -> np.poly1d:
        """
        Calculate the average of a list of numpy.poly1d objects.

        :param poly_list: List of numpy.poly1d objects
        :return: Averaged numpy.poly1d object
        """
        max_degree = max(poly.order for poly in poly_list)
        coeff_sum = np.zeros(max_degree + 1)

        for poly in poly_list:
            extended_coeffs = np.zeros(max_degree + 1)
            extended_coeffs[-(poly.order + 1):] = poly.coeffs
            coeff_sum += extended_coeffs

        avg_coeffs = coeff_sum / len(poly_list)
        average_poly1d = np.poly1d(avg_coeffs)

        logger.debug(f"Successfully calculated average polynomial with coefficients: {average_poly1d.coeffs}.")

        return average_poly1d

    @property
    def load_ztv(self) -> int:
        try:
            load_ztv = self.system.cable.load_ztv * 10
            return load_ztv
        except Exception as e:
            raise ValueError(f"Error getting load_ztv: {e}")

    def get_e_by_f_poly1d(self, force: float) -> Optional[float]:
        if self.cable_model is None:
            logger.warning(
                f"{self} - Poly1d-Model not available. Call Method '{self.__class__.__name__}.merge_cable_models' first.")
            return np.NAN

        cable_model = self.cable_model
        model: np.poly1d = cable_model.model

        if force < cable_model.lower_bound or force > cable_model.upper_bound:
            logger.info(
                f"Force '{force} kN is not in model bounds '{cable_model.lower_bound} - {cable_model.upper_bound}', returning NAN")
            return np.NAN

        return float(model(force))

    def get_e_at_load_ztv(self, percent: float) -> float:
        f = self.load_ztv * (percent / 100)
        e = self.get_e_by_f_poly1d(f)
        if e is None:
            logger.error(f"{self} Elongation could not be calculated.")
            return np.NAN
        return e

    def get_e_at_pre_tension_load(self) -> float:
        e = self.get_e_by_f_poly1d(self.pre_tension_load)
        if e is None:
            logger.error(f"{self} Elongation at pre-tension load could not be calculated.")
            return np.NAN
        return e

    def get_e_at_f_max(self) -> float:
        e = self.get_e_by_f_poly1d(self.f_max)
        if e is None:
            logger.error(f"{self} Elongation at max force could not be calculated.")
            return np.NAN
        return e

    def plot_average_poly1d(self, cable_models: List[CableModel], avg_cable_model: CableModel):

        fig = plt_average_poly1d(self.system.system, cable_models, avg_cable_model)

        plot_manager = self.get_plot_manager()
        filename = f'system_id_{self.system.system_id}_svn_{self.system_version_name}'
        subdir = f"{self.system_version_name}/average_poly1d"
        plot_manager.save_plot(fig, filename, subdir)
