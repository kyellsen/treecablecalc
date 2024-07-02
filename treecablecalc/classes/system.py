from ..common_imports.imports_classes import *

from .cable import Cable
from .sling import Sling
from .expansion_insert import ExpansionInsert
from .shock_absorber import ShockAbsorber
from .anti_abrasion_hose import AntiAbrasionHose
from .material_add import MaterialAdd
from .elongation_properties import ElongationProperties
from .system_version import SystemVersion


logger = get_logger(__name__)


class System(BaseClass):
    """
    This class represents a system.
    """
    __tablename__ = 'System'

    system_id = Column(Integer, primary_key=True, autoincrement=True)
    system = Column(String)
    brand_id = Column(Integer, ForeignKey('Brand.brand_id'), nullable=True)
    cable_id = Column(Integer, ForeignKey('Cable.cable_id'), nullable=True)
    sling_id = Column(Integer, ForeignKey('Sling.sling_id'), nullable=True)
    sling_count = Column(Integer)
    expansion_insert_id = Column(Integer, ForeignKey('ExpansionInsert.expansion_insert_id'), nullable=True)
    shock_absorber_id = Column(Integer, ForeignKey('ShockAbsorber.shock_absorber_id'), nullable=True)
    shock_absorber_count = Column(Integer)
    anti_abrasion_hose_id = Column(Integer, ForeignKey('AntiAbrasionHose.anti_abrasion_hose_id'), nullable=True)
    material_add_id = Column(Integer, ForeignKey('MaterialAdd.material_add_id'), nullable=True)
    elongation_properties_id = Column(Integer, ForeignKey('ElongationProperties.elongation_properties_id'),
                                      nullable=True)

    measurement = relationship("Measurement", backref="system", lazy="joined", cascade='all, delete-orphan',
                               order_by='Measurement.measurement_id')
    system_version = relationship(SystemVersion, backref="system", lazy="joined", cascade='all, delete-orphan',
                                  order_by='SystemVersion.system_version_id')

    def __init__(self, system_id: int = None, system: str = None, sling_count: int = None,
                 shock_absorber_count: int = None):
        super().__init__()
        self.system_id = system_id
        self.system = system
        self.sling_count = sling_count
        self.shock_absorber_count = shock_absorber_count

    def __str__(self) -> str:
        """
        Represents the System instance as a string.

        :return: A string representation of the System instance.
        """
        return f"System(system_id={self.system_id}, system={self.system})"

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the System instance for debugging.

        :return: A detailed string representation of the System instance.
        """
        return f"<System(system_id={self.system_id}, system={self.system})>"

    @classmethod
    def get_system_id_by_name(cls, system_name: str) -> Optional[int]:
        """
        Retrieve the system_id for a given system_name.

        :param system_name: Name of the system to retrieve the system_id for
        :return: The system_id if found, otherwise None
        """
        session = cls.get_database_manager().session

        system = session.query(System).filter_by(system=system_name).one_or_none()

        if system is None:
            logger.error(f"System with name '{system_name}' not found.")
            return None

        logger.debug(f"Successfully retrieved system_id {system.system_id} for system_name '{system_name}'.")
        return system.system_id

    def filter_measurements(self, filter_query: str) -> List:
        """
        Filters the measurements based on the provided SQL filter query and ensures they belong to the current system.

        Args:
            filter_query (str): SQL filter query string for measurement attributes.

        Returns:
            List[Measurement]: Filtered list of measurements.
        """
        # Convert the SQL filter query to a Python function
        def matches_filter(measurement, filter_query):
            try:
                return eval(filter_query, {"__builtins__": None}, measurement.__dict__)
            except Exception as e:
                logger.error(f"Error evaluating filter query '{filter_query}' on measurement '{measurement}': {e}")
                return False

        filtered_measurements = [m for m in self.measurement if matches_filter(m, filter_query)]
        return filtered_measurements

    def get_measurement_versions(self, version_name: str = None, measurements: List = None) -> List:
        """
        Get measurement versions by matching the version name.

        Args:
            version_name (str, optional): The name of the measurement version to match.
            measurements (List, optional): List of measurements to process. If None, uses all measurements.

        Returns:
            List: List of matching measurement versions.
        """
        if measurements is None:
            measurements = self.measurement
        mv_list = []
        for m in measurements:
            mv = next((mv for mv in m.measurement_version if mv.measurement_version_name == version_name), None)
            if mv:
                mv_list.append(mv)
        return mv_list

    def create_system_version(self, system_version_name: str, measurement_version_name: str,
                              filter_query: str, auto_commit: bool = True) -> SystemVersion:
        """
        Create a new system version and link it with relevant measurement versions.

        Args:
            system_version_name (str): The name of the new system version.
            measurement_version_name (str): The name of the measurement version to filter.
            filter_query (str): SQL filter query string for measurement attributes.
            auto_commit (bool, optional): If True, commits changes to the database. Defaults to True.

        Returns:
            SystemVersion: The newly created system version.
        """
        # Create new system version
        new_system_version = SystemVersion(system_version_name=system_version_name, system_id=self.system_id, filter_query=filter_query)

        # Filter measurements using the provided SQL query
        measurements = self.filter_measurements(filter_query)

        mv_list = self.get_measurement_versions(version_name=measurement_version_name, measurements=measurements)

        # Link measurement versions to new system version
        for mv in mv_list:
            mv.system_version_id = new_system_version.system_version_id
            new_system_version.measurement_version.append(mv)

        if not mv_list:
            logger.info(f"{self}: No MeasurementVersion available, mv_list empty.")

        # Add the new system version to the system
        self.system_version.append(new_system_version)

        if auto_commit:
            self.get_database_manager().commit()

        logger.info(f"New system_version '{new_system_version}' created: '{new_system_version}'")
        return new_system_version

    def create_system_version_plus_measurement_versions(self, version_name: str,
                                                        filter_query: str, auto_commit: bool = True,
                                                        selection_mode: str = "inc_preload",
                                                        selection_until: str = "first_drop",
                                                        update_existing: bool = False, filter_data: bool = True,
                                                        null_offset_f: bool = True, fit_model: bool = True,
                                                        plot_filter: bool = False, plot_extrema: bool = False,
                                                        plot_selection: bool = False, plot_f_vs_e: bool = False,
                                                        plot_fit_model: bool = False) -> Optional[SystemVersion]:
        """
        Create a new system version and link it with relevant measurement versions, creating them if necessary.

        Args:
            version_name (str): The name of the new system version and of the measurement versions.
            filter_query (str): SQL filter query string for measurement attributes.
            auto_commit (bool, optional): If True, commits changes to the database. Defaults to True.
            selection_mode (str, optional): The mode for selecting data.
            selection_until (str, optional): The point until which selection is done.
            update_existing (bool, optional): Whether to update existing data. Defaults to False.
            filter_data (bool, optional): Whether to apply filtering on the data. Defaults to True.
            null_offset_f (bool, optional): Whether to nullify offset forces. Defaults to True.
            fit_model (bool, optional): Whether to fit a model to data. Defaults to True.
            plot_filter (bool, optional): Whether to plot the filter process. Defaults to False.
            plot_extrema (bool, optional): Whether to plot the extrema calculation process. Defaults to False.
            plot_selection (bool, optional): Whether to plot the selection process. Defaults to False.
            plot_f_vs_e (bool, optional): Whether to plot force vs. extension. Defaults to False.
            plot_fit_model (bool, optional): Whether to plot the fit model. Defaults to False.

        Returns:
            Optional[SystemVersion]: The newly created system version or None if no measurements are available.
        """
        # Filter measurements using the provided SQL query
        measurements = self.filter_measurements(filter_query)

        # Check if there are any measurements to process
        if not measurements:
            logger.info(f"{self}: No Measurements available, system version not created.")
            return None

        # Create new system version
        new_system_version = SystemVersion(system_version_name=version_name, system_id=self.system_id, filter_query=filter_query)

        # Add the new system version to the session first
        session = self.get_database_manager().session
        session.add(new_system_version)
        session.flush()  # Ensure the system version gets an ID

        # Create and link measurement versions
        mv_list = []
        for measurement in measurements:
            mv = measurement.load_with_features(
                selection_mode=selection_mode,
                selection_until=selection_until,
                measurement_version_name=version_name,
                update_existing=update_existing,
                filter_data=filter_data,
                null_offset_f=null_offset_f,
                fit_model=fit_model,
                plot_filter=plot_filter,
                plot_extrema=plot_extrema,
                plot_selection=plot_selection,
                plot_f_vs_e=plot_f_vs_e,
                plot_fit_model=plot_fit_model
            )
            if mv:
                mv.system_version_id = new_system_version.system_version_id
                session.add(mv)  # Add the measurement version to the session
                new_system_version.measurement_version.append(mv)
                mv_list.append(mv)

        if not mv_list:
            logger.warning(f"{self}: No MeasurementVersion created, mv_list empty.")
            session.rollback()
            return None

        # Add the new system version to the system
        self.system_version.append(new_system_version)

        if auto_commit:
            self.get_database_manager().commit()

        logger.info(f"New system_version '{version_name}' created: '{new_system_version}'")
        return new_system_version

