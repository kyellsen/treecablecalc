from typing import Tuple, Optional, Any
from kj_logger import get_logger, LogManager, LOG_MANAGER

from .config import Config
from kj_core import DataManager
from kj_core import DatabaseManager
from kj_core import PlotManager

from .config import Config

from .classes.data_tcc import DataTCC

CONFIG = None
DATA_MANAGER = None
DATABASE_MANAGER = None
PLOT_MANAGER = None


def setup(working_directory: Optional[str] = None, log_level="info", safe_logs_to_file=True) -> tuple[
    Config, LogManager, DataManager, DatabaseManager, PlotManager]:
    """
    Set up the treecablecalc package with specific configurations.

    Parameters:
        working_directory (str, optional): Path to the working directory.
        log_level (str, optional): Logging level.
        safe_logs_to_file
    """
    global CONFIG, DATA_MANAGER, DATABASE_MANAGER, PLOT_MANAGER

    LOG_MANAGER.update_config(working_directory, log_level, safe_logs_to_file)

    logger = get_logger(__name__)

    CONFIG = Config(working_directory)

    name = CONFIG.package_name
    name_s = CONFIG.package_name_short

    logger.info(f"{name_s}: Setup {name} package!")
    DATA_MANAGER = DataManager(CONFIG)

    #Listen to changes on Attribut-"data" for all classes of type CoreDataClass
    DATA_MANAGER.register_listeners([DataTCC]) # add DataClasses to list

    DATABASE_MANAGER = DatabaseManager(CONFIG)

    PLOT_MANAGER = PlotManager(CONFIG)

    logger.info(f"{name_s}: {name} setup completed.")

    return CONFIG, LOG_MANAGER, DATA_MANAGER, DATABASE_MANAGER, PLOT_MANAGER

def help():
    """
    Provides detailed guidance on setting up and using the treecablecalc package.

    Setup:
        The setup function initializes the treecablecalc package with user-defined settings.
        It configures logging, data management, database management, plotting, and the operational environment for data analysis.

        Example Usage:
            config, log_manager, data_manager, database_manager, plot_manager = setup('/path/to/directory', 'debug', safe_logs_to_file=True)

    Main Components:
        DataTCC:
            Represents data specific to the Tree Cable Calculation (TCC) experiments, including measurement and metadata.

    Managers:
        DataManager:
            Manages the data flow within the package, including registering listeners for data changes.

        DatabaseManager:
            Handles database interactions, ensuring efficient data storage and retrieval.

        PlotManager:
            Manages the creation and customization of plots for data visualization.

    Further Information:
        For detailed API documentation, usage examples, and more, refer to the package documentation or visit the GitHub repository.
    """
    print(help.__doc__)
