from typing import Callable
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

import pickle
from sqlalchemy import LargeBinary

from ..common_imports.imports_classes import *

from ..utils.polyfit import polyfit_with_np, add_zeros, add_min_values
from ..plotting.plot_measurement_version import plt_filter_data, plt_extrema, plt_select_data, plt_f_vs_e
from ..plotting.plot_measurement_version import plotly_filter_data, plotly_f_vs_e
from ..plotting.plot_polyfit import plt_polyfit

logger = get_logger(__name__)

from .data_tcc import DataTCC
from .cable_model import CableModel


class MeasurementVersion(BaseClass):
    """
    This class represents a measurement in the system.
    """
    __tablename__ = 'MeasurementVersion'

    measurement_version_id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    measurement_version_name = Column(String)
    measurement_id = Column(Integer, ForeignKey('Measurement.measurement_id', onupdate='CASCADE'), nullable=False)
    cable_model_id = Column(Integer, ForeignKey('CableModel.cable_model_id', onupdate='CASCADE'), nullable=True)
    system_version_id = Column(Integer, ForeignKey('SystemVersion.system_version_id', onupdate='CASCADE'),
                               nullable=True)

    # Ändern Sie den Datentyp von 'model' in LargeBinary für die Serialisierung
    _params_dict = Column('param_dict', LargeBinary)
    _extrema_tuple = Column('extrema', LargeBinary)
    selection_mode = Column(String)
    selection_until = Column(String)
    null_offset = Column(Float)
    filter_flag = Column(Boolean)

    data_tcc = relationship("DataTCC", backref="measurement_version", lazy="joined", uselist=False,
                            cascade='all, delete-orphan')

    def __init__(self, measurement_version_id=None, measurement_version_name=None, measurement_id=None,
                 data_tcc_id: int = None, selection_mode: str = "default", selection_until: str = "end"):
        super().__init__()
        self.measurement_version_id = measurement_version_id
        self.measurement_version_name = measurement_version_name
        self.measurement_id = measurement_id
        self.data_tcc_id = data_tcc_id

        self.selection_mode = selection_mode
        self.selection_until = selection_until
        self.null_offset = None
        self.filter_flag = False
        self._extrema_tuple = None
        self._params_dict = None

    def __str__(self):
        return (f"{self.__class__.__name__}: measurement_version_id: {self.measurement_version_id}, "
                f"measurement_version_name: {self.measurement_version_name}, measurement_id: {self.measurement_id}")

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the MeasurementVersion instance for debugging.

        :return: A detailed string representation of the MeasurementVersion instance.
        """
        return f"<MeasurementVersion(measurement_version_id={self.measurement_version_id}, measurement_version_name={self.measurement_version_name}, measurement_id={self.measurement_id})>"

    @classmethod
    def create_from_csv(cls, csv_filepath: str, measurement_id: int, measurement_version_name: str = None) \
            -> Optional['MeasurementVersion']:
        """
        Loads TCC Data from a CSV file.

        :param csv_filepath: Path to the CSV file.
        :param measurement_id: ID of the measurement to which the data belongs.
        :param measurement_version_name: Version Name of the data.
        :return: MeasurementVersion object.
        """
        config = cls.get_config()
        obj = cls(measurement_id=measurement_id, measurement_version_name=measurement_version_name)

        data_directory = config.data_directory
        folder: str = config.DataTCC.data_directory
        filename: str = cls.get_data_manager().get_new_filename(measurement_id,
                                                                prefix=f"tcc_{measurement_version_name}",
                                                                file_extension="feather")

        data_filepath = str(data_directory / folder / filename)

        data_tcc = DataTCC.create_from_csv(csv_filepath, data_filepath, obj.measurement_version_id)

        obj.data_tcc = data_tcc

        session = cls.get_database_manager().session
        session.add(obj)
        logger.info(f"Created new '{obj}'")
        return obj

    def update_from_csv(self, csv_filepath: str) -> Optional['MeasurementVersion']:
        self.data_tcc = self.data_tcc.update_from_csv(csv_filepath)
        logger.info(f"{self} - Updated from csv")
        return self

    def filter(self,
               window_x: int = None, method_x: Optional[str] = None,
               window_f: int = None, method_f: Optional[str] = None,
               plot: bool = True,
               inplace: bool = True, auto_commit: bool = True) -> Optional[pd.DataFrame]:
        """
        Applies rolling window filters ('mean' or 'median') to the dataset columns 'x' (way) and 'f' (force).

        Parameters:
        - window_x (int): Rolling window size for 'x' column. Defaults to 11.
        - method_x (str, optional): Method to apply to 'x' column ('mean' or 'median'). Uses config default if None.
        - window_f (int, optional): Rolling window size for 'f' column. Uses config default if None.
        - method_f (str, optional): Method to apply to 'f' column ('mean' or 'median'). Uses config default if None.
        - inplace (bool): If True, updates data in place. Defaults to True.
        - auto_commit (bool): If True, commits changes to the database. Defaults to True.

        Returns:
        pd.DataFrame: The filtered DataFrame.
        """
        data = self.data.copy()
        try:
            data['raw_x'] = data['x'].astype(np.float32)
            data['raw_f'] = data['f'].astype(np.float32)

            # Define filter methods as a dictionary mapping to functions
            valid_methods = {
                "mean": lambda df, window: df.rolling(window=window, center=True).mean(),
                "median": lambda df, window: df.rolling(window=window, center=True).median()
            }

            method_x = method_x or self.config.filter_method_x
            method_f = method_f or self.config.filter_method_f
            window_x = window_x or self.config.filter_window_x
            window_f = window_f or self.config.filter_window_f

            # Validate the methods
            if method_x not in valid_methods or method_f not in valid_methods:
                logger.error("Unsupported filter method. Only 'mean' and 'median' are supported.")
                raise ValueError("Unsupported filter method. Only 'mean' and 'median' are supported.")

            # Apply the filter methods
            if method_x:
                data['x'] = valid_methods[method_x](data['x'], window_x)
                logger.debug(f"Applied {method_x} filter on 'x' with window {window_x}.")
            if method_f:
                data['f'] = valid_methods[method_f](data['f'], window_f)
                logger.debug(f"Applied {method_f} filter on 'f' with window {window_f}.")

            # Drop missing values
            data.dropna(inplace=True)
            logger.debug(f"{self} filter successfully")

        except Exception as e:
            logger.error(f"{self} filter Error, e: {e}")
            return None

        if plot:
            self.plot_filter_data(data)

        if inplace:
            self.data = data
            self.filter_flag = True

        if auto_commit:
            self.get_database_manager().commit()

        return data

    def null_offset_f(self, mean_first_n: int = 9, inplace: bool = True, auto_commit: bool = True) -> Optional[
        Tuple[pd.DataFrame, float]]:
        data = self.data.copy()
        try:

            f_mean_first_10 = data['f'][:mean_first_n].mean()  # to reduce force to null at beginn
            data['f'] = data['f'] - f_mean_first_10

            logger.debug(f"{self} - Nulloffset successfully")

        except Exception as e:
            logger.error(f"{self} - Nulloffset Error, e: {e}")
            return None

        if inplace:
            self.data = data
            self.null_offset = f_mean_first_10

        if auto_commit:
            self.get_database_manager().commit()

        return data, f_mean_first_10

    def calc_features(self, set_raw: bool = False, inplace: bool = True, auto_commit: bool = True) -> Optional[
        pd.DataFrame]:
        data = self.data.copy()

        try:
            data['l'] = abs(self.left_head - data['x'])  # distance between anchors
            l_min = data['l'].min()  # shortest distance
            data['d'] = abs(data['l'] - l_min)  # difference between distance at beginn and time x
            data['e'] = abs(data['d'] / l_min * 100)

            if set_raw:
                data['raw_l'] = data['l'].astype(np.float32)
                data['raw_d'] = data['d'].astype(np.float32)
                data['raw_e'] = data['e'].astype(np.float32)

            logger.debug(f"{self} - Features successfully")

        except Exception as e:
            logger.error(f"{self} - Features Error, e: {e}")
            return None

        if inplace:
            self.data = data

        if auto_commit:
            self.get_database_manager().commit()

        return data

    def calc_extrema(self, plot=True, inplace=True) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.Timestamp]:
        data = self.data["d"]

        peak_height = self.measurement.peak_height or self.config.peak_height
        peak_prominence = self.measurement.peak_prominence or self.config.peak_prominence
        peak_distance = self.measurement.peak_distance or self.config.peak_distance
        peak_width = self.measurement.peak_width or self.config.peak_width
        valley_height = self.measurement.valley_height or self.config.valley_height
        valley_prominence = self.measurement.valley_prominence or self.config.valley_prominence
        valley_distance = self.measurement.valley_distance or self.config.valley_distance
        valley_width = self.measurement.valley_width or self.config.valley_width

        peaks, _ = find_peaks(np.array(data),
                              height=peak_height,
                              prominence=peak_prominence,
                              distance=peak_distance,
                              width=peak_width)

        peaks_index = data.index[peaks]

        if len(peaks_index) != 3:
            logger.warning(
                f"mv_id: '{self.measurement_version_id}' found '{len(peaks_index)}' Peaks: {peaks_index.values}")
        else:
            logger.debug(f"mv_id: '{self.measurement_version_id}' found exactly 3 Peaks: {peaks_index.values}")

        valleys, _ = find_peaks(np.array(data * -1),
                                height=valley_height,
                                prominence=valley_prominence,
                                distance=valley_distance,
                                width=valley_width)

        valleys_index = data.index[valleys]

        if len(valleys_index) != 3:
            logger.warning(
                f"mv_id: '{self.measurement_version_id}' found '{len(valleys_index)}' Valleys: {valleys_index.values}")
        else:
            logger.debug(f"mv_id: '{self.measurement_version_id}' found exactly 3 Valleys: {valleys_index.values}")

        pre_tension_load = self.pre_tension_load * 1.1
        data = self.data["f"]
        data_above_threshold = data[data > pre_tension_load]

        peak_height = self.config.first_drop_peak_height
        peak_prominence = self.config.first_drop_peak_prominence
        peak_distance = self.config.first_drop_peak_distance
        peak_width = self.config.first_drop_peak_width

        peaks_first_drop, _ = find_peaks(np.array(data_above_threshold),
                                         height=peak_height,
                                         prominence=peak_prominence,
                                         distance=peak_distance,
                                         width=peak_width)

        peaks_first_drop_index = data_above_threshold.index[peaks_first_drop]
        logger.debug(f"peaks_first_drop_index: {peaks_first_drop_index.values}")
        first_drop_index = peaks_first_drop_index[0]
        logger.debug(f"Found first_drop_index: {first_drop_index}")

        if plot:
            self.plot_extrema(self.data, peaks_index, valleys_index, first_drop_index)

        if inplace:
            self.extrema = peaks_index, valleys_index, first_drop_index

        return peaks_index, valleys_index, first_drop_index

    @property
    def extrema(self):
        if hasattr(self, '_extrema') and self._extrema_tuple is not None:
            return pickle.loads(self._extrema_tuple)
        else:
            return self.calc_extrema(plot=False, inplace=True)

    @extrema.setter
    def extrema(self, extrema: Tuple):
        """Serialisiert das Pipeline-Objekt für die Speicherung in der Datenbank."""
        self._extrema_tuple = pickle.dumps(extrema)

    def select_inc_preload(self, selection_until: str = "f_max", plot: bool = True, inplace: bool = False,
                           auto_commit: bool = False):
        data = self.data.copy()

        if selection_until not in self.config.valide_selection_until:
            raise ValueError

        if self.measurement.execute == "dont_select":
            logger.warning(f"No selecting for '{self}' as 'execute' is set to 'dont_select'.")
            data_select = data

        else:
            peaks, valleys, first_drop = self.extrema

            # Wähle Daten von v3_datetime bis zum Ende aus
            data_select = data.loc[valleys[2]:].copy()

        if selection_until == "f_max":
            data_select = self.select_until_f_max(data_select)

        if selection_until == "first_drop":
            data_select = self.select_until_first_drop(data_select)

        if plot:
            self.plot_select(data, data_select, subdir="inc_preload")

        if inplace:
            self.data = data_select
            self.selection_mode = "inc_preload"
            self.selection_until = selection_until

        if auto_commit:
            self.get_database_manager().commit()
        return data_select

    def select_exc_preload(self, selection_until: str = "f_max", close_gap: bool = True,
                           plot: bool = True, inplace: bool = False, auto_commit: bool = False):
        data = self.data.copy()

        if selection_until not in self.config.valide_selection_until:
            raise ValueError

        if self.measurement.execute == "dont_select":
            logger.warning(f"No selecting for '{self}' as 'execute' is set to 'dont_select'.")
            data_select = data

        else:
            peaks, valleys, first_drop = self.extrema

            data_0_to_p1 = data.loc[:peaks[0]]

            # Hole den Wert von "f" an der Position p1
            p1_f_value = data.loc[peaks[0]]["f"]

            # Definiere die Bedingung für Daten, die später als v3 sind UND deren "f" größer ist als das von p1
            data_v3_and_higher_f = data.loc[valleys[2]:].query("f > @p1_f_value")

            if close_gap:
                x_max = data_0_to_p1["x"].max()
                x_min = data_v3_and_higher_f["x"].min()
                x_delta = x_min - x_max
                data_v3_and_higher_f["x"] = data_v3_and_higher_f["x"] - x_delta
                logger.info(f"x_max: {x_max}, x_min: {x_min}, x_delta: {x_delta}")

            # Füge die Daten zusammen: Von Anfang bis p1 UND die Daten, die die Bedingung erfüllen
            data_select = pd.concat([data_0_to_p1, data_v3_and_higher_f])

        if selection_until == "f_max":
            data_select = self.select_until_f_max(data_select)

        if selection_until == "first_drop":
            data_select = self.select_until_first_drop(data_select)

        if plot:
            self.plot_select(data, data_select, subdir="exc_preload")

        if inplace:
            self.data = data_select
            self.selection_mode = "exc_preload"
            self.selection_until = selection_until

        if auto_commit:
            self.get_database_manager().commit()
        return data_select

    @staticmethod
    def select_until_f_max(data: pd.DataFrame) -> pd.DataFrame:
        # Erhalte den DateTime-Indexwert für das Maximum von 'f'
        max_f_datetime = data['f'].idxmax()

        # Wähle Daten zwischen v3_datetime und max_f_datetime aus
        select_data = data.loc[:max_f_datetime]
        return select_data

    def select_until_first_drop(self, data: pd.DataFrame) -> pd.DataFrame:
        # Erhalte den DateTime-Indexwert für das Maximum von 'f'
        _, _, first_drop = self.extrema

        # Wähle Daten zwischen v3_datetime und max_f_datetime aus
        select_data = data.loc[:first_drop]

        return select_data

    def fit_model(self, add_zeros_n: int = 1000, degree_min: int = 3, degree_max: int = 15,
                  desired_quality_r2: float = .9995,
                  plot: bool = True, inplace: bool = True, auto_commit: bool = True) -> CableModel:
        try:
            data = self.data
            x = data["f"].copy()
            y = data["e"].copy()

            if add_zeros_n > 0:
                x = add_min_values(x, add_zeros_n)
                y = add_zeros(y, add_zeros_n)

            cable_model = polyfit_with_np(
                x, y, degree_min, degree_max, desired_quality_r2)

            logger.info(f"{self} - Successfully fitted CableModel: {cable_model}")

            if plot:
                fig = plt_polyfit(x, y, cable_model)

                plot_manager = self.get_plot_manager()
                filename = f'm_id_{self.measurement_id}_mv_id_{self.measurement_version_id}'
                subdir = f"{self.measurement_version_name}/polyfit"
                plot_manager.save_plot(fig, filename, subdir)

            if inplace:
                self.cable_model = cable_model

            if auto_commit:
                self.get_database_manager().commit()

            return cable_model
        except Exception as e:
            logger.error(f"{self} - Fitting of poly1d failed, e: {e}")

    def get_e_by_f(self, force: float, method: str = "interp1d") -> Optional[float]:
        """
        Generic method to get elongation (e) for a given force (f) using a specified method name.

        Args:
            force (float): The force for which to calculate the elongation.
            method (str): The name of the method to use ('interp1d' or 'poly1d').

        Returns:
            Optional[float]: The calculated elongation or None if not calculable.
        """
        # Verify that the provided force is a valid number
        if not isinstance(force, (int, float)):
            logger.error(f"Invalid force value '{force}'. Must be a number.")
            return None

        # Dictionary mapping method names to their corresponding functions
        methods: dict[str, Callable[[float], Optional[float]]] = {
            'interp1d': self.get_e_by_f_interp1d,
            'poly1d': self.get_e_by_f_poly1d
        }

        method_func = methods.get(method)

        # Check if the method is valid
        if method_func is None:
            logger.error(f"Invalid method '{method}'. Available methods are 'interp1d' and 'poly1d'.")
            return None

        # Call the selected method and handle any potential errors
        try:
            return method_func(force)
        except Exception as e:
            logger.error(f"Error calculating elongation with method '{method}': {e}")
            return None

    def get_e_by_f_interp1d(self, force: float) -> Optional[float]:
        data = self.data.drop_duplicates(subset='f')
        data = data.sort_values(by='f')

        f = data['f'].values
        e = data['e'].values

        model = interp1d(f, e, kind='linear')

        if force < f.min() or force > f.max():
            logger.info(f"{self} - Force {force} not in the range of the measurement, returning NAN")
            return np.NAN
        return float(model(force))

    def get_e_by_f_poly1d(self, force: float) -> Optional[float]:
        if self.cable_model is None:
            logger.info(
                f"{self} - Poly1d-Model not available. Call Method '{self.__class__.__name__}.fit_model' first.")
            return np.NAN

        cable_model = self.cable_model
        model: np.poly1d = cable_model.model

        if force < cable_model.lower_bound or force > cable_model.upper_bound:
            logger.warning(
                f"Force '{force} kN is not in model bounds '{cable_model.lower_bound} - {cable_model.upper_bound}', returning NAN")
            return np.NAN

        return float(model(force))

    def get_e_at_load_ztv(self, percent: float, method: str = 'poly1d') -> float:
        f = self.load_ztv * (percent / 100)
        e = self.get_e_by_f(f, method)
        if e is None:
            logger.error(f"{self} Elongation could not be calculated.")
            return np.NAN
        return e

    def get_e_at_pre_tension_load(self, method: str = 'poly1d') -> float:
        e = self.get_e_by_f(self.pre_tension_load, method)
        if e is None:
            logger.error(f"{self} Elongation at pre-tension load could not be calculated.")
            return np.NAN
        return e

    def get_e_at_f_max(self, method: str = 'poly1d') -> float:
        e = self.get_e_by_f(self.f_max, method)
        if e is None:
            logger.error(f"{self} Elongation at max force could not be calculated.")
            return np.NAN
        return e

    def get_params_dict(self, method: str = "interp1d", inplace: bool = True, auto_commit: bool = True) -> Dict[
        str, any]:
        """
        Generate a dictionary containing parameters using the specified method.

        Args:
            method (str): The method to use ('interp1d' or 'poly1d').
            inplace (bool): Whether to store the dictionary in the instance.
            auto_commit (bool): Whether to commit the changes to the database.

        Returns
        -------
        Dict[str, any]
            A dictionary containing the parameters.

        Raises
        ------
        ValueError
            If the method is 'poly1d' and self.cable_model is not available.
        """
        if method == "poly1d" and self.cable_model is None:
            error_message = f"{self} - No Cable Model Type Poly1D available."
            logger.warning(error_message)
            raise ValueError(error_message)

        params_dict: Dict[str, any] = {
            "measurement_version_id": getattr(self, "measurement_version_id", None),
            "measurement_version_name": getattr(self, "measurement_version_name", None),
            "measurement_id": getattr(self, "measurement_id", None),
            "system_name": getattr(self, "system", None),
            "brand": getattr(self, "brand_short", None),
            "selection_mode": getattr(self, "selection_mode", None),
            "selection_until": getattr(self, "selection_until", None),
            "filter_flag": getattr(self, "filter_flag", None),
            "e_by_f_method": method,
            "null_offset": getattr(self, "null_offset", None),
            "d_min": getattr(self, "d_min", None),
            "d_max": getattr(self, "d_max", None),
            "load_ztv": getattr(self, "load_ztv", None),
            "pre_tension_load": getattr(self, "pre_tension_load", None),
            "e_at_pre_tension_load": self.get_e_at_pre_tension_load(method),
            "f_min": getattr(self, "f_min", None),
            "f_max": getattr(self, "f_max", None),
            "get_e_at_f_max": self.get_e_at_f_max(method),
            "count": getattr(self, "count", None),
            "expansion_insert_count": getattr(self, "expansion_insert_count", None),
            "shock_absorber_count": getattr(self.system, "shock_absorber_count", None),
            "shock_absorber_l_delta": getattr(self, "shock_absorber_l_delta", None),
            "failure_loc": getattr(self, "failure_loc", None),
            "note": getattr(self, "note", None),
        }

        # Add values for e_at_load_ztv
        for percentage in self.config.param_e_at_load_ztv_values:
            params_dict[f"e_at_load_ztv_{percentage}"] = self.get_e_at_load_ztv(percentage, method)

        # Add values for e_by_f
        for force in self.config.param_e_by_f_values:
            params_dict[f"e_by_f_{force}"] = self.get_e_by_f(force, method)

        if inplace:
            self._params_dict = pickle.dumps(params_dict)

        if auto_commit:
            self.get_database_manager().commit()

        return params_dict

    # @staticmethod
    # def _get_readable_param_name_flex(param: str) -> str:
    #     """
    #     Generates a readable long name for a given flexible parameter.
    #
    #     Parameters:
    #     - param (str): The parameter name.
    #
    #     Returns:
    #     - str: The generated long name if the parameter matches known patterns,
    #            else returns the original parameter name.
    #     """
    #     base_names: Dict[str, str] = {
    #         "e_at_load_ztv": "Elongation at {} % Load-ZTV [%]",
    #         "e_by_f": "Elongation at {} kN [%]",
    #     }
    #     parts = param.split("_")
    #     if len(parts) > 1 and parts[-2] in base_names and parts[-1].isdigit():
    #         base_name = "_".join(parts[:-1])
    #         value = parts[-1]
    #         return base_names[base_name].format(value)
    #     return param
    #
    # def get_readable_param_name(self, param: str) -> str:
    #     """
    #     Returns the readable long form of a parameter.
    #
    #     If the parameter does not match any predefined or flexible naming patterns,
    #     logs a warning and returns the original parameter name.
    #
    #     Parameters:
    #     - param (str): The fixed or flexible parameter name.
    #
    #     Returns:
    #     - str: The long form of the parameter.
    #     """
    #     fixed_params: Dict[str, str] = {
    #         "measurement_version_id": "MeasurementVersion ID",
    #         "measurement_version_name": "MeasurementVersion Name",
    #         "measurement_id": "Measurement ID",
    #         "system_name": "System Name",
    #         "f_null_offset": "Force Null-Offset [kN]",
    #         "d_min": "Delta Length min. [mm] (=0)",
    #         "d_max": "Delta Length max. [mm]",
    #         "pre_tension_load": "Pre-Tension-Load [kN]",
    #         "get_e_at_pre_tension_load": "Elongation at Pre-Tension-Load",
    #         "load_ztv": "Load-ZTV/MBL [kN]",
    #         "f_min": "Force min. [kN]",
    #         "f_max": "Force max. [kN]",
    #         "get_e_at_f_max": "Elongation at Force max. [%]",
    #         "n": "n-values (n)"
    #     }
    #
    #     long_name = fixed_params.get(param, None)
    #     if long_name:
    #         return long_name
    #     long_name = self._get_readable_param_name_flex(param)
    #     if long_name == param:
    #         logger.warning(f"Parameter name '{param}' could not be mapped to a long name.")
    #     return long_name

    # Plotting
    def plot_filter_data(self, data: pd.DataFrame = None, p: str = "plt"):
        if p == "plt":
            fig = plt_filter_data(data)
        else:
            fig = plotly_filter_data(data)

        plot_manager = self.get_plot_manager()
        filename = f'm_id_{self.measurement_id}_mv_id_{self.measurement_version_id}'
        subdir = f"{self.measurement_version_name}/filter_{p}"
        plot_manager.save_plot(fig, filename, subdir)

    def plot_extrema(self, data: pd.DataFrame, peaks: pd.DatetimeIndex, valleys: pd.DatetimeIndex,
                     first_drop: pd.DatetimeIndex, p="plt"):
        fig = plt_extrema(data, peaks, valleys, first_drop)
        plot_manager = self.get_plot_manager()
        filename = f'm_id_{self.measurement_id}_mv_id_{self.measurement_version_id}'
        subdir = f"{self.measurement_version_name}/extrema_{p}"
        plot_manager.save_plot(fig, filename, subdir)

    def plot_select(self, data, data_select, p="plt", subdir: str = None):
        fig = plt_select_data(data, data_select)
        plot_manager = self.get_plot_manager()
        filename = f'm_id_{self.measurement_id}_mv_id_{self.measurement_version_id}'
        subdir = f"{self.measurement_version_name}/select_{subdir}_{p}"
        plot_manager.save_plot(fig, filename, subdir)

    def plot_f_vs_e(self, data: pd.DataFrame = None, plot_raw: bool = True, p="plt"):
        data = data or self.data

        if p == "plt":
            fig = plt_f_vs_e(data, plot_raw, self.system_name)
        else:
            fig = plotly_f_vs_e(data, plot_raw, self.system_name)

        plot_manager = self.get_plot_manager()
        filename = f'm_id_{self.measurement_id}_mv_id_{self.measurement_version_id}'
        subdir = f"{self.measurement_version_name}/f_vs_e_{p}"
        plot_manager.save_plot(fig, filename, subdir)

    # Property Attributs, from other classes #
    @property
    def config(self):
        try:
            return self.get_config().MeasurementVersion
        except Exception as e:
            raise ValueError(f"Error getting config: {e}")

    @property
    def system_name(self) -> Optional[str]:
        if self.measurement is None or self.measurement.system is None:
            logger.error("Measurement or System is not available")
            return None
        return getattr(self.measurement.system, "system", None)

    @property
    def left_head(self) -> int:
        try:
            left_head = self.measurement.left_head
            return left_head
        except Exception as e:
            raise ValueError(f"Error getting left_head: {e}")

    @property
    def load_ztv(self) -> int:
        try:
            load_ztv = self.measurement.system.cable.load_ztv * 10
            return load_ztv
        except Exception as e:
            raise ValueError(f"Error getting load_ztv: {e}")

    @property
    def pre_tension_load(self) -> int:
        try:
            pre_tension_load = self.measurement.pre_tension_load
            return pre_tension_load
        except Exception as e:
            raise ValueError(f"Error getting pre_tension_load: {e}")

    @property
    def data(self) -> pd.DataFrame:
        try:
            data = self.data_tcc.data
            return data
        except Exception as e:
            raise ValueError(f"Error getting data: {e}")

    @data.setter
    def data(self, data: pd.DataFrame):
        try:
            self.data_tcc.data = data
        except Exception as e:
            raise ValueError(f"Error setting data: {e}")

    # Property Features #

    @property
    def l_min(self):
        if 'l' in self.data.columns:
            return self.data['l'].min()
        else:
            logger.warning("Column 'l' not found in data. Returning None.")
            return None

    @property
    def l_max(self):
        if 'l' in self.data.columns:
            return self.data['l'].max()
        else:
            logger.warning("Column 'l' not found in data. Returning None.")
            return None

    @property
    def d_min(self):
        if 'd' in self.data.columns:
            return self.data['d'].min()
        else:
            logger.warning("Column 'd' not found in data. Returning None.")
            return None

    @property
    def d_max(self):
        if 'd' in self.data.columns:
            return self.data['d'].max()
        else:
            logger.warning("Column 'd' not found in data. Returning None.")
            return None

    @property
    def e_min(self):
        if 'e' in self.data.columns:
            return self.data['e'].min()
        else:
            logger.warning("Column 'e' not found in data. Returning None.")
            return None

    @property
    def e_max(self):
        if 'e' in self.data.columns:
            return self.data['e'].max()
        else:
            logger.warning("Column 'e' not found in data. Returning None.")
            return None

    @property
    def f_min(self):
        if 'f' in self.data.columns:
            return self.data['f'].min()
        else:
            logger.warning("Column 'f' not found in data. Returning None.")
            return None

    @property
    def f_max(self):
        if 'f' in self.data.columns:
            return self.data['f'].max()
        else:
            logger.warning("Column 'f' not found in data. Returning None.")
            return None

    @property
    def count(self):
        return self.data.shape[0]

    @property
    def brand(self) -> Optional[str]:
        if self.measurement is None or self.measurement.system is None or self.measurement.system.brand is None:
            logger.error("Measurement, System or Brand is not available")
            return None
        return getattr(self.measurement.system.brand, "brand_short", None)

    @property
    def expansion_insert_count(self) -> Optional[int]:
        if self.measurement is None:
            logger.error("Measurement is not available")
            return None
        return getattr(self.measurement, "expansion_insert_count", None)

    @property
    def shock_absorber_count(self) -> Optional[int]:
        if self.measurement is None or self.measurement.system is None:
            logger.error("Measurement or System is not available")
            return None
        return getattr(self.measurement.system, "shock_absorber_count", None)

    @property
    def shock_absorber_l_delta(self) -> Optional[float]:
        try:
            shock_absorber_l1 = getattr(self.measurement, "shock_absorber_l1", None)
            shock_absorber_l2 = getattr(self.measurement, "shock_absorber_l2", None)

            if shock_absorber_l1 is None or shock_absorber_l2 is None:
                return None

            shock_absorber_l_delta = shock_absorber_l2 - shock_absorber_l1
            return shock_absorber_l_delta
        except Exception as e:
            raise ValueError(f"Error getting shock_absorber_l_delta: {e}")

    @property
    def failure_loc(self) -> Optional[str]:
        if self.measurement is None:
            logger.error("Measurement is not available")
            return None
        return getattr(self.measurement, "failure_loc", None)

    @property
    def note(self) -> Optional[str]:
        if self.measurement is None:
            logger.error("Measurement is not available")
            return None
        return getattr(self.measurement, "note", None)
