from random import uniform
from typing import Callable

from numpy import array, diff, matmul, ndarray, sin, sqrt
from numpy.random import normal
from sympy import Matrix

from .base_models import evaluate_for_parameters, str_dict_to_symbol_dict
from .constants import DEFAULT_H, PARAMETER_SYMBOLS, STATION_PARAMETER_SYMBOLS
from .integrate import integrate, interpolate, theoretical_measurements
from .measurement_models import STATION_ANGLES_MODEL, T, get_station_parameters_without_noise
from .utils import load_base_model, rotation_matrix, save_base_model


def in_view(
    R_T: float, longitude_rotation_matrix: ndarray[float], latitude_rotation_matrix: ndarray[float], R: ndarray[float], sin_min_elevation: float
) -> bool:
    x_prime, y_prime, z_prime = matmul(latitude_rotation_matrix, matmul(longitude_rotation_matrix, R[:3]))
    d_z = z_prime - R_T
    return d_z / sqrt(d_z**2 + x_prime**2 + y_prime**2) > sin_min_elevation


def visibility_dichotomy(get_visibility: Callable, t_i: float, t_j: float, interpolation: Callable, precision: float) -> float:
    t = (t_i + t_j) / 2
    if t_j - t_i < precision:
        return t
    elif get_visibility(t=t, y_i=interpolation(x=t)) == get_visibility(t=t_j, y_i=interpolation(x=t_j)):
        return visibility_dichotomy(get_visibility=get_visibility, t_i=t_i, t_j=t, interpolation=interpolation, precision=precision)
    else:
        return visibility_dichotomy(get_visibility=get_visibility, t_i=t, t_j=t_j, interpolation=interpolation, precision=precision)


def generate_measurements(
    name: str = "measurements",
    h_0: float = DEFAULT_H,
    DR_0: dict[str, float] = {},  # Initial position Delta for orbit model.
    R_0_uncertainty: list[float] = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2],
    parameter_default_values_file_name: str = "parameter_default_values",
    stations_default_values_file_name: str = "station_default_values",
    integration_default_values_file_name: str = "integration_default_values",
) -> None:

    # Gets all default values.
    stations: dict[str, dict[str, float | dict[str, float]]] = load_base_model(name=stations_default_values_file_name)
    parameters: dict[str, float] = load_base_model(name=parameter_default_values_file_name)
    integration_parameters: dict[str, float] = load_base_model(name=integration_default_values_file_name)

    # Generates perturbates initial position.
    for parameter, value in DR_0.items():
        parameters[parameter] += value

    # Integrates model orbit.
    t, y = integrate(
        parameters=parameters,
        altitude_limit=h_0,
        integration_parameters=integration_parameters,
    )

    # Interpolates model orbit.
    interpolation = interpolate(t=t, y=y)

    # Generates model measurements per station.
    all_measurements = {}
    for id, station_parameters in stations.items():

        station_parameters_without_noise = get_station_parameters_without_noise(station_parameters=station_parameters)
        noise_amplitudes: dict[str, float] = station_parameters["noise_amplitudes"]
        n_measurement_types = len(noise_amplitudes)

        # Determines visibility in the station's coordinate system.
        station_angles_dynamic_model = evaluate_for_parameters(
            expression=Matrix(STATION_ANGLES_MODEL),
            symbols_to_values=str_dict_to_symbol_dict(str_dictionary=parameters, symbol_dictionary=PARAMETER_SYMBOLS)
            | str_dict_to_symbol_dict(
                str_dictionary=station_parameters_without_noise,
                symbol_dictionary=STATION_PARAMETER_SYMBOLS,
            ),
            additional_symbols=(T,),
            position_dependent=False,
        )

        def get_visibility(t: float, y_i: float) -> float:
            station_angles: ndarray[float] = station_angles_dynamic_model(T=t)
            min_elevation, cos_latitude, sin_latitude, cos_longitude, sin_longitude = tuple(station_angles.T[0])
            longitude_rotation_matrix = rotation_matrix(u=[0, 0, 1], c=cos_longitude, s=-sin_longitude)  # - lon rotation around Z axis.
            latitude_rotation_matrix = rotation_matrix(u=[0, 1, 0], c=sin_latitude, s=-cos_latitude)  # + (pi/2 - lat) rotation around Y axis.
            return 1 * in_view(
                R_T=parameters["R_T"],
                longitude_rotation_matrix=longitude_rotation_matrix,
                latitude_rotation_matrix=latitude_rotation_matrix,
                R=y_i,
                sin_min_elevation=sin(min_elevation),
            )

        visibility = [get_visibility(t=t_i, y_i=y_i) for t_i, y_i in zip(t, y)]

        # Gets every pass start time and duration by dichotomy.
        visibility_change_times = [
            visibility_dichotomy(
                get_visibility=get_visibility,
                t_i=t_i,
                t_j=t[min(t.index(t_i) + 1, len(t) - 1)],
                interpolation=interpolation,
                precision=station_parameters["measurements_interval"],
            )
            for t_i in [t[i] for i, v in enumerate(array(object=abs(diff(visibility)), dtype=bool)) if v]
        ]

        # Considers visibility at the beginning and end of the orbit.
        if get_visibility(t=t[0], y_i=interpolation(x=t[0])) == 1:
            visibility_change_times = [t[0]] + visibility_change_times
        if get_visibility(t=t[-1], y_i=interpolation(x=t[-1])) == 1:
            visibility_change_times += [t[-1]]

        # Generates measurement times.
        measurement_times = []
        for t_i, t_j in zip(visibility_change_times[::2], visibility_change_times[1::2]):
            pass_duration = t_j - t_i
            t_first_measurement = t_i + uniform(0.0, pass_duration % station_parameters["measurements_interval"])
            measurement_times += [
                t_first_measurement + i * station_parameters["measurements_interval"]
                for i in range(int(pass_duration // station_parameters["measurements_interval"]))
            ]

        # Generates measurements.
        measurements, _ = theoretical_measurements(
            parameters=parameters,
            station_parameters=station_parameters_without_noise,
            measurement_times=measurement_times,
            t=t,
            interpolation=interpolation,
            n_measurement_types=n_measurement_types,
            measurement_types=noise_amplitudes.keys(),
        )

        # Saves in the main measurements dictionary.
        all_measurements[id] = {"measurement_times": measurement_times} | {
            measurement_type: measurement_values
            + normal(loc=0, scale=station_parameters["noise_amplitudes"][measurement_type], size=(len(measurement_times)))
            for measurement_type, measurement_values in measurements.items()
        }

    # Saves in (.JSON) file.
    save_base_model(
        obj={
            "R_0": array(
                object=[parameters["X_0"], parameters["Y_0"], parameters["Z_0"], parameters["X_dot_0"], parameters["Y_dot_0"], parameters["Z_dot_0"]]
            )
            + array(object=[normal(loc=0, scale=uncertainty) for uncertainty in R_0_uncertainty]),
            "stations": all_measurements,
        },
        name=name,
    )
