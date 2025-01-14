from typing import Callable

from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, cos, diff, pi, sign, sin

from .base_models import evaluate_for_parameters, norm_sym, position_sym, speed_sym
from .constants import PARAMETER_SYMBOLS, POSITION_SYMBOLS, R_STATION_SYM, R_STATION_SYMBOLS, R_SYM, STATION_PARAMETER_SYMBOLS


def get_station_parameters_without_noise(station_parameters: dict[str, float | dict[str, float]]) -> dict[str, float]:
    return {parameter: value for parameter, value in station_parameters.items() if parameter != "noise_amplitudes"}


def station_angles(station_parameters: dict[str, Symbol], parameters: dict[str, Symbol], t: float) -> tuple[Expr, Expr, Expr, Expr, Expr]:
    latitude = pi / 180 * station_parameters["latitude"]
    longitude = pi / 180 * station_parameters["longitude"]
    min_elevation = pi / 180 * station_parameters["min_elevation"]
    cos_latitude = cos(latitude)
    sin_latitude = sin(latitude)
    cos_longitude = cos(longitude)
    sin_longitude = sin(longitude)
    return min_elevation, cos_latitude, sin_latitude, cos_longitude, sin_longitude


T = Symbol("T")
STATION_ANGLES_MODEL = station_angles(station_parameters=STATION_PARAMETER_SYMBOLS, parameters=PARAMETER_SYMBOLS, t=T)


def station_positions_sym(
    parameters: dict[str, Symbol],
    station_parameters: dict[str, Symbol],
    measurement_times: list[float],
) -> list[MutableDenseMatrix]:
    positions = [None] * len(measurement_times)
    for i, t in enumerate(measurement_times):
        _, cos_latitude, sin_latitude, cos_longitude, sin_longitude = station_angles(
            station_parameters=station_parameters, parameters=parameters, t=t
        )
        positions[i] = parameters["R_T"] * Matrix([cos_latitude * cos_longitude, cos_latitude * sin_longitude, sin_latitude, 0, 0, 0])
    return positions


STATION_POSITIONS_MODEL_GENERATOR = lambda measurement_times: Matrix(
    [
        position.transpose()
        for position in station_positions_sym(
            parameters=PARAMETER_SYMBOLS, station_parameters=STATION_PARAMETER_SYMBOLS, measurement_times=measurement_times
        )
    ]
)


def distance_sym(R: MutableDenseMatrix, R_station: MutableDenseMatrix) -> Expr:
    return norm_sym(R=R - R_station)


def doppler_sym(R: MutableDenseMatrix, R_station: MutableDenseMatrix) -> Expr:
    return speed_sym(R=R - R_station).dot(position_sym(R=R - R_station)) / distance_sym(R=R, R_station=R_station)


def station_model_function_to_lambda(model: Expr) -> Callable:
    model_raw = evaluate_for_parameters(expression=model, symbols_to_values={}, additional_symbols=R_STATION_SYMBOLS.values())
    return lambda R, R_station: model_raw(*tuple(list(R[:6]) + list(R_station)))


measurement_syms = {"distance": distance_sym, "doppler": doppler_sym}
MEASUREMENT_MODELS = {measurement_type: sym(R=R_SYM, R_station=R_STATION_SYM) for measurement_type, sym in measurement_syms.items()}
measurement_models = {measurement_type: station_model_function_to_lambda(model=MODEL) for measurement_type, MODEL in MEASUREMENT_MODELS.items()}


def dQ_dr_model_generation(measurement_types: list[str]) -> Callable:
    return station_model_function_to_lambda(
        model=Matrix.vstack(
            *(
                Matrix.hstack(*tuple(Matrix([diff(MEASUREMENT_MODELS[measurement_type], coordinate)]) for coordinate in POSITION_SYMBOLS.values()))
                for measurement_type in measurement_types
            )
        )
    )
