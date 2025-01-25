from typing import Callable

from numpy import matmul
from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, cos, diff, pi, sin

from .base_models import evaluate_for_parameters, norm_sym, position_sym, speed_sym
from .constants import PARAMETER_SYMBOLS, POSITION_SYMBOLS, R_STATION_SYM, R_STATION_SYMBOLS, R_SYM, STATION_PARAMETER_SYMBOLS
from .utils import rotation_matrix


def get_station_dyamic_parameters(station_parameters: dict[str, float | dict[str, float]]) -> dict[str, float]:
    return {parameter: value for parameter, value in station_parameters.items() if parameter != "static_parameters"}


def station_angles(station_parameters: dict[str, Symbol], parameters: dict[str, Symbol], t: Symbol) -> tuple[Expr, Expr, Expr, Expr, Expr]:
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


def station_position(parameters: dict[str, Symbol], station_parameters: dict[str, Symbol], t: Symbol) -> MutableDenseMatrix:
    _, cos_latitude, sin_latitude, cos_longitude, sin_longitude = station_angles(station_parameters=station_parameters, parameters=parameters, t=t)
    normalized_position_in_solid_Earth_reference_frame = Matrix([cos_latitude * cos_longitude, cos_latitude * sin_longitude, sin_latitude])
    normalized_inertial_speed_in_solid_Earth_reference_frame = Matrix([-cos_latitude * sin_longitude, cos_latitude * cos_longitude, 0.0])
    Earth_rotation_angle = 2 * pi * t * (1.0 + parameters["m_3"]) / parameters["Earth_mean_TOD"]
    Earth_rotation_matrix = rotation_matrix(u=[0, 0, 1], c=cos(Earth_rotation_angle), s=sin(Earth_rotation_angle))
    normalized_position = Matrix(matmul(Earth_rotation_matrix, normalized_position_in_solid_Earth_reference_frame))
    normalized_inertial_speed = Matrix(matmul(Earth_rotation_matrix, normalized_inertial_speed_in_solid_Earth_reference_frame))
    return parameters["R_T"] * Matrix.vstack(normalized_position, normalized_inertial_speed)


def station_positions_sym(
    parameters: dict[str, Symbol],
    station_parameters: dict[str, Symbol],
    measurement_times: list[float],
) -> list[MutableDenseMatrix]:
    return [station_position(parameters=parameters, station_parameters=station_parameters, t=T).xreplace(rule={T: t}) for t in measurement_times]


STATION_POSITION_MODEL = station_position(parameters=PARAMETER_SYMBOLS, station_parameters=STATION_PARAMETER_SYMBOLS, t=T)

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


def dQ_dgamma_model_generation(
    parameter_names: list[str],
    measurement_types: list[str],
    symbols_to_values: dict[Symbol, float],
) -> Callable:
    model = evaluate_for_parameters(
        expression=Matrix.vstack(
            *(
                Matrix.hstack(
                    *tuple(
                        Matrix(
                            [
                                diff(
                                    measurement_syms[measurement_type](R=R_SYM, R_station=STATION_POSITION_MODEL),
                                    symbol,
                                ),
                            ],
                        )
                        for parameter, symbol in PARAMETER_SYMBOLS.items()
                        if parameter in parameter_names
                    )
                )
                for measurement_type in measurement_types
            )
        ),
        symbols_to_values=symbols_to_values,
        additional_symbols=(T,),
    )
    return lambda R, t: model(*tuple(list(R[:6]) + [t]))
