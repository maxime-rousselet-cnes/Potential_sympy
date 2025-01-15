from numpy import eye, ndarray
from sympy import Matrix, MutableDenseMatrix, Symbol, symbols

from .utils import load_base_model

INF = 1e50
EPSILON = 1e-13

DEFAULT_H = 175e3

EPSILON_ARC_DURATION = 1000.0


positions = ["X", "Y", "Z", "X_dot", "Y_dot", "Z_dot"]
default_parameters: dict[str, float] = load_base_model("parameter_default_values")
parameter_names = default_parameters.keys()
all_station_default_parameters: dict[str, dict[str, float | dict[str, float]]] = load_base_model("station_default_values")
station_parameter_names = set()
for _, station_default_parameters in all_station_default_parameters.items():
    del station_default_parameters["noise_amplitudes"]
    station_parameter_names.update(station_default_parameters.keys())
station_parameter_names = list(station_parameter_names)
station_parameter_names.sort()

POSITION_SYMBOLS: dict[str, Symbol] = {coordinate: symbol for coordinate, symbol in zip(positions, symbols(" ".join(positions)))}
PARAMETER_SYMBOLS: dict[str, Symbol] = {parameter: symbol for parameter, symbol in zip(parameter_names, symbols(" ".join(parameter_names)))}
R_SYM: MutableDenseMatrix = Matrix([symbol for symbol in POSITION_SYMBOLS.values()])
INITIAL_DYNAMICAL_VALUES: dict[str, ndarray] = {coordinate + "_0": row for coordinate, row in zip(positions, eye(N=6))}
station_coordinates = [coordinate + "_station" for coordinate in positions]
R_STATION_SYMBOLS: dict[str, Symbol] = {coordinate: symbol for coordinate, symbol in zip(station_coordinates, symbols(" ".join(station_coordinates)))}
R_STATION_SYM: MutableDenseMatrix = Matrix([symbol for symbol in R_STATION_SYMBOLS.values()])
STATION_PARAMETER_SYMBOLS: dict[str, Symbol] = {
    parameter: symbol for parameter, symbol in zip(station_parameter_names, symbols(" ".join(station_parameter_names)))
}
