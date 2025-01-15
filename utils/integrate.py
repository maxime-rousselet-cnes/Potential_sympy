from itertools import product
from typing import Callable, Optional

from numpy import array, concatenate, dot, ndarray, transpose, zeros
from scipy.integrate import RK45
from scipy.interpolate import interp1d
from sympy import Expr, Matrix, symbols

from .acceleration_models import ACCELERATION_MODEL, variations_equations_sym
from .base_models import evaluate_for_parameters, str_dict_to_symbol_dict
from .constants import EPSILON, INF, INITIAL_DYNAMICAL_VALUES, PARAMETER_SYMBOLS, POSITION_SYMBOLS
from .measurement_models import STATION_PARAMETER_SYMBOLS, STATION_POSITIONS_MODEL_GENERATOR, dQ_dr_model_generation, measurement_models


def norm(R: ndarray[float]) -> float:
    return (sum(R[:3].flatten() ** 2)) ** 0.5


def integrate(
    parameters: dict[str, float],
    integration_parameters: dict[str, float],
    altitude_limit: Optional[float] = None,
    parameter_names: list[str] = [],
    acceleration_model: Expr = ACCELERATION_MODEL,
) -> tuple[list[float], list[ndarray[float]]]:

    symbols_to_values = str_dict_to_symbol_dict(str_dictionary=parameters, symbol_dictionary=PARAMETER_SYMBOLS)

    # Defines the acceleration equation.
    acceleration = evaluate_for_parameters(expression=acceleration_model, symbols_to_values=symbols_to_values)

    if parameter_names != []:
        # Defines variation equations to use for parameters of interest.
        dr_dgamma = symbols(
            ["_".join(("d" + position_symbol, "d" + parameter)) for parameter, position_symbol in product(parameter_names, POSITION_SYMBOLS.keys())]
        )
        variations_equations = evaluate_for_parameters(
            expression=variations_equations_sym(variations=Matrix(dr_dgamma), parameter_names=parameter_names),
            symbols_to_values=symbols_to_values,
            additional_symbols=dr_dgamma,
        )
    else:
        # Handles the simple orbit model propagation case with no inversion for parameters.
        variations_equations = lambda _: ((()))

    # Function to propagate.
    lambda_propagation = lambda _, R: (
        acceleration(*transpose(R[:6])[0])
        if parameter_names == []
        else concatenate(
            (
                acceleration(*transpose(R[:6])[0]),
                variations_equations(*R.flatten().tolist()),
            )
        )
    )

    # Initial partial derivatives are either 0 (by independence), or 1 (for initial position parameters).
    dr_d_gamma_0 = (
        zeros(shape=(0))
        if parameter_names == []
        else concatenate(
            [[0.0] * 6 if parameter not in INITIAL_DYNAMICAL_VALUES.keys() else INITIAL_DYNAMICAL_VALUES[parameter] for parameter in parameter_names]
        )
    )

    # Initial vector to integrate, corresponding to initial position, initial speed, and eventually initial dr_dgamma and initial dr_dot_dgamma.
    y0 = concatenate(
        (
            [
                parameters["X_0"],
                parameters["Y_0"],
                parameters["Z_0"],
                parameters["X_dot_0"],
                parameters["Y_dot_0"],
                parameters["Z_dot_0"],
            ],
            dr_d_gamma_0,
        ),
    )

    # Defines the integration method.
    rk45_integrator = RK45(
        fun=lambda_propagation,
        t0=0.0,
        y0=y0,
        t_bound=integration_parameters["arc_duration"],
        max_step=integration_parameters["max_step"],
        rtol=[EPSILON] * 3 + [INF] * (3 + len(dr_d_gamma_0)),
        atol=[integration_parameters["dR_tol_max"]] * 3 + [INF] * (3 + len(dr_d_gamma_0)),
        vectorized=True,
    )
    t = []
    y = []

    # Integrates for the arc duration or until the altitude limit is reached.
    while rk45_integrator.status == "running":
        t.append(rk45_integrator.t)
        y.append(rk45_integrator.y)
        if (altitude_limit is not None) and (norm(R=y[-1]) < parameters["R_T"] + altitude_limit):
            break
        rk45_integrator.step()

    # Cowell times and integrated vector containing position, speed and eventually dr_dgamma and dr_dot_dgamma.
    return t, y


def interpolate(t: list[float], y: list[ndarray[float]]) -> Callable:
    return interp1d(x=t, y=y, kind="cubic", axis=0)


def theoretical_measurements(
    parameters: dict[str, float],
    station_parameters: dict[str, float],
    measurement_times: list[float],
    t: list[float],
    interpolation: Callable,
    n_measurement_types: int,
    measurement_types: list[str],
) -> tuple[dict[str, ndarray[float]], ndarray[float]]:

    # Interpolates the orbit at the measurement times.
    y_on_measurements: ndarray = interpolation(x=measurement_times)
    n_times = len(y_on_measurements)

    # Generates theoretical station position with respect to time.
    station_position_model = STATION_POSITIONS_MODEL_GENERATOR(measurement_times=measurement_times)
    station_position_sym = evaluate_for_parameters(
        expression=station_position_model,
        symbols_to_values=str_dict_to_symbol_dict(str_dictionary=parameters, symbol_dictionary=PARAMETER_SYMBOLS),
        additional_symbols=STATION_PARAMETER_SYMBOLS.values(),
        position_dependent=False,
    )
    station_positions = station_position_sym(*tuple(station_parameters.values()))

    # Generates theoretical measurements.
    measurements = {
        measurement_type: array(
            object=[
                measurement_models[measurement_type](R=R_on_measurements, R_station=R_station)
                for R_station, R_on_measurements in zip(station_positions, y_on_measurements)
            ]
        )
        for measurement_type in measurement_types
    }

    # Gets dQ_j/dr_j.
    dQ_dr_model = dQ_dr_model_generation(measurement_types=measurement_types)
    dQ_j_dr_j = array(
        object=[dQ_dr_model(R=R_on_measurements, R_station=R_station) for R_station, R_on_measurements in zip(station_positions, y_on_measurements)]
    )  # Axes: (time, distance/doppler, coordinate).

    # Formats dr_j/dgamma_i. Axes: (time, coordinate, parameter).
    dr_j_dgamma_i: ndarray = y_on_measurements[:, 6:].reshape((n_times, -1, 6)).transpose((0, 2, 1))

    # Leibniz formula.
    A_mat = zeros(shape=(n_measurement_types * n_times, dr_j_dgamma_i.shape[2]))
    for i_t in range(n_times):
        for i, dQ_j_dr_j_line in enumerate(dQ_j_dr_j[i_t]):
            A_mat[n_measurement_types * i_t + i] = dot(a=dQ_j_dr_j_line, b=dr_j_dgamma_i[i_t])

    return measurements, A_mat
