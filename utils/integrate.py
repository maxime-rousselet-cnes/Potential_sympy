from itertools import product

from numpy import array, concatenate, dot, ndarray, transpose, zeros
from scipy.integrate import RK45
from scipy.interpolate import BSpline, make_interp_spline
from sympy import Matrix, symbols

from .base_models import evaluate_for_parameters, str_dict_to_symbol_dict
from .constants import EPSILON, INF, INITIAL_DYNAMICAL_VALUES, PARAMETER_SYMBOLS, POSITION_SYMBOLS, R_SYM
from .dynamic_models import acceleration_sym, variations_equations_sym
from .force_models import FORCES
from .measurement_models import (
    STATION_PARAMETER_SYMBOLS,
    STATION_POSITIONS_MODEL_GENERATOR,
    dQ_dgamma_model_generation,
    dQ_dr_model_generation,
    measurement_models,
)


def norm(R: ndarray[float]) -> float:
    return (sum(R[:3].flatten() ** 2)) ** 0.5


def integrate(
    parameters: dict[str, float],
    integration_parameters: dict[str, float],
    parameter_names: list[str] = [],
) -> tuple[list[float], list[ndarray[float]]]:

    symbols_to_values = str_dict_to_symbol_dict(str_dictionary=parameters, symbol_dictionary=PARAMETER_SYMBOLS)

    # Defines the acceleration equation.
    acceleration = evaluate_for_parameters(
        expression=acceleration_sym(R=R_SYM, parameters=PARAMETER_SYMBOLS, forces=FORCES), symbols_to_values=symbols_to_values
    )

    if parameter_names != []:
        # Defines variation equations to use for parameters of interest.
        dr_dgamma = symbols(
            ["_".join(("d" + position_symbol, "d" + parameter)) for parameter, position_symbol in product(parameter_names, POSITION_SYMBOLS.keys())]
        )
        variations_equations = evaluate_for_parameters(
            expression=variations_equations_sym(parameter_names=parameter_names, variations=Matrix(dr_dgamma), forces=FORCES),
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
                variations_equations(*transpose(R)[0]),
            )
        )
    )

    # Initial partial derivatives are either 0 (by independence), or 1 (for initial position parameters).
    dr_dgamma_0 = (
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
            dr_dgamma_0,
        ),
    )

    # Defines the integration method.
    rk45_integrator = RK45(
        fun=lambda_propagation,
        t0=0.0,
        y0=y0,
        t_bound=integration_parameters["arc_duration"],
        max_step=integration_parameters["max_step"],
        rtol=[EPSILON] * 3 + [INF] * (3 + len(dr_dgamma_0)),
        atol=[integration_parameters["dR_tol_max"]] * 3 + [INF] * (3 + len(dr_dgamma_0)),
        vectorized=True,
        first_step=EPSILON,
    )
    t = []
    y = []

    # Integrates for the arc duration or until the altitude limit is reached.
    while rk45_integrator.status == "running":
        t.append(rk45_integrator.t)
        y.append(rk45_integrator.y)
        if (integration_parameters["altitude_limit"] >= 0) and (norm(R=y[-1]) < parameters["R_T"] + integration_parameters["altitude_limit"]):
            break
        rk45_integrator.step()

    # Times and integrated vector containing position, speed and eventually dr_dgamma and dr_dot_dgamma.
    return t, y


def interpolate(t: list[float], y: list[ndarray[float]], integration_parameters: dict[str, float]) -> BSpline:
    return make_interp_spline(x=t, y=y, k=integration_parameters["interpolation_order"])


def theoretical_measurements(
    parameters: dict[str, float],
    station_parameters: dict[str, float],
    station_dynamic_parameters: dict[str, float],
    measurement_times: list[float],
    interpolation: BSpline,
    parameter_names: list[str] = [],
) -> tuple[dict[str, ndarray[float]], ndarray[float]]:

    n_parameters = len(parameter_names)
    if len(measurement_times) == 0:
        return {}, zeros(shape=(0, n_parameters))

    noise_amplitudes: dict[str, float] = station_parameters["static_parameters"]["noise_amplitudes"]
    measurement_types = noise_amplitudes.keys()
    n_measurement_types = len(measurement_types)

    # Interpolates the orbit at the measurement times.
    y_on_measurements: ndarray = interpolation(x=measurement_times)
    n_times = len(y_on_measurements)

    # Generates theoretical station position with respect to time.
    symbols_to_values = str_dict_to_symbol_dict(str_dictionary=parameters, symbol_dictionary=PARAMETER_SYMBOLS)
    station_position_model = STATION_POSITIONS_MODEL_GENERATOR(measurement_times=measurement_times)
    station_position_sym = evaluate_for_parameters(
        expression=station_position_model,
        symbols_to_values=symbols_to_values,
        additional_symbols=STATION_PARAMETER_SYMBOLS.values(),
        position_dependent=False,
    )
    station_positions = station_position_sym(*tuple(station_dynamic_parameters.values()))

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
    dQ_j_dr = array(
        object=[dQ_dr_model(R=R_on_measurements, R_station=R_station) for R_station, R_on_measurements in zip(station_positions, y_on_measurements)]
    )  # Axes: (time, distance/doppler, coordinate).

    # Formats dr_j/dgamma_i. Axes: (time, coordinate, parameter).
    dr_dgamma: ndarray = y_on_measurements[:, 6:]
    dr_j_dgamma_i: ndarray

    # Gets dQ_j/dgamma_i.
    dQ_dgamma_model = dQ_dgamma_model_generation(
        parameter_names=parameter_names,
        measurement_types=measurement_types,
        symbols_to_values=symbols_to_values
        | str_dict_to_symbol_dict(str_dictionary=station_dynamic_parameters, symbol_dictionary=STATION_PARAMETER_SYMBOLS),
    )
    dQ_j_dgamma = array(
        object=[dQ_dgamma_model(R=R_on_measurements, t=t_i) for t_i, R_on_measurements in zip(measurement_times, y_on_measurements)]
    )  # Axes: (time, distance/doppler, coordinate).

    # Leibniz formula.
    A_mat = zeros(shape=(n_measurement_types * n_times, n_parameters))
    for i_t, (dr_j_dgamma_i, dQ_j_dgamma_i, dQ_j_dr_j) in enumerate(zip(dr_dgamma, dQ_j_dgamma, dQ_j_dr)):
        for i, (dQ_j_dgamma_i_line, dQ_j_dr_j_line) in enumerate(zip(dQ_j_dgamma_i, dQ_j_dr_j)):
            A_mat[n_measurement_types * i_t + i] = dot(a=dQ_j_dr_j_line, b=dr_j_dgamma_i.reshape((-1, 6)).T) + dQ_j_dgamma_i_line

    return measurements, A_mat
