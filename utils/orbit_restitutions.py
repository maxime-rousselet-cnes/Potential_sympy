from multiprocessing import Pool
from os import path, remove, rmdir, walk
from os.path import exists
from pathlib import Path
from typing import Callable, Optional

from numpy import array, concatenate, matmul, ndarray
from scipy.linalg import cholesky, solve_triangular

from .constants import EPSILON_ARC_DURATION, INF, positions
from .integrate import integrate, interpolate, theoretical_measurements
from .measurement_models import get_station_dyamic_parameters
from .utils import get_parameters, load_base_model, save_base_model


def measurements_to_matrix(measurements: dict[str, ndarray]) -> ndarray:
    return array(object=list({key: values for key, values in measurements.items() if key != "measurement_times"}.values())).T.reshape((-1, 1))


def SE(B: ndarray) -> float:
    return sum(B.flatten() ** 2)


def remove_folder(folder_path):
    if exists(folder_path):
        for root, dirs, files in walk(folder_path, topdown=False):
            for name in files:
                remove(path.join(root, name))
            for name in dirs:
                rmdir(path.join(root, name))
        rmdir(folder_path)


def generate_matrices(
    parameters: dict[str, float],
    interpolation: Callable,
    parameter_names: list[str],
    station_measurements: dict[str, list[float]],
    station_parameters: dict[str, float | dict[str, float]],
) -> dict[str, ndarray]:
    station_dyamic_parameters = get_station_dyamic_parameters(station_parameters=station_parameters)

    # Gets dQ_j/dgamma_i via Leibniz formula.
    measurements, A_matrix = theoretical_measurements(
        parameters=parameters,
        station_parameters=station_parameters,
        station_dynamic_parameters=station_dyamic_parameters,
        measurement_times=station_measurements["measurement_times"],
        interpolation=interpolation,
        parameter_names=parameter_names,
    )

    # Builds B as Delta_Q_j. Current_residuals.
    B_matrix = measurements_to_matrix(measurements=station_measurements) - measurements_to_matrix(measurements=measurements)

    return {"A": A_matrix, "B": B_matrix}


def orbit_restitution(
    case_name: Optional[str] = None,
) -> None:

    if case_name is None:
        case_name = "default"

    # Gets all default/initial values and measurements.
    measurement_data: dict = load_base_model(name="measurements", path=Path("examples").joinpath(case_name))
    all_station_measurements: dict[str, dict[str, list[float]]] = measurement_data["stations"]
    stations, parameters, integration_parameters, parameter_names, _ = get_parameters(case_name=case_name)

    # Sets the arc to the minimum duration to get all measurements.
    integration_parameters["arc_duration"] = (
        max(
            [
                measurements["measurement_times"][-1]
                for _, measurements in all_station_measurements.items()
                if "measurement_times" in measurements.keys() and len(measurements["measurement_times"]) != 0
            ]
        )
        + EPSILON_ARC_DURATION
    )
    integration_parameters["altitude_limit"] = -1.0

    # Updates the initial-initial position.
    parameters = parameters | {position + "_0": initial_position for position, initial_position in zip(positions, measurement_data["R_0"])}
    parameter_names += [position + "_0" for position in positions]

    # Clears result folder.
    result_folder = Path("examples").joinpath(case_name).joinpath("results")
    remove_folder(result_folder)

    # Infinite initial residuals so the algorithm does not stop at first iteration.
    previous_residuals = array(object=[INF])

    for iteration in range(integration_parameters["max_iterations"]):

        # Integrate the orbit and the variation equations.
        t, y = integrate(parameters=parameters, integration_parameters=integration_parameters, parameter_names=parameter_names)

        # Interpolates.
        interpolation = interpolate(t=t, y=y, integration_parameters=integration_parameters)

        # Builds normal equations.
        with Pool() as p:
            all_matrices = p.starmap(
                generate_matrices,
                [(parameters, interpolation, parameter_names, all_station_measurements[id], stations[id]) for id in stations.keys()],
            )

        # Cumulates.
        A_matrix: ndarray = concatenate(list(matrices["A"] for matrices in all_matrices))
        B_matrix: ndarray = concatenate(list(matrices["B"] for matrices in all_matrices))

        # Stop criterion on parameter convergence.
        convergence_criterion = abs(SE(B=previous_residuals) - SE(B=B_matrix)) / SE(B=previous_residuals)
        print("Iteration", iteration, "- Convergence criterion:", convergence_criterion)
        if convergence_criterion < integration_parameters["convergence_epsilon"]:
            break
        else:
            previous_residuals = B_matrix

        # Builds semi-definite positive system to solve.
        N_matrix: ndarray = matmul(A_matrix.T, A_matrix)
        S_matrix: ndarray = matmul(A_matrix.T, B_matrix)

        # Solves normal equations via Cholesky.
        L_matrix: ndarray = cholesky(N_matrix, lower=True)
        Z_matrix: ndarray = solve_triangular(a=L_matrix, b=S_matrix, lower=True)
        x: ndarray = solve_triangular(a=L_matrix.T, b=Z_matrix)

        # Update parameters.
        for parameter, delta_gamma in zip(parameter_names, x.flatten()):
            parameters[parameter] += delta_gamma

        # With iteration number, saves the orbit, normal equations as A, B and parameter updated values.
        save_path = result_folder.joinpath(str(iteration))
        save_base_model(obj={"t": t, "y": y}, name="orbit", path=save_path)
        save_base_model(obj=A_matrix, name="A", path=save_path)
        save_base_model(obj=B_matrix, name="B", path=save_path)
        save_base_model(
            obj={parameter: value for parameter, value in parameters.items() if parameter in parameter_names}, name="parameter_values", path=save_path
        )
