from multiprocessing import Pool
from os import path, remove, rmdir, walk
from os.path import exists
from pathlib import Path
from typing import Callable, Optional

from numpy import array, concatenate, expand_dims, matmul, ndarray, repeat, zeros
from scipy.linalg import cholesky, inv, solve_triangular

from .constants import EPSILON_ARC_DURATION, INF
from .integrate import integrate, interpolate, theoretical_measurements
from .measurement_models import get_station_dyamic_parameters
from .utils import extend_parameters, get_parameters, load_base_model, save_base_model, update_stations


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
    parameter_names: list[str],
    parameters: dict[str, float],
    station_parameters: dict[str, float | dict[str, float]],
    station_free_parameters: dict[str, float],
    station_measurements: dict[str, list[float]],
    interpolation: Callable,
) -> dict[str, ndarray]:

    station_dyamic_parameters = get_station_dyamic_parameters(station_parameters=station_parameters)

    # Gets dQ_j/dgamma_i via Leibniz formula.
    measurements, A_dynamic_matrix, A_station_matrix = theoretical_measurements(
        parameters=parameters,
        station_parameters=station_parameters,
        station_dynamic_parameters=station_dyamic_parameters,
        measurement_times=station_measurements["measurement_times"],
        interpolation=interpolation,
        parameter_names=parameter_names,
        station_parameter_names=list(station_free_parameters.keys()),
    )

    # Builds B as Delta_Q_j. Current_residuals.
    B_matrix = measurements_to_matrix(measurements=station_measurements) - measurements_to_matrix(measurements=measurements)
    noise_amplitudes: dict = station_parameters["static_parameters"]["noise_amplitudes"]

    return {"A_dynamic": A_dynamic_matrix, "A_station": A_station_matrix, "B": B_matrix, "sigma": array(object=list(noise_amplitudes.values()))}


def orbit_restitution(
    case_name: Optional[str] = None,
) -> None:

    if case_name is None:
        case_name = "default"

    # Gets all default/initial values and measurements.
    measurement_data: dict = load_base_model(name="measurements", path=Path("examples").joinpath(case_name))
    all_station_measurements: dict[str, dict[str, list[float]]] = measurement_data["stations"]
    stations, parameters, integration_parameters, parameter_names, _, station_free_parameters = get_parameters(case_name=case_name)

    # Does not constrain unused stations.
    for id, station_measurements in all_station_measurements.items():
        if (
            len(station_measurements["measurement_times"]) < integration_parameters["min_measurements_per_station"]
            and id in station_free_parameters.keys()
        ):
            del station_free_parameters[id]

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
    parameters, parameter_names = extend_parameters(parameters=parameters, parameter_names=parameter_names, R_0=measurement_data["R_0"])

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
                [
                    (
                        parameter_names,
                        parameters,
                        stations[id],
                        {} if id not in station_free_parameters.keys() else station_free_parameters[id],
                        all_station_measurements[id],
                        interpolation,
                    )
                    for id in stations.keys()
                ],
            )

        # Cumulates.
        for id, station_parameters in station_free_parameters.items():
            for i_station, current_id in enumerate(stations.keys()):
                all_matrices[i_station]["A_dynamic"] = concatenate(
                    (
                        all_matrices[i_station]["A_dynamic"],
                        (
                            zeros(shape=(len(all_matrices[i_station]["A_dynamic"]), len(station_parameters)))
                            if current_id != id
                            else all_matrices[i_station]["A_station"]
                        ),
                    ),
                    axis=1,
                )
        A_matrix: ndarray = concatenate(list(matrices["A_dynamic"] for matrices in all_matrices))
        B_matrix: ndarray = concatenate(list(matrices["B"] for matrices in all_matrices))
        W_matrix = expand_dims(
            a=concatenate([repeat(a=matrices["sigma"], repeats=len(matrices["A_dynamic"]) // len(matrices["sigma"])) for matrices in all_matrices]),
            axis=1,
        ) ** (-2.0)

        # Stop criterion on parameter convergence.
        convergence_criterion = abs(SE(B=previous_residuals) - SE(B=B_matrix)) / SE(B=previous_residuals)
        print("Iteration", iteration, "- Convergence criterion:", convergence_criterion)
        if convergence_criterion < integration_parameters["convergence_epsilon"]:
            break
        else:
            previous_residuals = B_matrix

        # Builds semi-definite positive system to solve.
        N_matrix: ndarray = matmul(A_matrix.T, W_matrix * A_matrix)
        S_matrix: ndarray = matmul(A_matrix.T, W_matrix * B_matrix)

        # Solves normal equations via Cholesky.
        L_matrix: ndarray = cholesky(N_matrix, lower=True)
        Z_matrix: ndarray = solve_triangular(a=L_matrix, b=S_matrix, lower=True)
        x: ndarray = solve_triangular(a=L_matrix.T, b=Z_matrix)

        # Correlation_matrix.
        residual_matrix: ndarray = matmul(A_matrix, x) - B_matrix
        x = x.flatten()
        correlation_matrix: ndarray = matmul(
            residual_matrix.T,
            W_matrix * residual_matrix,
        ) * inv(a=N_matrix)

        # Updates parameters.
        i_parameter = 0
        for parameter in parameter_names:
            parameters[parameter] += x[i_parameter]
            i_parameter += 1

        # Updates station parameters.
        for id, station in station_free_parameters.items():
            for parameter in station.keys():
                station_free_parameters[id][parameter] += x[i_parameter]
                i_parameter += 1
        stations = update_stations(stations=stations, new_stations=station_free_parameters)

        # With iteration number, saves the orbit, normal equations as A, B and parameter updated values.
        save_path = result_folder.joinpath(str(iteration))
        save_base_model(obj={"t": t, "y": y}, name="orbit", path=save_path)
        save_base_model(obj=A_matrix, name="A", path=save_path)
        save_base_model(obj=B_matrix, name="B", path=save_path)
        save_base_model(obj=station_free_parameters, name="station_free_parameter_values", path=save_path)
        save_base_model(obj=correlation_matrix, name="correlation_matrix", path=save_path)
        save_base_model(
            obj={parameter: value for parameter, value in parameters.items() if parameter in parameter_names}, name="parameter_values", path=save_path
        )
