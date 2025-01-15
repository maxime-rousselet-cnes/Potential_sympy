from os import path, remove, rmdir, walk
from os.path import exists

from numpy import array, concatenate, matmul, ndarray
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular

from .constants import EPSILON_ARC_DURATION, INF, positions
from .integrate import integrate, interpolate, theoretical_measurements
from .measurement_models import get_station_parameters_without_noise
from .utils import load_base_model, save_base_model


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


def orbit_restitution(
    measurements_file_name: str = "measurements",
    parameter_default_values_file_name: str = "parameter_default_values",
    parameter_initial_values_file_name: str = "parameter_initial_values",
    station_default_values_file_name: str = "station_default_values",
    integration_default_values_file_name: str = "integration_default_values",
    result_folder_name: str = "results",
) -> None:

    # Gets all default/initial values and measurements.
    measurement_data: dict = load_base_model(name=measurements_file_name)
    station_measurements: dict[str, dict[str, list[float]]] = measurement_data["stations"]
    parameter_default_values: dict[str, float] = load_base_model(name=parameter_default_values_file_name)
    parameter_initial_values: dict[str, float] = load_base_model(name=parameter_initial_values_file_name)
    stations: dict[str, float] = load_base_model(name=station_default_values_file_name)
    integration_parameters = (
        load_base_model(name=integration_default_values_file_name)
        | {"arc_duration": max([measurements["measurement_times"][-1] for _, measurements in station_measurements.items()]) + EPSILON_ARC_DURATION},
    )[0]

    parameter_names = list(parameter_initial_values.keys())
    parameters = (
        parameter_default_values
        | {position + "_0": initial_position for position, initial_position in zip(positions, measurement_data["R_0"])}
        | parameter_initial_values
    )

    # Clears result folder.
    remove_folder(result_folder_name)

    # Infinite initial residuals so the algorithm does not stop at first iteration.
    previous_residuals = array(object=[INF])
    iteration = 0
    while True:

        # Integrate the orbit and the variation equations.
        t, y = integrate(parameters=parameters, integration_parameters=integration_parameters, parameter_names=parameter_names)

        # Interpolates.
        interpolation = interpolate(t=t, y=y)

        # Builds normal equations.
        A_matrices = {}
        B_matrices = {}
        for id, station_parameters in stations.items():

            station_parameters_without_noise = get_station_parameters_without_noise(station_parameters=station_parameters)
            noise_amplitudes: dict[str, float] = station_parameters["noise_amplitudes"]
            measurement_types = noise_amplitudes.keys()
            n_measurement_types = len(measurement_types)

            # Gets dQ_j/dgamma_i via Leibniz formula.
            measurements, A_matrices[id] = theoretical_measurements(
                parameters=parameters,
                station_parameters=station_parameters_without_noise,
                measurement_times=station_measurements[id]["measurement_times"],
                t=t,
                interpolation=interpolation,
                n_measurement_types=n_measurement_types,
                measurement_types=measurement_types,
            )

            # Builds B as Delta_Q_j. Current_residuals.
            B_matrices[id] = measurements_to_matrix(measurements=station_measurements[id]) - measurements_to_matrix(measurements=measurements)

        # Cumulates.
        A_matrix: ndarray = concatenate(list(A_matrices.values()))
        B_matrix: ndarray = concatenate(list(B_matrices.values()))

        # Stop criterion on parameter convergence.
        convergence_criterion = abs(SE(B=previous_residuals) - SE(B=B_matrix)) / SE(B=previous_residuals)
        print("Iteration", iteration, "- Convergence criterion:", convergence_criterion)
        if convergence_criterion < integration_parameters["convergence_epsilon"]:
            break
        else:
            previous_residuals = B_matrix
            iteration += 1

        # Builds semi-definite positive system to solve.
        N_matrix: ndarray = matmul(A_matrix.T, A_matrix)
        S_matrix: ndarray = matmul(A_matrix.T, B_matrix)

        # Solves normal equations via Cholesky.
        L_matrix: ndarray = cholesky(N_matrix)
        Z_matrix: ndarray = solve_triangular(a=L_matrix, b=S_matrix, lower=True)
        x: ndarray = solve_triangular(a=L_matrix.T, b=Z_matrix)

        # Update parameters.
        for parameter, delta_gamma in zip(parameter_names, x.flatten()):
            parameters[parameter] += integration_parameters["learning_rate"] * delta_gamma

        # With iteration number, saves the orbit, normal equations as A, B and parameter updated values.
        base_name = result_folder_name + "/" + str(iteration) + "/"
        save_base_model(obj=t, name=base_name + "t")
        save_base_model(obj=y, name=base_name + "orbit")
        save_base_model(obj=A_matrix, name=base_name + "A")
        save_base_model(obj=B_matrix, name=base_name + "B")
        save_base_model(
            obj={parameter: value for parameter, value in parameters.items() if parameter in parameter_names}, name=base_name + "parameter_values"
        )
