from pathlib import Path
from typing import Optional

from matplotlib.pyplot import figure, legend, plot, show, subplots, title, xlabel, ylabel
from numpy import array, concatenate, ndarray, zeros
from seaborn import heatmap

from .constants import default_parameters
from .integrate import norm
from .utils import get_parameters, load_base_model


def plot_orbit_iterations(case_name: str = "case_no_free", figsize: tuple[float, float] = (10, 10)) -> None:
    iteration_folders = list(Path(".").joinpath("examples").joinpath(case_name).joinpath("results").glob(pattern="*"))
    iteration_folders.sort(key=lambda path: int(path.name))
    figure(figsize=figsize)
    for iteration_folder in iteration_folders:
        orbit = load_base_model(name="orbit", path=iteration_folder)
        plot(
            array(object=orbit["t"]) / 3600,
            [(norm(R=r) - default_parameters["R_T"]) / 1e3 for r in array(object=orbit["y"])],
            label="Iteration " + iteration_folder.name,
        )
    legend()
    title("Orbit convergence")
    ylabel("Mean altitude (km)")
    xlabel("Time (h)")
    show()


def plot_orbit(case_name: str = "case_no_free", figsize: tuple[float, float] = (10, 6), iteration: Optional[int] = None, linewidth: int = 1) -> None:
    path = Path(".").joinpath("examples").joinpath(case_name)
    if iteration:
        path = path.joinpath("results").joinpath(str(iteration))
    name = "orbit" if iteration else "model_orbit"
    model_orbit: dict[str, ndarray] = load_base_model(name=name, path=path)
    fig = figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    y = array(object=model_orbit["y"])
    plot(y[:, 0] / 1e3, y[:, 1] / 1e3, y[:, 2] / 1e3, linewidth=linewidth)
    title("Orbit model")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    show()


def plot_errors(case_name: str = "case_no_free", figsize: tuple[float, float] = (10, 6), linewidth: int = 1) -> None:
    folders = list(Path(".").joinpath("examples").joinpath(case_name).joinpath("results").glob(pattern="*"))
    folders.sort(key=lambda path: int(path.name))
    _, base_parameters, _, _, _, _ = get_parameters(case_name=case_name, restitution=False)
    _, parameters, _, parameter_names, _, _ = get_parameters(case_name=case_name)
    parameter_names += [
        "X_0",
        "Y_0",
        "Z_0",
        "X_dot_0",
        "Y_dot_0",
        "Z_dot_0",
    ]
    for parameter in parameters.keys():
        parameters[parameter] = [parameters[parameter]]
    for folder in folders:
        parameter_values: dict[str, ndarray] = load_base_model(name="parameter_values", path=folder)
        for parameter in parameter_names:
            res = abs(parameter_values[parameter] - base_parameters[parameter])
            parameters[parameter] += [res]
    _, ax = subplots(1, 1, figsize=figsize)
    for parameter in parameter_names:
        ax.scatter(range(len(parameters[parameter])), parameters[parameter], label=parameter, linewidth=linewidth)
    ax.set_yscale("log")
    title("Parameters convergence")
    xlabel("Iterations")
    ylabel("Absolute error on parameters")
    legend()
    show()


def plot_correlations(case_name: str = "case_no_free", figsize: tuple[float, float] = (10, 6), iterate: bool = False) -> None:
    folders = list(Path(".").joinpath("examples").joinpath(case_name).joinpath("results").glob(pattern="*"))
    folders.sort(key=lambda path: int(path.name))
    _, _, _, parameter_names, _, stations = get_parameters(case_name=case_name)
    parameter_names += [
        "X_0",
        "Y_0",
        "Z_0",
        "X_dot_0",
        "Y_dot_0",
        "Z_dot_0",
    ] + (
        []
        if len(stations) == 0
        else list(
            concatenate(
                [
                    ["_".join((id, parameter)) for parameter in list(station_free_parameters.keys())]
                    for id, station_free_parameters in stations.items()
                ]
            )
        )
    )
    for folder in folders if iterate else [folders[-1]]:
        correlation_matrix = array(object=load_base_model(name="correlation_matrix", path=folder))
        normalized_correlation_matrix = zeros(correlation_matrix.shape)
        for i in range(len(parameter_names)):
            for j in range(len(parameter_names)):
                normalized_correlation_matrix[i, j] = correlation_matrix[i, j] / (correlation_matrix[i, i] * correlation_matrix[j, j]) ** 0.5
        figure(figsize=figsize)
        heatmap(data=normalized_correlation_matrix, xticklabels=parameter_names, yticklabels=parameter_names)
        title("Iteration " + folder.name)
        xlabel("")
        ylabel("")
        show()
