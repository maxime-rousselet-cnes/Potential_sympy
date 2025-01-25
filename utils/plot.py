from pathlib import Path
from typing import Optional

from matplotlib.pyplot import figure, legend, plot, show, subplots, title, xlabel, ylabel
from numpy import array, ndarray

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


def plot_orbit(case_name: str = "case_no_free", figsize: tuple[float, float] = (10, 10), iteration: Optional[int] = None, linewidth: int = 1) -> None:
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


def plot_errors(case_name: str = "case_no_free", figsize: tuple[float, float] = (10, 10), linewidth: int = 1) -> None:
    folders = list(Path(".").joinpath("examples").joinpath(case_name).joinpath("results").glob(pattern="*"))
    _, base_parameters, _, _, _ = get_parameters()
    parameters = {}
    for folder in folders:
        parameter_values: dict[str, ndarray] = load_base_model(name="parameter_values", path=folder)
        for parameter, value in parameter_values.items():
            res = abs(value - base_parameters[parameter])
            if parameter in parameters.keys():
                parameters[parameter] += [res]
            else:
                parameters[parameter] = [res]
    _, ax = subplots(1, 1, figsize=figsize)
    for parameter, values in parameters.items():
        ax.plot(values, label=parameter, linewidth=linewidth)
    ax.set_yscale("log")
    title("Parameters convergence")
    xlabel("Iterations")
    show()
