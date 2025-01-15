from pathlib import Path

from matplotlib.pyplot import figure, legend, plot, show
from numpy import array

from .constants import default_parameters
from .integrate import norm
from .utils import load_base_model


def plot_orbit_iterations(result_file_name: str = "results", figsize: tuple[float, float] = (10, 10)) -> None:
    iteration_folders = list(Path(result_file_name).glob(pattern="*"))
    iteration_folders.sort()
    figure(figsize=figsize)
    for iteration_folder in iteration_folders:
        t = array(object=load_base_model(path=iteration_folder, name="t"))
        orbit = array(object=load_base_model(path=iteration_folder, name="orbit"))
        plot(t / 3600, [(norm(R=R) - default_parameters["R_T"]) / 1e3 for R in orbit], label=iteration_folder.name)
    legend()
    show()
