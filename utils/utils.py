from json import JSONEncoder, dump, load
from pathlib import Path
from typing import Any, Optional

from numpy import array, cos, ndarray, pi, sin, sqrt
from pydantic import BaseModel

positions = ["X", "Y", "Z", "X_dot", "Y_dot", "Z_dot"]


def norm(R: ndarray[float]) -> float:
    return (sum(R[:3].flatten() ** 2)) ** 0.5


class JSONSerialize(JSONEncoder):
    """
    Handmade JSON encoder that correctly encodes special structures.
    """

    def default(self, obj: Any):
        if type(obj) is ndarray:
            return obj.tolist()
        elif isinstance(obj, BaseModel):
            return obj.__dict__
        else:
            JSONEncoder().default(obj)


def save_base_model(obj: Any, name: str, path: Path = Path(".")):
    """
    Saves a JSON serializable type.
    """
    # Eventually considers subpath.
    while len(name.split("/")) > 1:
        path = path.joinpath(name.split("/")[0])
        name = "/".join(name.split("/")[1:])
    # May create the directory.
    path.mkdir(exist_ok=True, parents=True)
    # Saves the object.
    with open(path.joinpath(name + ".json"), "w") as file:
        dump(obj, fp=file, cls=JSONSerialize)


def load_base_model(
    name: str,
    path: Path = Path("."),
    base_model_type: Optional[Any] = None,
) -> Any:
    """
    Loads a JSON serializable type.
    """
    filepath = path.joinpath(name + ("" if ".json" in name else ".json"))
    if filepath.exists():
        with open(filepath, "r") as file:
            loaded_content = load(fp=file)
        return loaded_content if not base_model_type else base_model_type(**loaded_content)
    else:
        return {}


def rotation_matrix(u: ndarray[float], c: float, s: float) -> ndarray[float]:
    return array(
        object=[
            [u[0] ** 2 * (1 - c) + c, u[0] * u[1] * (1 - c) - u[2] * s, u[0] * u[2] * (1 - c) + u[1] * s],
            [u[0] * u[1] * (1 - c) + u[2] * s, u[1] ** 2 * (1 - c) + c, u[1] * u[2] * (1 - c) - u[0] * s],
            [u[0] * u[2] * (1 - c) - u[1] * s, u[1] * u[2] * (1 - c) + u[0] * s, u[2] ** 2 * (1 - c) + c],
        ]
    )


def update_parameters(parameters: dict[str, float], potential: list[list[list[float]]]) -> dict[str, float]:

    # Update parameters with potential field parameters.
    for phase_name, phase in zip(["C", "S"], potential):
        for degree, values in enumerate(phase):
            if degree == 0:
                continue
            for order, value in enumerate(values[: degree + 1]):
                if phase_name == "S" and order == 0:
                    continue
                parameters["_".join((phase_name, str(degree), str(order)))] = value

    # Orbital to cartesian.
    if "a_0" in parameters.keys():
        parameters["X_0"], parameters["Y_0"], parameters["Z_0"], parameters["X_dot_0"], parameters["Y_dot_0"], parameters["Z_dot_0"] = (
            orbital_to_cartesian(
                a=parameters["a_0"],
                e=parameters["e_0"],
                i=parameters["i_0"],
                Omega_RAAN=parameters["Omega_RAAN_0"],
                omega=parameters["omega_0"],
                E=parameters["E_0"],
                GM=parameters["GM"],
            )
        )
        for element in ["a_0", "e_0", "i_0", "Omega_RAAN_0", "omega_0", "E_0"]:
            del parameters[element]

    return parameters


def get_parameters(case_name: Optional[str] = None, restitution: bool = True, path: Path = Path(".").joinpath("examples")) -> tuple[
    dict[str, dict[str, float | dict[str, float]]],
    dict[str, float],
    dict[str, float],
    Optional[list[str]],
    Optional[dict[str, float]],
    Optional[dict[str, dict[str, float]]],
]:

    # Load all parameter files.
    if case_name is None:
        case_name = "default"
    case_path = path.joinpath(case_name)
    if case_name != "default":
        case_path = case_path.joinpath("initial_values" if restitution else "measurements_generation")
    stations: dict[str, dict[str, float | dict]] = load_base_model(name="stations", path=case_path)
    parameters: dict = load_base_model(name="parameters", path=case_path)
    potential = load_base_model(name="potential", path=case_path)
    integration_parameters = load_base_model(name="integration", path=case_path)
    initial_position_uncertainty = load_base_model(name="initial_position_uncertainty", path=case_path)

    # Updates parameters with potential field parameters.
    parameters = update_parameters(parameters=parameters, potential=potential)

    # Returns default case.
    if case_name == "default":
        return stations, parameters, integration_parameters, None, initial_position_uncertainty, None

    # Updates default values.
    default_stations, default_parameters, default_integration_parameters, _, default_initial_position_uncertainty, _ = get_parameters(
        case_name="default", path=path
    )
    for parameter in list(parameters.keys()):
        if default_parameters[parameter] == parameters[parameter]:
            del parameters[parameter]

    return (
        update_stations(stations=default_stations, new_stations=stations),
        default_parameters | parameters,
        default_integration_parameters | integration_parameters,
        list(parameters.keys()),
        default_initial_position_uncertainty | initial_position_uncertainty,
        stations,
    )


def update_stations(
    stations: dict[str, dict[str, float | dict[str, float]]], new_stations: dict[str, dict[str, float]]
) -> dict[str, dict[str, float | dict[str, float]]]:
    return stations | {id: stations[id] | station for id, station in new_stations.items()}


def extend_parameters(
    parameters: dict[str, float],
    parameter_names: list[str],
    R_0: list[float],
) -> tuple[dict[str, float], list[float], dict[str, float], list[str]]:
    parameters = parameters | {position + "_0": initial_position for position, initial_position in zip(positions, R_0)}
    parameter_names += [position + "_0" for position in positions]
    return (
        parameters,
        parameter_names,
    )


def orbital_to_cartesian(
    a: float, e: float, i: float, Omega_RAAN: float, omega: float, E: float, GM: float
) -> tuple[float, float, float, float, float, float]:
    Omega_RAAN = pi / 180 * Omega_RAAN
    omega = pi / 180 * omega
    i = pi / 180 * i
    E = pi / 180 * E
    p = array(
        object=[
            cos(Omega_RAAN) * cos(omega) - cos(i) * sin(Omega_RAAN) * sin(omega),
            sin(Omega_RAAN) * cos(omega) + cos(i) * cos(Omega_RAAN) * sin(omega),
            sin(i) * sin(omega),
        ]
    )
    q = array(
        object=[
            -cos(Omega_RAAN) * sin(omega) - cos(i) * sin(Omega_RAAN) * cos(omega),
            -sin(Omega_RAAN) * sin(omega) + cos(i) * cos(Omega_RAAN) * cos(omega),
            sin(i) * cos(omega),
        ]
    )
    sqrt_fact = sqrt(1 - e**2)
    x_p = a * (cos(E) - e)
    y_q = a * sqrt_fact * sin(E)
    position = x_p * p + y_q * q
    n = sqrt(GM / a**3)
    r = norm(R=position)
    dE_dt = n * a / r
    x_dot_p = -a * dE_dt * sin(E)
    y_dot_q = a * dE_dt * sqrt_fact * cos(E)
    speed = x_dot_p * p + y_dot_q * q
    return position[0], position[1], position[2], speed[0], speed[1], speed[2]
