from json import JSONEncoder, dump, load
from pathlib import Path
from typing import Any, Optional

from numpy import array, ndarray
from pydantic import BaseModel


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
    with open(filepath, "r") as file:
        loaded_content = load(fp=file)
    return loaded_content if not base_model_type else base_model_type(**loaded_content)


def rotation_matrix(u: ndarray[float], c: float, s: float) -> ndarray[float]:
    return array(
        object=[
            [u[0] ** 2 * (1 - c) + c, u[0] * u[1] * (1 - c) - u[2] * s, u[0] * u[2] * (1 - c) + u[1] * s],
            [u[0] * u[1] * (1 - c) + u[2] * s, u[1] ** 2 * (1 - c) + c, u[1] * u[2] * (1 - c) - u[0] * s],
            [u[0] * u[2] * (1 - c) - u[1] * s, u[1] * u[2] * (1 - c) + u[0] * s, u[2] ** 2 * (1 - c) + c],
        ]
    )
