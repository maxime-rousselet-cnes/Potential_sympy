from typing import Callable

from sympy import MutableDenseMatrix, Symbol

from .drag import drag_sym
from .potential import potential_force_sym
from .radiation_pressure import solar_radiation_pressure_sym

FORCES: list[Callable[[MutableDenseMatrix, dict[str, Symbol]], MutableDenseMatrix]] = [potential_force_sym, drag_sym, solar_radiation_pressure_sym]
