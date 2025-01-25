from typing import Callable

from sympy import Matrix, MutableDenseMatrix, Symbol, ZeroMatrix, diff

from .base_models import speed_sym
from .constants import PARAMETER_SYMBOLS, R_SYM, positions


def forces_sym(
    R: MutableDenseMatrix,
    parameters: dict[str, Symbol],
    forces: list[Callable[[MutableDenseMatrix, dict[str, Symbol]], MutableDenseMatrix]],
) -> MutableDenseMatrix:
    sum_forces = ZeroMatrix(m=3, n=1)
    for force in forces:
        sum_forces += force(R=R, parameters=parameters)
    return sum_forces


def acceleration_sym(
    R: MutableDenseMatrix,
    parameters: dict[str, Symbol],
    forces: list[Callable[[MutableDenseMatrix, dict[str, Symbol]], MutableDenseMatrix]],
) -> MutableDenseMatrix:
    return Matrix.vstack(speed_sym(R=R), forces_sym(R=R, parameters=parameters, forces=forces))


def variations_equations_per_parameter_sym(
    variations: MutableDenseMatrix, parameters: dict[str, Symbol], parameter: str, i_parameter: int, forces_model: MutableDenseMatrix
) -> MutableDenseMatrix:

    # R derivative with respect to parameters.
    dr_dgamma = variations[6 * i_parameter : 6 * (i_parameter + 1)]
    dr_dot_dgamma = speed_sym(R=dr_dgamma)

    # Differentiates symbolic equation with respect to r, r_dot and parameters.
    df_dgamma = diff(forces_model, parameters[parameter])
    df_dr: MutableDenseMatrix = Matrix.hstack(*(diff(forces_model, coordinate) for coordinate in positions))
    df_dr_0: MutableDenseMatrix = df_dr.row(0)
    df_dr_1: MutableDenseMatrix = df_dr.row(1)
    df_dr_2: MutableDenseMatrix = df_dr.row(2)

    # Leibniz formula d/dt(dr_dot_dgamma) = df_dr * dr_dgamma  + df_dgamma.
    return Matrix.vstack(
        dr_dot_dgamma,
        Matrix([[df_dr_0.dot(dr_dgamma) + df_dgamma[0]], [df_dr_1.dot(dr_dgamma) + df_dgamma[1]], [df_dr_2.dot(dr_dgamma) + df_dgamma[2]]]),
    )


def variations_equations_sym(
    parameter_names: list[str],
    variations: MutableDenseMatrix,
    forces: list[Callable[[MutableDenseMatrix, dict[str, Symbol]], MutableDenseMatrix]],
    parameters: dict[str, Symbol] = PARAMETER_SYMBOLS,
) -> MutableDenseMatrix:
    forces_model = forces_sym(R=R_SYM, parameters=PARAMETER_SYMBOLS, forces=forces)
    return Matrix.vstack(
        *(
            variations_equations_per_parameter_sym(
                variations=variations, parameters=parameters, parameter=parameter, i_parameter=i_parameter, forces_model=forces_model
            )
            for i_parameter, parameter in enumerate(parameter_names)
        )
    )
