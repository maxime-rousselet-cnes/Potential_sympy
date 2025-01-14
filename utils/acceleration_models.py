from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, diff, exp

from .base_models import norm_sym, position_sym, speed_sym
from .constants import PARAMETER_SYMBOLS, R_SYM, positions


def rho_atm_func_sym(R: MutableDenseMatrix, R_T: Symbol, rho_atm_0: Symbol, H_0: Symbol) -> Expr:
    return rho_atm_0 * exp(-(norm_sym(R=R) - R_T) / H_0)


def drag_sym(R: MutableDenseMatrix, parameters: dict[str, Symbol]) -> MutableDenseMatrix:
    rho_atm = rho_atm_func_sym(R=R, R_T=parameters["R_T"], rho_atm_0=parameters["rho_atm_0"], H_0=parameters["H_0"])
    return -0.5 * parameters["C_x"] * parameters["surface_mass_ratio"] * rho_atm * norm_sym(R=speed_sym(R=R)) * speed_sym(R=R)


def uniform_potential_force_sym(R: MutableDenseMatrix, GM: Symbol) -> MutableDenseMatrix:
    return -GM / norm_sym(R=R) ** 3 * position_sym(R=R)


# TODO: Here to code new force models as sympy expressions.


def forces_sym(R: MutableDenseMatrix, parameters: dict[str, Symbol]) -> MutableDenseMatrix:
    forces = uniform_potential_force_sym(R=R, GM=parameters["GM"])
    # TODO: Here to add force models.
    forces += drag_sym(R=R, parameters=parameters)
    return forces


def acceleration_sym(R: MutableDenseMatrix, parameters: dict[str, Symbol]) -> MutableDenseMatrix:
    return Matrix.vstack(speed_sym(R=R), forces_sym(R=R, parameters=parameters))


ACCELERATION_MODEL = acceleration_sym(R=R_SYM, parameters=PARAMETER_SYMBOLS)
FORCES_MODEL = forces_sym(R=R_SYM, parameters=PARAMETER_SYMBOLS)


def variations_equations_per_parameter_sym(
    variations: MutableDenseMatrix, parameters: dict[str, Symbol], parameter: str, i_parameter: int
) -> MutableDenseMatrix:

    # R derivative with respect to parameters.
    dr_dgamma = variations[6 * i_parameter : 6 * (i_parameter + 1)]
    dr_dot_dgamma = speed_sym(R=dr_dgamma)

    # Differentiates symbolic equation with respect to r, r_dot and parameters.
    df_dgamma = diff(FORCES_MODEL, parameters[parameter])
    df_dr: MutableDenseMatrix = Matrix.hstack(*(diff(FORCES_MODEL, coordinate) for coordinate in positions))
    df_dgamma_0: MutableDenseMatrix = df_dr.row(0)
    df_dgamma_1: MutableDenseMatrix = df_dr.row(1)
    df_dgamma_2: MutableDenseMatrix = df_dr.row(2)

    # Leibniz formula d/dt(dr_dot_dgamma) = df_dr * dr_dgamma  + df_dgamma.
    return Matrix.vstack(
        dr_dot_dgamma,
        Matrix(
            [[df_dgamma_0.dot(dr_dgamma) + df_dgamma[0]], [df_dgamma_1.dot(dr_dgamma) + df_dgamma[1]], [df_dgamma_2.dot(dr_dgamma) + df_dgamma[2]]]
        ),
    )


def variations_equations_sym(
    parameter_names: list[str], variations: MutableDenseMatrix, parameters: dict[str, Symbol] = PARAMETER_SYMBOLS
) -> MutableDenseMatrix:
    return Matrix.vstack(
        *(
            variations_equations_per_parameter_sym(variations=variations, parameters=parameters, parameter=parameter, i_parameter=i_parameter)
            for i_parameter, parameter in enumerate(parameter_names)
        )
    )
