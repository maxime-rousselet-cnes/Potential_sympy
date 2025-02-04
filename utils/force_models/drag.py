from sympy import Expr, MutableDenseMatrix, Pow, Symbol

from ..base_models import norm_sym, speed_sym


def rho_atm_func_sym(
    R: MutableDenseMatrix, R_T: Symbol, rho_thermosphere_base: Symbol, thermosphere_ref_altitude: Symbol, thermosphere_scale_factor: Symbol
) -> Expr:
    # Isothermal spherical profile model.
    return rho_thermosphere_base * Pow(10.0, -(norm_sym(R=R) - R_T - thermosphere_ref_altitude) / thermosphere_scale_factor)


def drag_sym(R: MutableDenseMatrix, parameters: dict[str, Symbol]) -> MutableDenseMatrix:
    rho_atm = rho_atm_func_sym(
        R=R,
        R_T=parameters["R_T"],
        rho_thermosphere_base=parameters["rho_thermosphere_base"],
        thermosphere_ref_altitude=parameters["thermosphere_ref_altitude"],
        thermosphere_scale_factor=parameters["thermosphere_scale_factor"],
    )
    return -0.5 * parameters["C_D"] * parameters["surface_mass_ratio"] * rho_atm * norm_sym(R=speed_sym(R=R)) * speed_sym(R=R)
