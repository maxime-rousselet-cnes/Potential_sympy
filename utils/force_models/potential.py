from sympy import Matrix, MutableDenseMatrix, Symbol, diff
from sympy.functions.special.polynomials import assoc_legendre, chebyshevt, chebyshevu

from ..base_models import norm_sym


def potential_force_sym(R: MutableDenseMatrix, parameters: dict[str, Symbol]) -> MutableDenseMatrix:
    r = norm_sym(R=R)
    sin_colat = (1 - (R[2] / r) ** 2) ** 0.5
    l_max = max([int(parameter.split("_")[1]) for parameter in parameters.keys() if "C" in parameter and len(parameter.split("_")) == 3])
    potential = (
        parameters["GM"]
        / r
        * (
            1
            + sum([(parameters["R_T"] / r) ** l * (assoc_legendre(l, 0, sin_colat) * parameters["C_" + str(l) + "_0"]) for l in range(2, l_max + 1)])
        )
    )  # Zonal coefficients only.
    """
    rho = (R[0] ** 2 + R[1] ** 2) ** 0.5
    sin_lon = R[1] / rho
    cos_lon = R[0] / rho
    potential = (
        parameters["GM"]
        / r
        * (
            1
            + sum(
                [
                    (parameters["R_T"] / r) ** l
                    * (
                        assoc_legendre(l, 0, sin_colat) * parameters["C_" + str(l) + "_0"]
                        + sum(
                            [
                                assoc_legendre(l, m, sin_colat)
                                * (
                                    chebyshevt(m, cos_lon) * parameters["_".join(("C", str(l), str(m)))]
                                    + sin_lon * chebyshevu(m - 1, cos_lon) * parameters["_".join(("S", str(l), str(m)))]
                                )
                                for m in range(1, l_max + 1)
                            ]
                        )
                    )
                    for l in range(2, l_max + 1)
                ]
            )
        )
    )
    """
    return Matrix([[diff(potential, R[0])], [diff(potential, R[1])], [diff(potential, R[2])]])
