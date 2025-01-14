from typing import Callable

from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, evaluate, lambdify, sqrt

from .constants import POSITION_SYMBOLS


def speed_sym(R: MutableDenseMatrix) -> MutableDenseMatrix:
    return Matrix(R[3:6])


def position_sym(R: MutableDenseMatrix) -> MutableDenseMatrix:
    return Matrix(R[:3])


def norm_sym(R: MutableDenseMatrix) -> Expr:
    return sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)


def str_dict_to_symbol_dict(str_dictionary: dict[str, float], symbol_dictionary: dict[str, Symbol]) -> dict[Symbol, float]:
    return {symbol_dictionary[parameter]: value for parameter, value in str_dictionary.items()}


def evaluate_for_parameters(
    expression: Expr, symbols_to_values: dict[Symbol, float], additional_symbols: tuple = (), position_dependent: bool = True
) -> Callable:
    with evaluate(False):
        simplified_model = expression if symbols_to_values == {} else expression.xreplace(rule=symbols_to_values)
        return lambdify(args=list(POSITION_SYMBOLS.values() if position_dependent else ()) + list(additional_symbols), expr=simplified_model)
