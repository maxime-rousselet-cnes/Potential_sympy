from sympy import Matrix, MutableDenseMatrix, Symbol


def solar_radiation_pressure_sym(R: MutableDenseMatrix, parameters: dict[str, Symbol]) -> MutableDenseMatrix:
    # Constant simple model. No eclipse. No Earth revolution dependency.
    return Matrix([[parameters["surface_mass_ratio"] * parameters["solar_radiation_pressure"] * (1.0 + parameters["reflexivity"])], [0.0], [0.0]])
