def get_linear_equations(n_vars, max_lag):
    """Return the TimeGraph A1 linear structural equations."""

    if n_vars == 4:

        if max_lag == 2:
            return [
                "X4[t] = 0.25 * X1[t-2] + e4",
                "X3[t] = 0.35 * X4[t] + e3",
                "X2[t] = 0.3 * X3[t-1] + e2",
                "X1[t] = 0.4 * X2[t] + e1",
            ]

        elif max_lag == 3:
            return [
                "X4[t] = 0.25 * X1[t-2] + e4",
                "X3[t] = 0.35 * X4[t] + 0.2 * X2[t-3] + e3",
                "X2[t] = 0.3 * X3[t-1] + e2",
                "X1[t] = 0.4 * X2[t] + e1",
            ]

        elif max_lag == 4:
            return [
                "X4[t] = 0.25 * X1[t-4] + e4",
                "X3[t] = 0.35 * X4[t] + 0.2 * X2[t-3] + e3",
                "X2[t] = 0.3 * X3[t-1] + e2",
                "X1[t] = 0.4 * X2[t] + e1",
            ]

    raise ValueError(
        f"Unsupported configuration: n_vars={n_vars}, max_lag={max_lag}"
    )


def extract_linear_links(equations):
    """Convert structural equations into causal links."""

    links = {}

    for equation in equations:

        left, right = equation.split("=")

        target = left.strip().split("[")[0]

        terms = [term.strip() for term in right.split("+")]

        for term in terms:

            if "*" not in term or "X" not in term:
                continue

            parts = term.split("*")

            coefficient = float(parts[0].strip())

            var_part = parts[1].strip()

            source = var_part.split("[")[0]

            lag_part = var_part.split("[")[1].split("]")[0]

            if lag_part == "t":
                lag = 0
            else:
                lag = -int(lag_part.split("-")[1])

            links[(source, lag, target)] = coefficient

    return links


if __name__ == "__main__":

    equations = get_linear_equations(
        n_vars=4,
        max_lag=2
    )

    links = extract_linear_links(equations)

    print("Equations:")
    for equation in equations:
        print(equation)

    print("\nExtracted causal links:")

    for link, weight in links.items():
        print(link, "=>", weight)
