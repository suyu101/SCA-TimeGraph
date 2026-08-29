import numpy as np
import pandas as pd


def generate_regime_change_data(
    n_points=5000,
    change_point=2500,
    noise_scale=0.1,
    random_state=42,
):
    """
    Generate a 4-variable time series with a known
    causal-structure change at `change_point`.

    Regime 1:
        X1(t-2) -> X4(t)
        X4(t)   -> X3(t)
        X3(t-1) -> X2(t)
        X2(t)   -> X1(t)

    Regime 2:
        X1(t-2) -> X4(t)
        X2(t-1) -> X3(t)   NEW
        X3(t-1) -> X2(t)
        X2(t)   -> X1(t)
    """

    rng = np.random.default_rng(random_state)

    X = np.zeros((n_points, 4), dtype=np.float32)

    # --------------------------------------------
    # Initialize first two time points
    # --------------------------------------------

    X[:2] = rng.normal(
        0,
        noise_scale,
        size=(2, 4),
    )

    # --------------------------------------------
    # Generate observations
    # --------------------------------------------

    for t in range(2, n_points):

        e1, e2, e3, e4 = rng.normal(
            0,
            noise_scale,
            size=4,
        )

        # X4 depends on X1 two steps ago
        X4 = 0.25 * X[t - 2, 0] + e4

        # X2 depends on X3 one step ago
        X2 = 0.30 * X[t - 1, 2] + e2

        # X1 depends contemporaneously on X2
        X1 = 0.40 * X2 + e1

        # ----------------------------------------
        # Regime change
        # ----------------------------------------

        if t < change_point:

            # BEFORE:
            # X3 <- X4
            X3 = 0.35 * X4 + e3

        else:

            # AFTER:
            # X3 <- X2(t-1)
            X3 = 0.35 * X[t - 1, 1] + e3

        X[t] = [
            X1,
            X2,
            X3,
            X4,
        ]

    time = np.arange(n_points)

    df = pd.DataFrame(
        X,
        columns=["X1", "X2", "X3", "X4"],
    )

    df["time"] = time

    return df


if __name__ == "__main__":

    df = generate_regime_change_data()

    output_path = (
        "Datasets/regime_change_a1.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("===== Regime Change Dataset =====")
    print("Shape:", df.shape)
    print("Change point: 2500")
    print("Output:", output_path)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nRows around change point:")
    print(
        df.iloc[2498:2503]
    )
