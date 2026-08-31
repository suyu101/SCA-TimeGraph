import numpy as np
import torch
import torch.nn as nn


class SCAModel(nn.Module):
    """
    Initial SCA model.

    Input:
        Temporal window of shape:
        (batch, 3, 4)

    Output:
        Causal predictions of shape:
        (batch, 4, 4, 3)

    Output meaning:
        A_pred[source, target, lag]

    where:
        source = causal source variable
        target = affected variable
        lag = 0, 1, or 2
    """

    def __init__(
        self,
        n_variables=4,
        max_lag=2,
        hidden_dim=64,
    ):
        super().__init__()

        self.n_variables = n_variables
        self.max_lag = max_lag
        self.n_lags = max_lag + 1

        # -------------------------------------------------
        # Temporal encoder
        #
        # Each variable gets its own temporal representation.
        #
        # For max_lag = 2:
        # [t-2, t-1, t]
        # -------------------------------------------------

        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.n_lags, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # -------------------------------------------------
        # Pairwise causal scorer
        #
        # For every source -> target pair,
        # predict one value for each lag.
        # -------------------------------------------------

        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_lags),
        )

    def forward(self, x):
        """
        Input:
            x shape = (batch, 3, 4)

        Output:
            shape = (batch, 4, 4, 3)
        """

        batch_size = x.shape[0]

        # -------------------------------------------------
        # Rearrange input
        #
        # Before:
        # (batch, time, variables)
        #
        # After:
        # (batch, variables, time)
        #
        # This gives each variable its own temporal history.
        # -------------------------------------------------

        x = x.transpose(1, 2)

        # Shape:
        # (batch, 4, 3)

        # -------------------------------------------------
        # Encode each variable's temporal history
        # -------------------------------------------------

        encoded = self.temporal_encoder(x)

        # Shape:
        # (batch, 4, hidden_dim)

        # -------------------------------------------------
        # Create source and target representations
        # -------------------------------------------------

        source = encoded.unsqueeze(2)
        target = encoded.unsqueeze(1)

        # source:
        # (batch, 4, 1, hidden)

        # target:
        # (batch, 1, 4, hidden)

        source = source.expand(
            batch_size,
            self.n_variables,
            self.n_variables,
            -1,
        )

        target = target.expand(
            batch_size,
            self.n_variables,
            self.n_variables,
            -1,
        )

        # -------------------------------------------------
        # Combine source and target representations
        # -------------------------------------------------

        pair_representation = torch.cat(
            [source, target],
            dim=-1,
        )

        # Shape:
        # (batch, 4, 4, hidden_dim * 2)

        # -------------------------------------------------
        # Predict causal strength for each lag
        # -------------------------------------------------

        edge_scores = self.edge_scorer(
            pair_representation
        )

        # Shape:
        # (batch, 4, 4, 3)

        # Convert predictions to range [0, 1]
        edge_scores = torch.sigmoid(edge_scores)

        return edge_scores


def create_temporal_windows(X, max_lag=2):
    """
    Create chronological temporal windows.

    Input:
        X shape = (time, variables)

    Output:
        windows shape =
        (time - max_lag, max_lag + 1, variables)

    For max_lag = 2:

        target t = 2
        window = [X[0], X[1], X[2]]
    """

    windows = []

    for t in range(max_lag, len(X)):
        window = X[t - max_lag : t + 1]
        windows.append(window)

    return np.asarray(
        windows,
        dtype=np.float32,
    )


def generate_A_pred(X, model):
    """
    Generate predictions for the complete time series.

    Final output:
        (time, source, target, lag)

    The first max_lag time points do not have enough
    historical observations to form a complete window,
    so they are initialized to zero for this initial
    interface test.
    """

    model.eval()

    n_time = X.shape[0]
    n_variables = X.shape[1]
    n_lags = model.n_lags

    # -------------------------------------------------
    # Create complete output array
    # -------------------------------------------------

    A_pred = np.zeros(
        (
            n_time,
            n_variables,
            n_variables,
            n_lags,
        ),
        dtype=np.float32,
    )

    # -------------------------------------------------
    # Create temporal windows
    # -------------------------------------------------

    windows = create_temporal_windows(
        X,
        model.max_lag,
    )

    # Convert to PyTorch tensor

    x_tensor = torch.tensor(
        windows,
        dtype=torch.float32,
    )

    # -------------------------------------------------
    # Generate predictions
    # -------------------------------------------------

    with torch.no_grad():

        predictions = model(
            x_tensor
        )

    # -------------------------------------------------
    # Store predictions
    #
    # First valid prediction corresponds to t = max_lag
    # -------------------------------------------------

    A_pred[model.max_lag:] = (
        predictions.cpu().numpy()
    )

    return A_pred


if __name__ == "__main__":

    print(
        "===== SCA MODEL SANITY CHECK ====="
    )

    # -------------------------------------------------
    # Dummy development input
    #
    # 5000 time points
    # 4 variables
    # -------------------------------------------------

    X_dummy = np.random.randn(
        5000,
        4,
    ).astype(np.float32)

    print(
        "\nCreating model..."
    )

    # -------------------------------------------------
    # Create model
    # -------------------------------------------------

    model = SCAModel(
        n_variables=4,
        max_lag=2,
        hidden_dim=64,
    )

    # -------------------------------------------------
    # Generate predictions
    # -------------------------------------------------

    print(
        "Generating predictions..."
    )

    A_pred = generate_A_pred(
        X_dummy,
        model,
    )

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print(
        "\nInput shape:"
    )

    print(
        X_dummy.shape
    )

    print(
        "\nA_pred shape:"
    )

    print(
        A_pred.shape
    )

    print(
        "\nExpected:"
    )

    print(
        "(5000, 4, 4, 3)"
    )

    print(
        "\nA_pred dtype:"
    )

    print(
        A_pred.dtype
    )

    print(
        "\nMinimum prediction:"
    )

    print(
        A_pred.min()
    )

    print(
        "\nMaximum prediction:"
    )

    print(
        A_pred.max()
    )

    # -------------------------------------------------
    # Verify interface
    # -------------------------------------------------

    if A_pred.shape == (
        5000,
        4,
        4,
        3,
    ):

        print(
            "\nSUCCESS: "
            "A_pred interface is correct!"
        )

    else:

        print(
            "\nERROR: "
            "A_pred shape is incorrect."
        )