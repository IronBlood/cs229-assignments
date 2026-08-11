from os import PathLike
import numpy as np
import numpy.typing as npt
from . import util

from .linear_model import LinearModel
from .types import Float64Matrix

def main(train_path: str | PathLike[str], eval_path: str | PathLike[str], pred_path: str | PathLike[str]):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    y_pred = clf.predict(x_valid)
    np.savetxt(pred_path, y_pred)
    error_rate = np.mean(y_pred != y_valid)
    print(f"Validation error rate: {error_rate:.4f}")
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x: Float64Matrix, y: npt.NDArray[np.float64]) -> None:
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        n = x.shape[1]

        acc_mu0 = np.zeros(n, dtype=x.dtype)
        acc_mu1 = np.zeros(n, dtype=x.dtype)

        count_1 = 0

        if y.shape[0] != m:
            raise ValueError("Rows of inputs and labels must match")

        for i in range(m):
            x_row = x[i]
            y_value = y[i]

            if y_value == 1.0:
                count_1 += 1
                acc_mu1 += x_row
            else:
                acc_mu0 += x_row

        phi = count_1 / m
        mu0: npt.NDArray[np.float64] = acc_mu0 / (m - count_1)
        mu1: npt.NDArray[np.float64] = acc_mu1 / count_1

        Sigma = np.zeros((n, n))
        for i in range(m):
            if y[i] == 1.0:
                sigma = x[i] - mu1
            else:
                sigma = x[i] - mu0
            Sigma += np.outer(sigma, sigma)
        Sigma /= m

        Sigma_inv = np.linalg.inv(Sigma)
        self.theta = Sigma_inv @ (mu1 - mu0)
        self.theta_0 = (mu0 @ Sigma_inv @ mu0 - mu1 @ Sigma_inv @ mu1) / 2 - np.log((1-phi) / phi)
        # *** END CODE HERE ***

    def predict(self, x: Float64Matrix):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        if self.theta is None or self.theta_0 is None:
            raise ValueError("Model must be fit before calling predict.")

        z = self.theta_0 + x @ self.theta

        return (z >= 0).astype(np.float64)
        # *** END CODE HERE
