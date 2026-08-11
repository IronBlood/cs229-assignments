from os import PathLike
import numpy as np
import numpy.typing as npt
from . import util

from .linear_model import LinearModel
from .types import Float64Matrix

def main(train_path: str | PathLike[str], eval_path: str | PathLike[str], pred_path: str | PathLike[str]):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_valid)
    np.savetxt(pred_path, y_pred)

    error_rate = np.mean(y_pred != y_valid)
    print(f"Validation error rate: {error_rate:.4f}")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x: Float64Matrix, y: npt.NDArray[np.float64]) -> None:
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def h(theta: npt.NDArray[np.float64], x: Float64Matrix):
            """Calculate the hypothesis vector

            Related to the equation (LaTeX) h_\\theta(x) = \\frac{1}{1 + \\exp^{-\\theta^T x}}

            Args:
                theta: 1D array, shape of (n,)
                x: features, shape of (m, n)
            Returns:
                The hypothesis of all training examples, shape of (m,)
            """
            return 1 / (1 + np.exp(- (x @ theta)))

        def gradient(theta: npt.NDArray[np.float64], x: Float64Matrix, y: npt.NDArray[np.float64]):
            """Calculate the gradient

            Args:
                theta: 1D array, shape of (n,)
                x: features, shape of (m, n)
                y: 1D array, shape of (m,)
            Returns:
                shape of (n,)
            """

            m = x.shape[0]
            # vector of h
            _tmp = h(theta, x)
            # calculate the residual in place (h - y)
            _tmp -= y
            # pay attention to the sign, `-` isn't used because the residual is already (h - y)
            return x.T @ _tmp / m

        def hessian(theta: npt.NDArray[np.float64], x: Float64Matrix) -> Float64Matrix:
            """
            Args:
                theta: 1D array, shape of (n,)
                x: features, shape of (m, n)
            Returns:
                shape of (n,n)
            """

            m = x.shape[0]
            # shape: (m,)
            _h = h(theta, x)
            # shape: (m,)
            weights = _h * (1 - _h)
            # Reshape weights to (m, 1), then broadcasting scales each row of x.
            weighted_x = weights.reshape(m, 1) * x
            return x.T @ weighted_x / m

        def next_theta(theta: npt.NDArray[np.float64], x: Float64Matrix, y: npt.NDArray[np.float64]):
            # `solve(a, b)` solves the equation `ax = b`
            # this is equivalent to `np.linalg.inv(hessian(theta, x)) @ gradient(theta, x, y)`
            # because ax = b => a^{-1}ax = x = a^{-1}b
            return theta - np.linalg.solve(hessian(theta, x), gradient(theta, x, y))

        n = x.shape[1]
        prev_theta = np.zeros(n)
        curr_theta = next_theta(prev_theta, x, y)

        while np.linalg.norm(curr_theta - prev_theta, 1) >= self.eps:
            prev_theta = curr_theta
            curr_theta = next_theta(curr_theta, x, y)

        self.theta = curr_theta
        # *** END CODE HERE ***

    def predict_probability(self, x):
        if self.theta is None:
            raise ValueError("Model must be fit before calling predict")

        return 1 / (1 + np.exp(-(x @ self.theta)))

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (self.predict_probability(x) >= 0.5).astype(np.float64)
        # *** END CODE HERE ***
