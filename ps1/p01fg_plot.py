from os import PathLike
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ps1.types import Float64Matrix
from . import util
from .p01b_logreg import LogisticRegression
from .p01e_gda import GDA

def plot(
    x: Float64Matrix,
    y: npt.NDArray[np.float64],
    logreg_theta: npt.NDArray[np.float64],
    gda_theta: npt.NDArray[np.float64],
    gda_theta_0: float,
    title: str,
    save_path: str | PathLike[str] | None,
    optimized: bool,
) -> None:
    plt.figure()

    plt.plot(x[y == 1, -2], x[y == 1, -1], "b.", linewidth=2, label="y = 1")
    plt.plot(x[y == 0, -2], x[y == 0, -1], "g.", linewidth=2, label="y = 0")

    margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1]))*0.2

    x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
    logreg_x2 = -(logreg_theta[0] + logreg_theta[1] * x1) / logreg_theta[2]
    gda_x2 = -(gda_theta_0 + gda_theta[0] * x1) / gda_theta[1]
    if optimized:
        gda_x2 = np.exp(gda_x2)

    plt.plot(x1, logreg_x2, c="red", linewidth=2, label="Logistic regression")
    plt.plot(x1, gda_x2, c="blue", linewidth=2, label="GDA")

    plt.xlim(x[:, -2].min()-margin1, x[:, -2].max()+margin1)
    plt.ylim(x[:, -1].min()-margin2, x[:, -1].max()+margin2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)

def main(train_path: str | PathLike[str], title: str, save_path: str | PathLike[str]):
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    log_clf = LogisticRegression()
    log_clf.fit(x_train, y_train)

    x_train_gda = x_train[:, 1:]
    gda_clf = GDA()
    gda_clf.fit(x_train_gda, y_train)

    plot(x_train, y_train, log_clf.theta, gda_clf.theta, gda_clf.theta_0, title, save_path, optimized=False)

def main_h(
    train_path: str | PathLike[str],
    train_title: str,
    save_train_path: str | PathLike[str],
    valid_path: str|PathLike[str],
    valid_title: str,
    save_valid_path: str | PathLike[str],
):
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    log_clf = LogisticRegression()
    log_clf.fit(x_train, y_train)

    x_train_gda = x_train[:, 1:].copy()
    x_train_gda[:, 1] = np.log(x_train_gda[:, 1])

    gda_clf = GDA()
    gda_clf.fit(x_train_gda, y_train)

    plot(x_train, y_train, log_clf.theta, gda_clf.theta, gda_clf.theta_0, train_title, save_train_path, optimized=True)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    plot(x_valid, y_valid, log_clf.theta, gda_clf.theta, gda_clf.theta_0, valid_title, save_valid_path, optimized=True)

    x_valid_gda = x_valid[:,1:].copy()
    x_valid_gda[:, 1] = np.log(x_valid_gda[:, 1])
    y_pred = gda_clf.predict(x_valid_gda)
    error_rate = np.mean(y_pred != y_valid)
    print(f"Validation error rate: {error_rate:.4f}")
