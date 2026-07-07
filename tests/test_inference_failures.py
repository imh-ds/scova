import numpy as np
import pytest

from scova.inference import run_simultaneous_inference


def valid_inputs():
    influence = np.array([[-1.0], [0.0], [1.0]])
    return {
        "family": ("a",),
        "estimates": np.array([0.2]),
        "standard_errors": np.array([0.1]),
        "influence_values": influence,
        "weights": np.array([[1.0, -1.0]]),
        "group_covariance": np.array([[0.01, 0.0], [0.0, 0.01]]),
        "confidence_level": 0.95,
        "n_bootstrap": 9,
        "random_state": 1,
        "batch_size": 3,
    }


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"family": ()}, "at least one"),
        ({"family": ("a", "a")}, "duplicate"),
        ({"confidence_level": 1}, "strictly between"),
        ({"n_bootstrap": 0}, "at least 1"),
        ({"batch_size": 0}, "at least 1"),
        ({"estimates": np.array([np.nan])}, "finite"),
        ({"standard_errors": np.array([0.0])}, "positive"),
        ({"estimates": np.array([1.0, 2.0])}, "must align"),
        ({"weights": np.ones((2, 2))}, "weights and family"),
    ],
)
def test_simultaneous_inference_validation(updates, message) -> None:
    values = valid_inputs()
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        run_simultaneous_inference(**values)


def test_zero_rank_and_missing_contrast() -> None:
    values = valid_inputs()
    values["group_covariance"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="zero numerical rank"):
        run_simultaneous_inference(**values)

