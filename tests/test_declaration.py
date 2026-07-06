from dataclasses import FrozenInstanceError

import pytest

from scova import ContrastSpec, SCOVADeclaration


def test_declaration_is_immutable_and_hash_is_stable() -> None:
    first = SCOVADeclaration("y", "a", ("x1", "x2"), random_state=7)
    second = SCOVADeclaration("y", "a", ("x1", "x2"), random_state=7)
    assert first.declaration_hash == second.declaration_hash
    with pytest.raises(FrozenInstanceError):
        first.n_splits = 3  # type: ignore[misc]


def test_contrast_contract() -> None:
    contrast = ContrastSpec("one minus two", (("one", 1), ("two", -1)))
    declaration = SCOVADeclaration("y", "a", ("x",), contrasts=(contrast,))
    assert declaration.to_dict()["contrasts"][0]["name"] == "one minus two"
    with pytest.raises(ValueError, match="sum to zero"):
        ContrastSpec("bad", (("one", 1), ("two", 1)))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"covariates": ()}, "covariate"),
        ({"outcome": "x", "covariates": ("x",)}, "distinct"),
        ({"n_splits": 1}, "at least 2"),
    ],
)
def test_invalid_declarations(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {"outcome": "y", "group": "a", "covariates": ("x",)}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SCOVADeclaration(**values)  # type: ignore[arg-type]

