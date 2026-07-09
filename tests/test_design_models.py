import numpy as np
import pandas as pd
import pytest

from scova import DesignDeclaration, DesignLock, OutcomeFreeDesignData


def declaration(seed: int = 7) -> DesignDeclaration:
    return DesignDeclaration(
        group="group",
        covariates=("x1", "x2"),
        random_state=seed,
        candidate_subsets=(("a", "b"),),
        gate_policy={"threshold_version": "stage3-directional-v1"},
    )


def design_data() -> OutcomeFreeDesignData:
    return OutcomeFreeDesignData.from_arrays(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]],
        ["a", "b", "a", "b"],
        row_ids=["r0", "r1", "r2", "r3"],
    )


def test_design_declaration_is_stable_and_outcome_free() -> None:
    first = declaration()
    second = declaration()
    assert first.declaration_hash == second.declaration_hash
    assert "outcome" not in first.to_dict()
    with pytest.raises(ValueError, match="candidate subsets"):
        DesignDeclaration("group", ("x1",), candidate_subsets=(("a", "a"),))


def test_outcome_free_data_copies_and_hashes_arrays() -> None:
    values = np.array([[0.0], [1.0], [2.0], [3.0]])
    data = OutcomeFreeDesignData.from_arrays(values, ["a", "b", "a", "b"])
    original_hash = data.data_hash
    values[0, 0] = 100.0
    assert data.covariates[0, 0] == 0.0
    assert data.data_hash == original_hash
    assert not data.covariates.flags.writeable
    with pytest.raises(TypeError, match="not a dataframe"):
        OutcomeFreeDesignData.from_arrays(
            pd.DataFrame({"x1": [0.0, 1.0], "outcome": [1.0, 2.0]}), ["a", "b"]
        )


def test_lock_is_deterministic_round_trippable_and_verifiable() -> None:
    declared = declaration()
    data = design_data()
    assignments = ("design", "estimation", "design", "estimation")
    first = DesignLock.create(
        declared, data, assignments, design_metadata={"graph": {"edges": [["a", "b"]]}}
    )
    second = DesignLock.create(
        declared, data, assignments, design_metadata={"graph": {"edges": [["a", "b"]]}}
    )
    assert first.lock_hash == second.lock_hash
    assert first.estimation_row_ids == ("r1", "r3")
    assert DesignLock.from_dict(first.to_dict()) == first
    first.verify(declared, data)


def test_lock_rejects_tampering_and_design_changes() -> None:
    declared = declaration()
    data = design_data()
    lock = DesignLock.create(
        declared,
        data,
        ("design", "estimation", "design", "estimation"),
        design_metadata={"graph": {"edges": []}},
    )
    with pytest.raises(ValueError, match="locked declaration"):
        lock.verify(declaration(seed=8), data)
    altered = OutcomeFreeDesignData.from_arrays(
        [[9.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]],
        ["a", "b", "a", "b"],
        row_ids=["r0", "r1", "r2", "r3"],
    )
    with pytest.raises(ValueError, match="design data"):
        lock.verify(declared, altered)
    invalid = lock.to_dict()
    invalid["design_metadata"] = {"graph": {"edges": [["a", "b"]]}}
    with pytest.raises(ValueError, match="checksum"):
        DesignLock.from_dict(invalid)
