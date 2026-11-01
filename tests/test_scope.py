"""Pytest suite for :mod:`mads.memory.scope`.

The scope module declares the unified scoping model (USER / PROJECT /
DATASET axes, IDENTITY / AUDIT / REFERENCE roles, per-category specs)
that the stores and retrievers now share. These tests confirm the
semantics of :meth:`CategorySpec.filter_clause` for each shipped spec
and the basic dataclass/enum contract.
"""

from __future__ import annotations

import pytest

from mads.memory.scope import (
    Axis,
    AxisRole,
    CategorySpec,
    DATASET_SPEC,
    DATA_QUALITY_SPEC,
    EPISODE_SPEC,
    METRIC_SPEC,
    PREFERENCE_SPEC,
    Scope,
)


# ----------------------------------------------------------- enum + dataclass


def test_axis_enum_values():
    """1. Axis enum has the three documented members with stable string values."""
    assert Axis.USER.value == "user"
    assert Axis.PROJECT.value == "project"
    assert Axis.DATASET.value == "dataset"
    assert {a.value for a in Axis} == {"user", "project", "dataset"}


def test_axis_role_enum_values():
    """2. AxisRole enum has the four documented members."""
    assert AxisRole.IDENTITY_REQUIRED.value == "identity_required"
    assert AxisRole.IDENTITY_OPTIONAL.value == "identity_optional"
    assert AxisRole.AUDIT.value == "audit"
    assert AxisRole.REFERENCE.value == "reference"


def test_scope_dataclass_defaults_and_immutability():
    """3. Scope is frozen; project_id and dataset_fingerprint default to None."""
    s = Scope(user_id="alice")
    assert s.user_id == "alice"
    assert s.project_id is None
    assert s.dataset_fingerprint is None
    # Frozen → cannot mutate.
    with pytest.raises(Exception):
        s.user_id = "bob"  # type: ignore[misc]


def test_scope_with_all_axes():
    """4. Scope accepts all three axes."""
    s = Scope(user_id="alice", project_id="proj1", dataset_fingerprint="abc")
    assert s.project_id == "proj1"
    assert s.dataset_fingerprint == "abc"


# ----------------------------------------------------------- filter_clause


def test_filter_clause_identity_required():
    """5. IDENTITY_REQUIRED axis emits ``col = ?`` and binds its scope value."""
    spec = CategorySpec(
        name="x",
        axes={Axis.USER: AxisRole.IDENTITY_REQUIRED},
    )
    sql, params = spec.filter_clause(Scope(user_id="alice"))
    assert sql == "user_id = ?"
    assert params == ["alice"]


def test_filter_clause_identity_optional_emits_null_or_equal():
    """6. IDENTITY_OPTIONAL axis emits ``(col IS NULL OR col = ?)``."""
    spec = CategorySpec(
        name="x",
        axes={Axis.PROJECT: AxisRole.IDENTITY_OPTIONAL},
    )
    sql, params = spec.filter_clause(
        Scope(user_id="alice", project_id="proj1")
    )
    assert sql == "(project_id IS NULL OR project_id = ?)"
    assert params == ["proj1"]


def test_filter_clause_audit_and_reference_are_silent():
    """7. AUDIT and REFERENCE axes contribute no SQL fragment."""
    spec = CategorySpec(
        name="x",
        axes={
            Axis.USER: AxisRole.AUDIT,
            Axis.DATASET: AxisRole.REFERENCE,
        },
    )
    sql, params = spec.filter_clause(
        Scope(user_id="alice", dataset_fingerprint="abc")
    )
    assert sql == ""
    assert params == []


def test_filter_clause_combines_with_and():
    """8. Multiple identity axes combine with `` AND ``."""
    spec = CategorySpec(
        name="x",
        axes={
            Axis.USER: AxisRole.IDENTITY_REQUIRED,
            Axis.PROJECT: AxisRole.IDENTITY_OPTIONAL,
        },
    )
    sql, params = spec.filter_clause(
        Scope(user_id="alice", project_id="proj1")
    )
    assert sql == "user_id = ? AND (project_id IS NULL OR project_id = ?)"
    assert params == ["alice", "proj1"]


# ----------------------------------------------------------- per-category specs


def test_preference_spec_axes():
    """9. PREFERENCE_SPEC (v1.3): USER required, PROJECT optional, DATASET optional."""
    assert PREFERENCE_SPEC.name == "user_preference"
    assert PREFERENCE_SPEC.axes == {
        Axis.USER: AxisRole.IDENTITY_REQUIRED,
        Axis.PROJECT: AxisRole.IDENTITY_OPTIONAL,
        Axis.DATASET: AxisRole.IDENTITY_OPTIONAL,
    }
    sql, params = PREFERENCE_SPEC.filter_clause(
        Scope(user_id="alice", project_id="proj1", dataset_fingerprint="fp1")
    )
    assert sql == (
        "user_id = ? "
        "AND (project_id IS NULL OR project_id = ?) "
        "AND (dataset_fingerprint IS NULL OR dataset_fingerprint = ?)"
    )
    assert params == ["alice", "proj1", "fp1"]


def test_preference_spec_dataset_axis_role():
    """9b (v1.3). DATASET axis on PREFERENCE_SPEC is IDENTITY_OPTIONAL."""
    assert PREFERENCE_SPEC.axes[Axis.DATASET] == AxisRole.IDENTITY_OPTIONAL


def test_dataset_spec_axes():
    """10. DATASET_SPEC: DATASET required, USER audit."""
    assert DATASET_SPEC.name == "dataset_schema"
    assert DATASET_SPEC.axes[Axis.DATASET] == AxisRole.IDENTITY_REQUIRED
    assert DATASET_SPEC.axes[Axis.USER] == AxisRole.AUDIT
    sql, params = DATASET_SPEC.filter_clause(
        Scope(user_id="alice", dataset_fingerprint="abc")
    )
    assert sql == "dataset_fingerprint = ?"
    assert params == ["abc"]


def test_metric_spec_axes():
    """11. METRIC_SPEC: PROJECT required, USER audit, DATASET reference."""
    assert METRIC_SPEC.name == "metric_definition"
    assert METRIC_SPEC.axes[Axis.PROJECT] == AxisRole.IDENTITY_REQUIRED
    assert METRIC_SPEC.axes[Axis.USER] == AxisRole.AUDIT
    assert METRIC_SPEC.axes[Axis.DATASET] == AxisRole.REFERENCE
    sql, params = METRIC_SPEC.filter_clause(
        Scope(user_id="alice", project_id="proj1", dataset_fingerprint="abc")
    )
    # DATASET REFERENCE and USER AUDIT contribute nothing.
    assert sql == "project_id = ?"
    assert params == ["proj1"]


def test_data_quality_spec_axes():
    """12. DATA_QUALITY_SPEC: DATASET required, USER + PROJECT audit."""
    assert DATA_QUALITY_SPEC.name == "data_quality_issue"
    assert DATA_QUALITY_SPEC.axes[Axis.DATASET] == AxisRole.IDENTITY_REQUIRED
    assert DATA_QUALITY_SPEC.axes[Axis.USER] == AxisRole.AUDIT
    assert DATA_QUALITY_SPEC.axes[Axis.PROJECT] == AxisRole.AUDIT
    sql, params = DATA_QUALITY_SPEC.filter_clause(
        Scope(user_id="alice", project_id="proj1", dataset_fingerprint="abc")
    )
    assert sql == "dataset_fingerprint = ?"
    assert params == ["abc"]


def test_episode_spec_axes():
    """13. EPISODE_SPEC: PROJECT required, USER audit, DATASET reference."""
    assert EPISODE_SPEC.name == "analysis_episode"
    assert EPISODE_SPEC.axes[Axis.PROJECT] == AxisRole.IDENTITY_REQUIRED
    assert EPISODE_SPEC.axes[Axis.USER] == AxisRole.AUDIT
    assert EPISODE_SPEC.axes[Axis.DATASET] == AxisRole.REFERENCE
    sql, params = EPISODE_SPEC.filter_clause(
        Scope(user_id="alice", project_id="proj1")
    )
    assert sql == "project_id = ?"
    assert params == ["proj1"]


def test_preference_spec_filter_global_only():
    """14. PreferenceSpec with project_id=None / dataset_fingerprint=None
    yields the IS NULL form on both optional axes."""
    sql, params = PREFERENCE_SPEC.filter_clause(
        Scope(user_id="alice", project_id=None, dataset_fingerprint=None)
    )
    # IDENTITY_OPTIONAL fires regardless of whether the value is None;
    # the IS NULL clause still matches NULL rows even when the bound
    # param is None.
    assert sql == (
        "user_id = ? "
        "AND (project_id IS NULL OR project_id = ?) "
        "AND (dataset_fingerprint IS NULL OR dataset_fingerprint = ?)"
    )
    assert params == ["alice", None, None]
