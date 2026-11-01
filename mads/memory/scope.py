"""Unified scoping model for memory categories.

Three axes: USER, PROJECT, DATASET. Each category declares which axes are
part of its identity (filterable on retrieval) vs. audit (recorded but not
filtered).

Roles:

* ``IDENTITY_REQUIRED`` — column is part of the identity tuple; rows must
  match exactly. Emitted as ``col = ?`` in filter clauses.
* ``IDENTITY_OPTIONAL`` — column is part of the identity tuple but
  nullable; rows match if the column IS NULL (= global) OR equals the
  active scope value. Emitted as ``(col IS NULL OR col = ?)``.
* ``AUDIT`` — column is recorded on every row for accountability but
  does NOT participate in retrieval filtering.
* ``REFERENCE`` — column carries a plain string pointer to another
  category (e.g. cat 5's ``datasets_used`` referencing cat 2 fingerprints);
  no FK enforcement, no filtering by default.

The :class:`CategorySpec` instances at the bottom of this module are the
*truth* for each shipped memory category. Store classes may issue
hand-written SQL but the spec is the authoritative declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Axis(Enum):
    """One of the three scoping axes."""

    USER = "user"
    PROJECT = "project"
    DATASET = "dataset"


@dataclass(frozen=True)
class Scope:
    """Active scope for a retrieval call.

    ``user_id`` is required (every retrieval is on behalf of *someone*).
    ``project_id`` and ``dataset_fingerprint`` are optional — categories
    that don't use those axes will simply ignore them in their filter
    clause.
    """

    user_id: str
    project_id: Optional[str] = None
    dataset_fingerprint: Optional[str] = None


class AxisRole(Enum):
    """Per-category role of one axis."""

    IDENTITY_REQUIRED = "identity_required"   # WHERE col = ? always
    IDENTITY_OPTIONAL = "identity_optional"   # WHERE (col IS NULL OR col = ?)
    AUDIT = "audit"                            # recorded; not filtered
    REFERENCE = "reference"                    # plain string, no FK enforcement


# Column name used in the SQL filter clause for each axis.
_AXIS_COLUMN = {
    Axis.USER: "user_id",
    Axis.PROJECT: "project_id",
    Axis.DATASET: "dataset_fingerprint",
}


@dataclass(frozen=True)
class CategorySpec:
    """Declarative spec for a memory category's scoping axes.

    The dict ``axes`` maps each :class:`Axis` the category uses to its
    :class:`AxisRole`. Axes not listed are not part of the category's
    schema at all.
    """

    name: str
    axes: dict[Axis, AxisRole]

    def filter_clause(self, scope: Scope) -> tuple[str, list]:
        """Return ``(WHERE clause fragment, params)`` for ``scope``.

        Only ``IDENTITY_REQUIRED`` and ``IDENTITY_OPTIONAL`` axes
        contribute to the clause; ``AUDIT`` / ``REFERENCE`` are noise from
        a retrieval standpoint and emit nothing.

        The fragment is suitable for splicing into a larger ``WHERE``
        clause via ``AND``. If no axes participate, returns ``("", [])``.
        """
        clauses: list[str] = []
        params: list = []
        scope_values = {
            Axis.USER: scope.user_id,
            Axis.PROJECT: scope.project_id,
            Axis.DATASET: scope.dataset_fingerprint,
        }
        # Iterate in Axis-declaration order for stable SQL output.
        for axis in Axis:
            if axis not in self.axes:
                continue
            role = self.axes[axis]
            col = _AXIS_COLUMN[axis]
            val = scope_values[axis]
            if role == AxisRole.IDENTITY_REQUIRED:
                clauses.append(f"{col} = ?")
                params.append(val)
            elif role == AxisRole.IDENTITY_OPTIONAL:
                clauses.append(f"({col} IS NULL OR {col} = ?)")
                params.append(val)
            # AUDIT and REFERENCE contribute nothing to the filter.
        return " AND ".join(clauses), params


# ---------------------------------------------------------------- specs


PREFERENCE_SPEC = CategorySpec(
    name="user_preference",
    axes={
        Axis.USER: AxisRole.IDENTITY_REQUIRED,
        Axis.PROJECT: AxisRole.IDENTITY_OPTIONAL,
        # v1.3: dataset axis. NULL = applies to all datasets; a non-NULL
        # value scopes the preference to that specific dataset
        # (identified by its semantic_fingerprint).
        Axis.DATASET: AxisRole.IDENTITY_OPTIONAL,
    },
)

DATASET_SPEC = CategorySpec(
    name="dataset_schema",
    axes={
        Axis.DATASET: AxisRole.IDENTITY_REQUIRED,
        Axis.USER: AxisRole.AUDIT,
    },
)

METRIC_SPEC = CategorySpec(
    name="metric_definition",
    axes={
        Axis.PROJECT: AxisRole.IDENTITY_REQUIRED,
        Axis.USER: AxisRole.AUDIT,
        Axis.DATASET: AxisRole.REFERENCE,
    },
)

DATA_QUALITY_SPEC = CategorySpec(
    name="data_quality_issue",
    axes={
        Axis.DATASET: AxisRole.IDENTITY_REQUIRED,
        Axis.USER: AxisRole.AUDIT,
        Axis.PROJECT: AxisRole.AUDIT,
    },
)

EPISODE_SPEC = CategorySpec(
    name="analysis_episode",
    axes={
        Axis.PROJECT: AxisRole.IDENTITY_REQUIRED,
        Axis.USER: AxisRole.AUDIT,
        Axis.DATASET: AxisRole.REFERENCE,
    },
)


__all__ = [
    "Axis",
    "Scope",
    "AxisRole",
    "CategorySpec",
    "PREFERENCE_SPEC",
    "DATASET_SPEC",
    "METRIC_SPEC",
    "DATA_QUALITY_SPEC",
    "EPISODE_SPEC",
]
