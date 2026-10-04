"""human-in-the-loop approvals

Revision ID: 9b4e2c7a1d35
Revises: 3c1d9a4b7f20
Create Date: 2026-10-04 12:00:00.000000

Hand-written: autogenerate does not detect changed CHECK constraints. The two CHECK constraints
keep their names and get the new values, and tool_calls gets the unique constraint that lets a
pending row be updated instead of duplicated.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b4e2c7a1d35"
down_revision: Union[str, Sequence[str], None] = "3c1d9a4b7f20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_STATUSES = "'running', 'completed', 'failed', 'step_limit', 'timeout', 'cancelled'"
NEW_STATUSES = OLD_STATUSES + ", 'waiting_approval', 'expired'"
OLD_APPROVALS = "'not_required', 'pending', 'approved', 'rejected'"
NEW_APPROVALS = OLD_APPROVALS + ", 'expired'"


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("ck_agent_runs_status", "agent_runs", type_="check")
    op.create_check_constraint("ck_agent_runs_status", "agent_runs", f"status IN ({NEW_STATUSES})")
    op.drop_constraint("ck_tool_calls_approval_status", "tool_calls", type_="check")
    op.create_check_constraint(
        "ck_tool_calls_approval_status", "tool_calls", f"approval_status IN ({NEW_APPROVALS})"
    )
    op.create_unique_constraint(
        "uq_tool_calls_run_id_tool_call_id", "tool_calls", ["run_id", "tool_call_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Rows that use the new values would violate the old constraints: refuse instead of guessing
    op.drop_constraint("uq_tool_calls_run_id_tool_call_id", "tool_calls", type_="unique")
    op.drop_constraint("ck_tool_calls_approval_status", "tool_calls", type_="check")
    op.create_check_constraint(
        "ck_tool_calls_approval_status", "tool_calls", f"approval_status IN ({OLD_APPROVALS})"
    )
    op.drop_constraint("ck_agent_runs_status", "agent_runs", type_="check")
    op.create_check_constraint("ck_agent_runs_status", "agent_runs", f"status IN ({OLD_STATUSES})")
