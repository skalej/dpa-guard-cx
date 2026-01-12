"""add review playbook id

Revision ID: 0006_add_review_playbook_id
Revises: 0005_add_playbooks_pgvector
Create Date: 2024-01-06 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0006_add_review_playbook_id"
down_revision = "0005_add_playbooks_pgvector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("reviews", sa.Column("playbook_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index("ix_reviews_playbook_id", "reviews", ["playbook_id"])


def downgrade() -> None:
    op.drop_index("ix_reviews_playbook_id", table_name="reviews")
    op.drop_column("reviews", "playbook_id")
