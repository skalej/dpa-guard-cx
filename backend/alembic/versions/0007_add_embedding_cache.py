"""add embedding cache

Revision ID: 0007_add_embedding_cache
Revises: 0006_add_review_playbook_id
Create Date: 2026-01-12 11:41:00.000000
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0007_add_embedding_cache"
down_revision = "0006_add_review_playbook_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "embedding_cache",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("dims", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_embedding_cache_sha256", "embedding_cache", ["sha256"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_embedding_cache_sha256", table_name="embedding_cache")
    op.drop_table("embedding_cache")
