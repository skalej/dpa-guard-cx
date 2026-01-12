"""add playbooks and pgvector

Revision ID: 0005_add_playbooks_pgvector
Revises: 0004_add_export_fields
Create Date: 2024-01-05 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = "0005_add_playbooks_pgvector"
down_revision = "0004_add_export_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "playbooks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("version", sa.String(length=32), nullable=True),
        sa.Column("source_object_key", sa.Text(), nullable=True),
        sa.Column("source_filename", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="uploaded"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "playbook_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("playbook_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("meta_json", postgresql.JSONB(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("playbook_chunks")
    op.drop_table("playbooks")
