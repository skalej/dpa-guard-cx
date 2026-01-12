"""add review fields

Revision ID: 0002_review_fields
Revises: 0001_create_reviews
Create Date: 2024-01-02 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0002_review_fields"
down_revision = "0001_create_reviews"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "reviews",
        "status",
        existing_type=sa.String(length=32),
        server_default="draft",
        existing_nullable=False,
    )
    op.alter_column(
        "reviews",
        "context_json",
        existing_type=postgresql.JSON(),
        type_=postgresql.JSONB(),
        existing_nullable=True,
    )
    op.add_column("reviews", sa.Column("results_json", postgresql.JSONB(), nullable=True))
    op.add_column("reviews", sa.Column("source_object_key", sa.Text(), nullable=True))
    op.add_column("reviews", sa.Column("source_filename", sa.Text(), nullable=True))
    op.add_column("reviews", sa.Column("source_mime", sa.Text(), nullable=True))
    op.add_column("reviews", sa.Column("error_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("reviews", "error_message")
    op.drop_column("reviews", "source_mime")
    op.drop_column("reviews", "source_filename")
    op.drop_column("reviews", "source_object_key")
    op.drop_column("reviews", "results_json")
    op.alter_column(
        "reviews",
        "context_json",
        existing_type=postgresql.JSONB(),
        type_=postgresql.JSON(),
        existing_nullable=True,
    )
    op.alter_column(
        "reviews",
        "status",
        existing_type=sa.String(length=32),
        server_default="created",
        existing_nullable=False,
    )
