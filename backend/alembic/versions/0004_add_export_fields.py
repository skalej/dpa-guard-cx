"""add export fields

Revision ID: 0004_add_export_fields
Revises: 0003_add_extraction_fields
Create Date: 2024-01-04 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0004_add_export_fields"
down_revision = "0003_add_extraction_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("reviews", sa.Column("export_object_key", sa.Text(), nullable=True))
    op.add_column(
        "reviews", sa.Column("export_created_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("reviews", "export_created_at")
    op.drop_column("reviews", "export_object_key")
