"""add extraction fields

Revision ID: 0003_add_extraction_fields
Revises: 0002_review_fields
Create Date: 2024-01-03 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0003_add_extraction_fields"
down_revision = "0002_review_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("reviews", sa.Column("extracted_text", sa.Text(), nullable=True))
    op.add_column("reviews", sa.Column("extracted_meta", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("reviews", "extracted_meta")
    op.drop_column("reviews", "extracted_text")
