"""Regenerate database/schema.sql (PostgreSQL DDL) from the SQLAlchemy models."""
from pathlib import Path
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable, CreateIndex
from database.models import metadata

def main():
    out = ["-- PostgreSQL schema for customer_retention_db",
           "-- Generated from database/models.py via `python -m database.export_schema`.",
           "-- Create the database/user first (see README), then:  psql -d customer_retention_db -f database/schema.sql\n"]
    for t in metadata.sorted_tables:
        out.append(str(CreateTable(t, if_not_exists=True).compile(dialect=postgresql.dialect())).strip() + ";\n")
    out.append("CREATE INDEX IF NOT EXISTS ix_churn_pred_prob ON churn_predictions (churn_probability DESC);")
    out.append("CREATE INDEX IF NOT EXISTS ix_segments_cluster ON customer_segments (cluster_id);")
    Path(__file__).with_name("schema.sql").write_text("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
