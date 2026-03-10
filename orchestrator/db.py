import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

# Get Database URL from environment or fallback to in-memory sqlite for
# local testing/dev
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///:memory:")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"

    # Only use schema if it's not SQLite, because SQLite doesn't support
    # schemas like Postgres
    if not str(DATABASE_URL).startswith("sqlite"):
        __table_args__ = {'schema': 'agent_system'}

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now())
    hypothesis_text = Column(Text, nullable=True)
    generated_code = Column(Text, nullable=True)
    status = Column(String(20), default='pending')
    k8s_job_name = Column(String(100), nullable=True)
    mlflow_run_id = Column(String(100), nullable=True)
    best_cv_score = Column(Float, nullable=True)
    error_log = Column(Text, nullable=True)


from sqlalchemy import text

def init_db():
    # If using sqlite memory, we don't have schemas, so we need to handle that gracefully
    # For actual Postgres, the init.sql creates the schema.
    if not str(engine.url).startswith("sqlite"):
        with engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS agent_system"))
            conn.commit()
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
