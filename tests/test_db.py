import unittest
from unittest.mock import patch, MagicMock
from orchestrator.db import init_db, SessionLocal, Experiment, engine


class TestDB(unittest.TestCase):
    def setUp(self):
        # Using sqlite memory which is setup in db.py fallback
        init_db()
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()
        # Clean up all tables after tests
        from orchestrator.db import Base
        Base.metadata.drop_all(bind=engine)

    def test_create_experiment(self):
        exp = Experiment(
            hypothesis_text="test hypothesis",
            generated_code="print('hello')",
            status="pending"
        )
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)

        self.assertIsNotNone(exp.id)
        self.assertEqual(exp.hypothesis_text, "test hypothesis")
        self.assertEqual(exp.status, "pending")

    def test_update_experiment_status(self):
        exp = Experiment(
            hypothesis_text="test",
            status="running",
            k8s_job_name="ml-job-1"
        )
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)

        # Update
        exp.status = "success"
        self.db.commit()

        updated_exp = self.db.query(Experiment).filter(
            Experiment.id == exp.id).first()
        self.assertEqual(updated_exp.status, "success")


if __name__ == '__main__':
    unittest.main()
