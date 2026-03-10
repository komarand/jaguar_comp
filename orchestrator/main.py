import time
import logging
import os
from orchestrator.db import init_db, SessionLocal, Experiment
from orchestrator.k8s_client import create_experiment_job, get_job_status
from orchestrator.agent import generate_hypothesis_and_code

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))


def control_loop():
    logger.info("Starting Orchestrator Control Loop...")
    init_db()

    while True:
        db = None
        try:
            db = SessionLocal()

            # 1. Update statuses of active jobs (running or pending)
            active_experiments = db.query(Experiment).filter(
                Experiment.status.in_(["running", "pending"])).all()
            for exp in active_experiments:
                if exp.k8s_job_name:
                    current_status = get_job_status(exp.k8s_job_name)
                    if current_status != exp.status:
                        exp.status = current_status
                        logger.info(
                            f"Experiment {
                                exp.id} status changed to {current_status}")
            db.commit()

            # 2. Check current active jobs limit
            current_active_count = db.query(Experiment).filter(
                Experiment.status.in_(["running", "pending"])).count()

            if current_active_count < MAX_CONCURRENT_JOBS:
                logger.info(
                    f"Active jobs ({current_active_count}) < Limit ({MAX_CONCURRENT_JOBS}). Generating new hypothesis.")
                new_exp = None
                try:
                    # Request new hypothesis
                    hypothesis, code = generate_hypothesis_and_code()

                    # Create new experiment
                    new_exp = Experiment(
                        hypothesis_text=hypothesis,
                        generated_code=code,
                        status="pending"
                    )
                    db.add(new_exp)
                    db.commit()
                    db.refresh(new_exp)

                    # Launch Job
                    job_name = create_experiment_job(new_exp.id, code)
                    new_exp.k8s_job_name = job_name
                    new_exp.status = "running"
                    db.commit()

                    logger.info(
                        f"Successfully launched new experiment Job: {job_name}")
                except Exception as e:
                    logger.error(
                        f"Failed to generate/launch new experiment: {e}")
                    if new_exp and new_exp.id:
                        # Mark as failed so it doesn't block slots forever
                        new_exp.status = "failed"
                        new_exp.error_log = str(e)
                        db.commit()
                    else:
                        db.rollback()
            else:
                logger.info("At max capacity. Waiting...")

        except Exception as e:
            logger.error(f"Error in control loop: {e}")
        finally:
            if db:
                db.close()

        time.sleep(30)  # Poll every 30 seconds


if __name__ == "__main__":
    control_loop()
