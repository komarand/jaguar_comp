import os
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def _get_k8s_client():
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.BatchV1Api(), client.CoreV1Api()


def create_experiment_job(
        job_id: int,
        python_code: str,
        namespace: str = "default"):
    batch_api, core_api = _get_k8s_client()

    configmap_name = f"code-{job_id}"
    job_name = f"ml-experiment-{job_id}"

    # 1. Create ConfigMap
    configmap = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=configmap_name),
        data={"agent_code.py": python_code}
    )

    try:
        # Dry-run
        core_api.create_namespaced_config_map(
            namespace=namespace, body=configmap, dry_run="All"
        )
        # Actual creation
        core_api.create_namespaced_config_map(
            namespace=namespace, body=configmap
        )
    except ApiException as e:
        if e.status == 409:  # Already exists
            pass
        else:
            raise e

    # 2. Parse Job Template and create Job
    # We could read from infra/k8s/job_template.yaml but for simplicity we build it dynamically
    # based on the requested specification.
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "worker",
                            "image": "your-registry/ml-worker:latest",
                            "volumeMounts": [
                                {
                                    "name": "code-volume",
                                    "mountPath": "/app/agent_code.py",
                                    "subPath": "agent_code.py"
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": "1"
                                }
                            }
                        }
                    ],
                    "volumes": [
                        {
                            "name": "code-volume",
                            "configMap": {
                                "name": configmap_name
                            }
                        }
                    ],
                    "restartPolicy": "Never"
                }
            }
        }
    }

    try:
        # Dry-run
        batch_api.create_namespaced_job(
            namespace=namespace, body=job_manifest, dry_run="All"
        )
        # Actual creation
        batch_api.create_namespaced_job(
            namespace=namespace, body=job_manifest
        )
    except ApiException as e:
        if e.status == 409:  # Already exists
            pass
        else:
            raise e

    return job_name


def get_job_status(job_name: str, namespace: str = "default"):
    batch_api, _ = _get_k8s_client()
    try:
        job = batch_api.read_namespaced_job_status(
            name=job_name, namespace=namespace)
        if job.status.active:
            return "running"
        elif job.status.succeeded:
            return "success"
        elif job.status.failed:
            return "failed"
        return "pending"
    except ApiException as e:
        if e.status == 404:
            return "not_found"
        raise e
