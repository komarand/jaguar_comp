import unittest
from unittest.mock import patch, MagicMock
from kubernetes.client.rest import ApiException
from orchestrator.k8s_client import create_experiment_job, get_job_status


class TestK8sClient(unittest.TestCase):

    @patch('orchestrator.k8s_client._get_k8s_client')
    def test_create_experiment_job(self, mock_get_client):
        mock_batch_api = MagicMock()
        mock_core_api = MagicMock()
        mock_get_client.return_value = (mock_batch_api, mock_core_api)

        job_id = 1
        python_code = "print('hello world')"
        namespace = "default"

        job_name = create_experiment_job(job_id, python_code, namespace)

        self.assertEqual(job_name, f"ml-experiment-{job_id}")

        # Check ConfigMap creation
        # Expect dry-run and actual call
        self.assertEqual(
            mock_core_api.create_namespaced_config_map.call_count, 2)

        # Check Job creation
        # Expect dry-run and actual call
        self.assertEqual(mock_batch_api.create_namespaced_job.call_count, 2)

    @patch('orchestrator.k8s_client._get_k8s_client')
    def test_get_job_status_active(self, mock_get_client):
        mock_batch_api = MagicMock()
        mock_get_client.return_value = (mock_batch_api, MagicMock())

        mock_job = MagicMock()
        mock_job.status.active = True
        mock_job.status.succeeded = False
        mock_job.status.failed = False

        mock_batch_api.read_namespaced_job_status.return_value = mock_job

        status = get_job_status("ml-experiment-1")
        self.assertEqual(status, "running")

    @patch('orchestrator.k8s_client._get_k8s_client')
    def test_get_job_status_success(self, mock_get_client):
        mock_batch_api = MagicMock()
        mock_get_client.return_value = (mock_batch_api, MagicMock())

        mock_job = MagicMock()
        mock_job.status.active = False
        mock_job.status.succeeded = True
        mock_job.status.failed = False

        mock_batch_api.read_namespaced_job_status.return_value = mock_job

        status = get_job_status("ml-experiment-1")
        self.assertEqual(status, "success")


if __name__ == '__main__':
    unittest.main()
