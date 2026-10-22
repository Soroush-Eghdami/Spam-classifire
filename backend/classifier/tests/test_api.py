"""API tests.

A tiny, cheap pipeline is trained once per test class so the suite does not
depend on the real multi-megabyte artifact and stays fast.  The point is to
test the web layer (validation, persistence, error shapes, pagination), not to
re-validate model quality.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import joblib
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITestCase
from sklearn.linear_model import LogisticRegression

from classifier.models import Prediction
from classifier.services import model_service
from ml.features import build_pipeline


def _training_rows() -> tuple[list[str], list[int]]:
    spam = [
        "Congratulations! You have won a free cash prize. Click here now to claim your money.",
        "URGENT: your account will expire. Verify your credit card immediately at www.example.com",
        "Buy cheap viagra and pills online, guaranteed discount, limited time offer!",
        "You are a winner! Claim your lottery cash reward today, act now.",
        "Free investment opportunity, guaranteed profit, click the link to double your money.",
        "Final notice: verify your account and send your password to unlock the prize.",
        "Hot deal! 100% free trial, order now and get a bonus cash reward.",
        "Unclaimed funds waiting for you, click immediately to receive your million dollars.",
    ]
    ham = [
        "Hi team, the project meeting is moved to Thursday at 10am. Let me know if that works.",
        "Please find attached the invoice for last month's consulting work. Thanks.",
        "Hey, are we still having lunch tomorrow? I can meet near the office at noon.",
        "The deployment succeeded and all tests are green. I will review the logs later.",
        "Thanks for your feedback on the draft. I updated section three and fixed the typo.",
        "Can you send me the notes from yesterday's call? I missed the last ten minutes.",
        "Reminder: the library books are due on Friday, and the study room is booked.",
        "Good morning, the train is delayed by fifteen minutes. I will be in the office soon.",
    ]
    texts = spam * 3 + ham * 3
    labels = [1] * len(spam) * 3 + [0] * len(ham) * 3
    return texts, labels


class ApiTestCase(APITestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)

        texts, labels = _training_rows()
        pipeline = build_pipeline(
            LogisticRegression(C=1.0, solver="liblinear", max_iter=1000),
            word_max_features=800,
            word_min_df=1,
            char_max_features=800,
            char_min_df=1,
            char_ngram_range=(3, 4),
            char_max_chars=300,
        )
        pipeline.fit(texts, labels)
        pipeline.threshold_ = 0.5
        pipeline.version_ = "test-1.0.0"

        cls.model_path = root / "pipeline.joblib"
        joblib.dump(pipeline, cls.model_path)
        cls.metrics_path = root / "metrics.json"
        cls.metrics_path.write_text(
            json.dumps(
                {
                    "version": "test-1.0.0",
                    "dataset": {"rows_clean": 48, "source": "C:/secret/path.csv"},
                    "model": {"family": "LogisticRegression", "threshold": 0.5},
                    "test": {"f1": 0.99},
                    "artifact": {"path": str(cls.model_path), "size_mb": 0.1},
                }
            ),
            encoding="utf-8",
        )

        cls.settings_override = override_settings(
            MODEL_PATH=cls.model_path,
            METRICS_PATH=cls.metrics_path,
        )
        cls.settings_override.enable()
        model_service.reset()

    @classmethod
    def tearDownClass(cls) -> None:
        model_service.reset()
        cls.settings_override.disable()
        cls.temp_dir.cleanup()
        super().tearDownClass()

    def _predict(self, text: str):
        return self.client.post("/api/predict/", {"text": text}, format="json")

    def test_predict_returns_documented_contract(self):
        response = self._predict(
            "Congratulations! You won a free cash prize, click here immediately to claim it."
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
        payload = response.data
        for key in (
            "id",
            "label",
            "spam_probability",
            "ham_probability",
            "threshold",
            "confidence",
            "top_words",
            "signals",
            "model_version",
            "text_length",
            "created_at",
        ):
            self.assertIn(key, payload)
        self.assertIn(payload["label"], {"spam", "ham"})
        self.assertGreaterEqual(payload["spam_probability"], 0.0)
        self.assertLessEqual(payload["spam_probability"], 1.0)
        self.assertEqual(Prediction.objects.count(), 1)
        stored = Prediction.objects.get()
        self.assertEqual(stored.source, Prediction.SOURCE_SINGLE)
        self.assertEqual(stored.model_version, "test-1.0.0")

    def test_predict_rejects_blank_text(self):
        response = self._predict("   \n  ")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)
        self.assertTrue(response.data["error"]["message"])

    def test_predict_rejects_oversized_text(self):
        response = self._predict("a" * 20_001)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("too long", response.data["error"]["message"].lower())

    def test_predict_requires_text_field(self):
        response = self.client.post("/api/predict/", {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_batch_csv_returns_results_and_persists_rows(self):
        csv_bytes = (
            'text,category\n'
            '"You have won a free prize, click now to claim cash!",promo\n'
            '"Hello team, the meeting is at 10am tomorrow.",work\n'
            ',empty\n'
        ).encode("utf-8")
        upload = SimpleUploadedFile("messages.csv", csv_bytes, content_type="text/csv")
        response = self.client.post(
            "/api/predict/batch/",
            {"file": upload, "text_column": "text"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
        payload = response.data
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["skipped_empty"], 1)
        self.assertEqual(payload["text_column"], "text")
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(payload["spam_count"] + payload["ham_count"], 2)
        self.assertEqual(Prediction.objects.filter(source=Prediction.SOURCE_BATCH).count(), 2)

    def test_batch_rejects_unknown_column(self):
        upload = SimpleUploadedFile("messages.csv", b"body\nhello\n", content_type="text/csv")
        response = self.client.post(
            "/api/predict/batch/",
            {"file": upload, "text_column": "text"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("columns", response.data["error"]["details"])

    def test_batch_rejects_unquoted_commas(self):
        csv_bytes = b"text\nfree money, click here now\nhello, lunch tomorrow\n"
        upload = SimpleUploadedFile("messages.csv", csv_bytes, content_type="text/csv")
        response = self.client.post(
            "/api/predict/batch/",
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("quotes", response.data["error"]["message"].lower())

    def test_batch_rejects_missing_file(self):
        response = self.client.post("/api/predict/batch/", {}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_history_pagination_and_filtering(self):
        for index in range(12):
            Prediction.objects.create(
                text_preview=f"message {index}",
                text_length=10,
                label=Prediction.LABEL_SPAM if index % 2 else Prediction.LABEL_HAM,
                spam_probability=0.9 if index % 2 else 0.1,
                threshold=0.5,
                model_version="test-1.0.0",
            )

        response = self.client.get("/api/history/?page_size=5")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 12)
        self.assertEqual(len(response.data["results"]), 5)
        self.assertIsNotNone(response.data["next"])

        response = self.client.get("/api/history/?label=spam&page_size=100")
        self.assertEqual(response.data["count"], 6)
        self.assertTrue(all(row["label"] == "spam" for row in response.data["results"]))

        response = self.client.get("/api/history/?label=invalid")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_model_info_hides_local_paths(self):
        response = self.client.get("/api/model-info/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["version"], "test-1.0.0")
        self.assertNotIn("source", response.data["dataset"])
        self.assertNotIn("path", response.data["artifact"])
        self.assertIn("runtime", response.data)

    def test_health_reports_loaded_model(self):
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
        self.assertTrue(response.data["model_loaded"])
        self.assertEqual(response.data["model_version"], "test-1.0.0")

    def test_stats_aggregates_predictions(self):
        self._predict("Free money, click now to claim your prize!")
        self._predict("Hello, the report is attached for your review.")
        response = self.client.get("/api/stats/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["total"], 2)
        self.assertEqual(response.data["spam"] + response.data["ham"], 2)
        self.assertIn("trend", response.data)

    def test_model_is_loaded_once_across_requests(self):
        model_service.reset()

        with mock.patch(
            "classifier.services.ml_predict.load_model",
            wraps=__import__("ml.predict", fromlist=["load_model"]).load_model,
        ) as load_mock:
            self._predict("Free prize, act now!")
            self._predict("Meeting notes are attached.")
            self.assertEqual(load_mock.call_count, 1)
