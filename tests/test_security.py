"""
Security tests for ClimateVision API.

Tests input validation, sanitization, and security controls.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climatevision.security import (
    validate_payload_size,
    validate_bbox,
    validate_file_upload,
    sanitize_string_input,
    SecurityConfig,
    RateLimiter,
    detect_adversarial_input,
    validate_model_output,
    InputAnomalyDetector,
    PipelineGuard,
)


class TestPayloadValidation:
    """Test payload size validation."""

    def test_valid_payload(self):
        data = b"x" * 1000
        is_valid, error = validate_payload_size(data, max_size=2000)
        assert is_valid
        assert error == ""

    def test_oversized_payload(self):
        data = b"x" * 10000
        is_valid, error = validate_payload_size(data, max_size=1000)
        assert not is_valid
        assert "exceeds maximum" in error


class TestBboxValidation:
    """Test bounding box validation."""

    def test_valid_bbox(self):
        bbox = [-60.0, -15.0, -45.0, -5.0]
        is_valid, error = validate_bbox(bbox)
        assert is_valid
        assert error == ""

    def test_invalid_longitude(self):
        bbox = [200.0, 10.0, 30.0, 40.0]
        is_valid, error = validate_bbox(bbox)
        assert not is_valid
        assert "longitude" in error.lower()

    def test_invalid_latitude(self):
        bbox = [10.0, 100.0, 30.0, 40.0]
        is_valid, error = validate_bbox(bbox)
        assert not is_valid
        assert "latitude" in error.lower()

    def test_wrong_order_longitude(self):
        bbox = [30.0, 10.0, 20.0, 40.0]
        is_valid, error = validate_bbox(bbox)
        assert not is_valid
        assert "West" in error

    def test_wrong_order_latitude(self):
        bbox = [10.0, 50.0, 30.0, 40.0]
        is_valid, error = validate_bbox(bbox)
        assert not is_valid
        assert "South" in error

    def test_too_large_area(self):
        bbox = [-180.0, -90.0, 180.0, 90.0]
        is_valid, error = validate_bbox(bbox, max_area=100.0)
        assert not is_valid
        assert "area" in error.lower()

    def test_wrong_element_count(self):
        bbox = [10.0, 20.0, 30.0]
        is_valid, error = validate_bbox(bbox)
        assert not is_valid
        assert "exactly 4" in error


class TestFileUploadValidation:
    """Test file upload validation."""

    def test_valid_png(self):
        content = b"\x89PNG\r\n\x1a\n" + b"x" * 100
        is_valid, error = validate_file_upload(content, "image.png")
        assert is_valid
        assert error == ""

    def test_valid_tiff(self):
        content = b"II*\x00" + b"x" * 100
        is_valid, error = validate_file_upload(content, "satellite.tif")
        assert is_valid
        assert error == ""

    def test_invalid_extension(self):
        content = b"\x89PNG\r\n\x1a\n" + b"x" * 100
        is_valid, error = validate_file_upload(content, "malware.exe")
        assert not is_valid
        assert "not allowed" in error

    def test_path_traversal(self):
        content = b"\x89PNG\r\n\x1a\n" + b"x" * 100
        is_valid, error = validate_file_upload(content, "../../../etc/passwd")
        assert not is_valid
        assert "path traversal" in error.lower()

    def test_extension_mismatch(self):
        content = b"\x89PNG\r\n\x1a\n" + b"x" * 100
        is_valid, error = validate_file_upload(content, "image.jpg")
        assert not is_valid
        assert "does not match" in error

    def test_filename_too_long(self):
        content = b"\x89PNG\r\n\x1a\n" + b"x" * 100
        filename = "a" * 300 + ".png"
        is_valid, error = validate_file_upload(content, filename)
        assert not is_valid
        assert "too long" in error


class TestStringSanitization:
    """Test string input sanitization."""

    def test_normal_string(self):
        result, warnings = sanitize_string_input("Hello World")
        assert "Hello" in result
        assert len(warnings) == 0 or "sanitized" in warnings[0].lower()

    def test_sql_injection(self):
        result, warnings = sanitize_string_input("'; DROP TABLE users; --")
        assert "DROP TABLE" not in result
        assert any("blocked" in w.lower() or "sanitized" in w.lower() for w in warnings)

    def test_xss_script(self):
        result, warnings = sanitize_string_input("<script>alert(1)</script>")
        assert "<script" not in result
        assert len(warnings) > 0

    def test_path_traversal(self):
        result, warnings = sanitize_string_input("../../../etc/passwd")
        assert "../" not in result

    def test_truncation(self):
        long_string = "a" * 2000
        result, warnings = sanitize_string_input(long_string, max_length=100)
        assert len(result) <= 100
        assert any("truncated" in w.lower() for w in warnings)


class TestRateLimiter:
    """Test rate limiting."""

    def test_allows_under_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("test_key")

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            assert limiter.is_allowed("test_key")
        assert not limiter.is_allowed("test_key")

    def test_separate_keys(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("key1")
        assert limiter.is_allowed("key1")
        assert not limiter.is_allowed("key1")
        assert limiter.is_allowed("key2")  # Different key

    def test_remaining_count(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.get_remaining("key") == 5
        limiter.is_allowed("key")
        assert limiter.get_remaining("key") == 4


class TestAdversarialDetection:
    """Test adversarial input detection."""

    def test_normal_image(self):
        image = np.random.randn(4, 256, 256).astype(np.float32)
        result = detect_adversarial_input(image)
        assert not result.is_anomalous
        assert result.anomaly_score < 0.5

    def test_uniform_image(self):
        image = np.ones((4, 256, 256), dtype=np.float32)
        result = detect_adversarial_input(image)
        assert result.is_anomalous
        assert "uniform" in str(result.details).lower() or result.anomaly_score > 0.3

    def test_nan_values(self):
        image = np.random.randn(4, 256, 256).astype(np.float32)
        image[0, 100, 100] = np.nan
        result = detect_adversarial_input(image)
        assert result.is_anomalous
        assert result.anomaly_score >= 0.5

    def test_inf_values(self):
        image = np.random.randn(4, 256, 256).astype(np.float32)
        image[0, 100, 100] = np.inf
        result = detect_adversarial_input(image)
        assert result.is_anomalous

    def test_out_of_range(self):
        image = np.random.randn(4, 256, 256).astype(np.float32) * 100
        result = detect_adversarial_input(image)
        assert result.anomaly_score > 0


class TestOutputValidation:
    """Test model output validation."""

    def test_valid_output(self):
        predictions = np.random.randint(0, 2, (256, 256))
        result = validate_model_output(predictions, n_classes=2)
        assert result.is_valid
        assert result.confidence > 0.5

    def test_invalid_class_values(self):
        predictions = np.array([[0, 1, 5, 10]])
        result = validate_model_output(predictions, n_classes=2)
        assert not result.is_valid or len(result.issues) > 0

    def test_single_class_domination(self):
        predictions = np.ones((256, 256), dtype=np.int32)
        result = validate_model_output(predictions, n_classes=2)
        assert len(result.issues) > 0
        assert any("dominates" in issue.lower() for issue in result.issues)

    def test_nan_in_predictions(self):
        predictions = np.array([[0.0, 1.0, np.nan]])
        result = validate_model_output(predictions, n_classes=2)
        assert not result.is_valid


class TestPipelineGuard:
    """Test complete pipeline guard."""

    def test_blocks_adversarial(self):
        guard = PipelineGuard()
        adversarial_image = np.ones((4, 256, 256), dtype=np.float32) * 0.5

        result = guard.check_input(adversarial_image)
        # Uniform image should be flagged
        assert result.anomaly_score > 0

    def test_passes_normal_image(self):
        guard = PipelineGuard()
        normal_image = np.random.randn(4, 256, 256).astype(np.float32)

        result = guard.check_input(normal_image)
        assert not result.is_anomalous


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
