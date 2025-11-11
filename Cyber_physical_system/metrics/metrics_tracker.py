# cyber_physical_system/metrics/metrics_tracker.py

import numpy as np
from typing import Dict
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Comprehensive metrics tracking system for monitoring model and system performance.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics tracker.

        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        # Detection metrics
        self.detection_metrics = {
            "total_detections": 0,
            "detections_per_frame": deque(maxlen=window_size),
            "inference_times": deque(maxlen=window_size),
            "confidence_scores": deque(maxlen=window_size),
            "class_distribution": {},
        }

        # Anomaly metrics
        self.anomaly_metrics = {
            "total_predictions": 0,
            "anomaly_count": 0,
            "anomaly_scores": deque(maxlen=window_size),
            "severity_counts": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0,
            },
            "inference_times": deque(maxlen=window_size),
        }

        # System metrics
        self.system_metrics = {
            "uptime_seconds": 0,
            "start_time": None,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": deque(maxlen=100),
            "warnings": deque(maxlen=100),
        }

        # Performance metrics
        self.performance_metrics = {
            "cpu_usage": deque(maxlen=window_size),
            "memory_usage": deque(maxlen=window_size),
            "fps": deque(maxlen=window_size),
        }

        logger.info("Metrics tracker initialized")

    def start_tracking(self) -> None:
        """Start tracking system uptime."""
        self.system_metrics["start_time"] = datetime.now()
        logger.info("Metrics tracking started")

    def update_detection_metrics(self, detection_result: Dict) -> None:
        """
        Update object detection metrics.

        Args:
            detection_result: Detection results dictionary
        """
        try:
            if not detection_result.get("success", False):
                return

            detections = detection_result.get("detections", [])
            inference_time = detection_result.get("inference_time_ms", 0)

            self.detection_metrics["total_detections"] += len(detections)
            self.detection_metrics["detections_per_frame"].append(
                len(detections)
            )
            self.detection_metrics["inference_times"].append(inference_time)

            # Update confidence scores and class distribution
            for detection in detections:
                confidence = detection.get("confidence", 0)
                class_name = detection.get("class_name", "unknown")

                self.detection_metrics["confidence_scores"].append(confidence)

                if class_name in self.detection_metrics["class_distribution"]:
                    self.detection_metrics["class_distribution"][
                        class_name
                    ] += 1
                else:
                    self.detection_metrics["class_distribution"][
                        class_name
                    ] = 1

        except Exception as e:
            logger.error(f"Error updating detection metrics: {e}")
            self.log_error(str(e))

    def update_anomaly_metrics(self, anomaly_result: Dict) -> None:
        """
        Update anomaly detection metrics.

        Args:
            anomaly_result: Anomaly detection results dictionary
        """
        try:
            if not anomaly_result.get("success", False):
                return
            anomaly_score = anomaly_result.get("anomaly_score", 0)
            is_anomalous = anomaly_result.get("is_anomalous", False)
            severity = anomaly_result.get("severity_level", "low")
            inference_time = anomaly_result.get("inference_time_ms", 0)

            self.anomaly_metrics["total_predictions"] += 1
            self.anomaly_metrics["anomaly_scores"].append(anomaly_score)
            self.anomaly_metrics["inference_times"].append(inference_time)

            if is_anomalous:
                self.anomaly_metrics["anomaly_count"] += 1
                self.anomaly_metrics["severity_counts"][severity] += 1

        except Exception as e:
            logger.error(f"Error updating anomaly metrics: {e}")
            self.log_error(str(e))

    def update_system_metrics(self, metric_type: str, value: float) -> None:
        """
        Update system performance metrics.

        Args:
            metric_type: Type of metric ('cpu', 'memory', 'fps')
            value: Metric value
        """
        try:
            if metric_type == "cpu":
                self.performance_metrics["cpu_usage"].append(value)
            elif metric_type == "memory":
                self.performance_metrics["memory_usage"].append(value)
            elif metric_type == "fps":
                self.performance_metrics["fps"].append(value)
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def log_error(self, error_message: str) -> None:
        """Log an error."""
        self.system_metrics["errors"].append(
            {"timestamp": datetime.now(), "message": error_message}
        )

    def log_warning(self, warning_message: str) -> None:
        """Log a warning."""
        self.system_metrics["warnings"].append(
            {"timestamp": datetime.now(), "message": warning_message}
        )

    def get_detection_summary(self) -> Dict:
        """Get detection metrics summary."""
        if not self.detection_metrics["detections_per_frame"]:
            return {
                "total_detections": 0,
                "avg_detections_per_frame": 0,
                "avg_inference_time_ms": 0,
                "avg_confidence": 0,
                "class_distribution": {},
            }

        return {
            "total_detections": self.detection_metrics["total_detections"],
            "avg_detections_per_frame": np.mean(
                list(self.detection_metrics["detections_per_frame"])
            ),
            "avg_inference_time_ms": np.mean(
                list(self.detection_metrics["inference_times"])
            ),
            "max_inference_time_ms": np.max(
                list(self.detection_metrics["inference_times"])
            ),
            "avg_confidence": (
                np.mean(list(self.detection_metrics["confidence_scores"]))
                if self.detection_metrics["confidence_scores"]
                else 0
            ),
            "class_distribution": self.detection_metrics["class_distribution"],
        }

    def get_anomaly_summary(self) -> Dict:
        """Get anomaly detection metrics summary."""
        if self.anomaly_metrics["total_predictions"] == 0:
            return {
                "total_predictions": 0,
                "anomaly_rate": 0,
                "avg_anomaly_score": 0,
                "avg_inference_time_ms": 0,
                "severity_distribution": {},
            }

        return {
            "total_predictions": self.anomaly_metrics["total_predictions"],
            "anomaly_count": self.anomaly_metrics["anomaly_count"],
            "anomaly_rate": self.anomaly_metrics["anomaly_count"]
            / self.anomaly_metrics["total_predictions"],
            "avg_anomaly_score": np.mean(
                list(self.anomaly_metrics["anomaly_scores"])
            ),
            "max_anomaly_score": np.max(
                list(self.anomaly_metrics["anomaly_scores"])
            ),
            "avg_inference_time_ms": np.mean(
                list(self.anomaly_metrics["inference_times"])
            ),
            "severity_distribution": self.anomaly_metrics["severity_counts"],
        }

    def get_system_summary(self) -> Dict:
        """Get system metrics summary."""
        uptime = 0
        if self.system_metrics["start_time"]:
            uptime = (
                datetime.now() - self.system_metrics["start_time"]
            ).total_seconds()

        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "messages_sent": self.system_metrics["messages_sent"],
            "messages_received": self.system_metrics["messages_received"],
            "error_count": len(self.system_metrics["errors"]),
            "warning_count": len(self.system_metrics["warnings"]),
            "recent_errors": list(self.system_metrics["errors"])[-5:],
            "recent_warnings": list(self.system_metrics["warnings"])[-5:],
        }

    def get_performance_summary(self) -> Dict:
        """Get performance metrics summary."""
        if not self.performance_metrics["cpu_usage"]:
            return {"avg_cpu_usage": 0, "avg_memory_usage": 0, "avg_fps": 0}

        return {
            "avg_cpu_usage": np.mean(
                list(self.performance_metrics["cpu_usage"])
            ),
            "max_cpu_usage": (
                np.max(list(self.performance_metrics["cpu_usage"]))
                if self.performance_metrics["cpu_usage"]
                else 0
            ),
            "avg_memory_usage": np.mean(
                list(self.performance_metrics["memory_usage"])
            ),
            "max_memory_usage": (
                np.max(list(self.performance_metrics["memory_usage"]))
                if self.performance_metrics["memory_usage"]
                else 0
            ),
            "avg_fps": np.mean(list(self.performance_metrics["fps"])),
            "min_fps": (
                np.min(list(self.performance_metrics["fps"]))
                if self.performance_metrics["fps"]
                else 0
            ),
        }

    def get_comprehensive_report(self) -> Dict:
        """Get comprehensive metrics report."""
        return {
            "timestamp": datetime.now(),
            "detection_metrics": self.get_detection_summary(),
            "anomaly_metrics": self.get_anomaly_summary(),
            "system_metrics": self.get_system_summary(),
            "performance_metrics": self.get_performance_summary(),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.detection_metrics = {
            "total_detections": 0,
            "detections_per_frame": deque(maxlen=self.window_size),
            "inference_times": deque(maxlen=self.window_size),
            "confidence_scores": deque(maxlen=self.window_size),
            "class_distribution": {},
        }

        self.anomaly_metrics = {
            "total_predictions": 0,
            "anomaly_count": 0,
            "anomaly_scores": deque(maxlen=self.window_size),
            "severity_counts": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0,
            },
            "inference_times": deque(maxlen=self.window_size),
        }

        logger.info("Metrics reset")
