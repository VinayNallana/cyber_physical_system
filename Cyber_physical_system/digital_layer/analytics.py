# cyber_physical_system/digital_layer/analytics.py

import numpy as np
from typing import Dict
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Analytics engine for processing system data and generating insights.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize analytics engine.

        Args:
            window_size: Size of the sliding window for metrics
        """
        self.window_size = window_size
        self.detection_metrics = deque(maxlen=window_size)
        self.anomaly_metrics = deque(maxlen=window_size)
        self.sensor_metrics = deque(maxlen=window_size)
        self.performance_metrics = deque(maxlen=window_size)
        logger.info("Analytics engine initialized")

    def update_detection_metrics(self, detection_data: Dict) -> None:
        """
        Update detection metrics.

        Args:
            detection_data: Detection results
        """
        try:
            metric = {
                "timestamp": detection_data.get("timestamp", datetime.now()),
                "num_detections": len(detection_data.get("detections", [])),
                "inference_time_ms": detection_data.get(
                    "inference_time_ms", 0
                ),
                "high_confidence_count": sum(
                    1
                    for d in detection_data.get("detections", [])
                    if d.get("confidence", 0) >= 0.7
                ),
            }
            self.detection_metrics.append(metric)
        except Exception as e:
            logger.error(f"Error updating detection metrics: {e}")

    def update_anomaly_metrics(self, anomaly_data: Dict) -> None:
        """
        Update anomaly detection metrics.

        Args:
            anomaly_data: Anomaly detection results
        """
        try:
            metric = {
                "timestamp": anomaly_data.get("timestamp", datetime.now()),
                "anomaly_score": anomaly_data.get("anomaly_score", 0),
                "is_anomalous": anomaly_data.get("is_anomalous", False),
                "severity": anomaly_data.get("severity_level", "low"),
                "inference_time_ms": anomaly_data.get("inference_time_ms", 0),
            }
            self.anomaly_metrics.append(metric)
        except Exception as e:
            logger.error(f"Error updating anomaly metrics: {e}")

    def update_sensor_metrics(self, sensor_data: Dict) -> None:
        """
        Update sensor metrics.

        Args:
            sensor_data: Sensor reading data
        """
        try:
            metric = {
                "timestamp": sensor_data.get("timestamp", datetime.now()),
                "sensor_type": sensor_data.get("sensor_type", "unknown"),
                "value": sensor_data.get("value", 0),
                "is_normal": sensor_data.get("is_normal", True),
            }
            self.sensor_metrics.append(metric)
        except Exception as e:
            logger.error(f"Error updating sensor metrics: {e}")

    def get_detection_statistics(self) -> Dict:
        """Get detection statistics."""
        if not self.detection_metrics:
            return {
                "avg_detections": 0,
                "avg_inference_time_ms": 0,
                "total_processed": 0,
                "avg_high_confidence": 0,
            }

        metrics_list = list(self.detection_metrics)
        return {
            "avg_detections": np.mean(
                [m["num_detections"] for m in metrics_list]
            ),
            "avg_inference_time_ms": np.mean(
                [m["inference_time_ms"] for m in metrics_list]
            ),
            "total_processed": len(metrics_list),
            "avg_high_confidence": np.mean(
                [m["high_confidence_count"] for m in metrics_list]
            ),
        }

    def get_anomaly_statistics(self) -> Dict:
        """Get anomaly detection statistics."""
        if not self.anomaly_metrics:
            return {
                "avg_anomaly_score": 0,
                "anomaly_rate": 0,
                "total_processed": 0,
                "severity_distribution": {},
            }

        metrics_list = list(self.anomaly_metrics)
        severity_dist = {}
        for m in metrics_list:
            severity = m["severity"]
            severity_dist[severity] = severity_dist.get(severity, 0) + 1

        return {
            "avg_anomaly_score": np.mean(
                [m["anomaly_score"] for m in metrics_list]
            ),
            "anomaly_rate": sum(1 for m in metrics_list if m["is_anomalous"])
            / len(metrics_list),
            "total_processed": len(metrics_list),
            "severity_distribution": severity_dist,
        }

    def get_system_health(self) -> Dict:
        """
        Get overall system health assessment.

        Returns:
            System health status
        """
        detection_stats = self.get_detection_statistics()
        anomaly_stats = self.get_anomaly_statistics()

        # Calculate health score (0-100)
        health_score = 100

        # Reduce score based on anomaly rate
        if anomaly_stats["anomaly_rate"] > 0.5:
            health_score -= 30
        elif anomaly_stats["anomaly_rate"] > 0.2:
            health_score -= 15

        # Reduce score based on inference time
        if detection_stats["avg_inference_time_ms"] > 100:
            health_score -= 20
        elif detection_stats["avg_inference_time_ms"] > 50:
            health_score -= 10

        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "poor"

        return {
            "health_score": max(0, health_score),
            "status": status,
            "timestamp": datetime.now(),
            "metrics": {
                "detection": detection_stats,
                "anomaly": anomaly_stats,
            },
        }

    def generate_report(self) -> Dict:
        """
        Generate comprehensive analytics report.

        Returns:
            Analytics report
        """
        return {
            "timestamp": datetime.now(),
            "detection_statistics": self.get_detection_statistics(),
            "anomaly_statistics": self.get_anomaly_statistics(),
            "system_health": self.get_system_health(),
            "data_points_analyzed": {
                "detections": len(self.detection_metrics),
                "anomalies": len(self.anomaly_metrics),
                "sensors": len(self.sensor_metrics),
            },
        }
