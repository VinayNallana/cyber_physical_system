# cyber_physical_system/physical_layer/sensors.py

import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Optional, Tuple
from threading import Lock
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraSensor:
    """
    Simulates an industrial camera sensor providing image streams.
    """

    def __init__(
        self,
        sensor_id: str,
        fps: int = 30,
        resolution: Tuple[int, int] = (640, 480),
    ):
        """
        Initialize camera sensor.

        Args:
            sensor_id: Unique identifier for the sensor
            fps: Frames per second
            resolution: Image resolution (width, height)
        """
        self.sensor_id = sensor_id
        self.fps = fps
        self.resolution = resolution
        self.is_active = False
        self.health_status = "healthy"
        self.lock = Lock()
        self.frame_count = 0
            logger.info(
                f"Camera sensor {sensor_id} initialized at {fps} FPS, "
                f"resolution {resolution}"
            )

    def start(self) -> None:
        """Start the camera sensor."""
        with self.lock:
            self.is_active = True
            logger.info(f"Camera sensor {self.sensor_id} started")

    def stop(self) -> None:
        """Stop the camera sensor."""
        with self.lock:
            self.is_active = False
            logger.info(f"Camera sensor {self.sensor_id} stopped")

    def capture_frame(
        self, image_path: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Capture a frame from the camera or load from file.

        Args:
            image_path: Optional path to image file to simulate capture

        Returns:
            Dictionary containing frame data and metadata
        """
        if not self.is_active:
            logger.warning(f"Camera sensor {self.sensor_id} is not active")
            return None

        try:
            if image_path:
                frame = cv2.imread(image_path)
                if frame is None:
                    logger.error(f"Failed to load image from {image_path}")
                    return None
            else:
                # Simulate frame capture with random noise
                frame = np.random.randint(
                    0, 255, (*self.resolution[::-1], 3), dtype=np.uint8
                )

            timestamp = datetime.now()
            self.frame_count += 1

            return {
                "sensor_id": self.sensor_id,
                "frame": frame,
                "timestamp": timestamp,
                "frame_number": self.frame_count,
                "resolution": self.resolution,
                "health_status": self.health_status,
            }
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            self.health_status = "error"
            return None

    def get_health_status(self) -> str:
        """Get sensor health status."""
        return self.health_status

    def perform_health_check(self) -> bool:
        """Perform sensor health check."""
        # Simulate health check
        self.health_status = (
            "healthy" if np.random.random() > 0.01 else "warning"
        )
        return self.health_status == "healthy"


class EnvironmentalSensor:
    """
    Simulates environmental sensors (temperature, pressure, vibration).
    """

    def __init__(
        self,
        sensor_id: str,
        sensor_type: str,
        unit: str,
        normal_range: Tuple[float, float],
    ):
        """
        Initialize environmental sensor.

        Args:
            sensor_id: Unique identifier
            sensor_type: Type of sensor (temperature, pressure, vibration)
            unit: Measurement unit
            normal_range: Normal operating range (min, max)
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.unit = unit
        self.normal_range = normal_range
        self.is_active = False
        self.health_status = "healthy"
        self.lock = Lock()
        logger.info(
            f"{sensor_type} sensor {sensor_id} initialized with "
            f"range {normal_range} {unit}"
        )

    def start(self) -> None:
        """Start the sensor."""
        with self.lock:
            self.is_active = True
            logger.info(f"Environmental sensor {self.sensor_id} started")

    def stop(self) -> None:
        """Stop the sensor."""
        with self.lock:
            self.is_active = False
            logger.info(f"Environmental sensor {self.sensor_id} stopped")

    def read_value(self) -> Optional[Dict]:
        """
        Read sensor value.

        Returns:
            Dictionary containing sensor reading and metadata
        """
        if not self.is_active:
            logger.warning(f"Sensor {self.sensor_id} is not active")
            return None

        try:
            # Simulate sensor reading with some noise
            min_val, max_val = self.normal_range
            value = np.random.uniform(min_val * 0.9, max_val * 1.1)

            # Determine if value is within normal range
            is_normal = min_val <= value <= max_val

            timestamp = datetime.now()

            return {
                "sensor_id": self.sensor_id,
                "sensor_type": self.sensor_type,
                "value": value,
                "unit": self.unit,
                "timestamp": timestamp,
                "is_normal": is_normal,
                "health_status": self.health_status,
            }
        except Exception as e:
            logger.error(f"Error reading sensor value: {e}")
            self.health_status = "error"
            return None

    def get_health_status(self) -> str:
        """Get sensor health status."""
        return self.health_status

    def perform_health_check(self) -> bool:
        """Perform sensor health check."""
        self.health_status = (
            "healthy" if np.random.random() > 0.01 else "warning"
        )
        return self.health_status == "healthy"
