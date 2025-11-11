# cyber_physical_system/models/schemas.py

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any, List


class SensorDataMessage(BaseModel):
    """Message schema for sensor data."""

    message_id: str
    sensor_id: str
    sensor_type: str
    timestamp: datetime
    data: Dict[str, Any]
    sequence_number: int

    class Config:
        arbitrary_types_allowed = True


class DetectionMessage(BaseModel):
    """Message schema for object detection results."""

    message_id: str
    timestamp: datetime
    image_id: str
    detections: List[Dict[str, Any]]
    inference_time_ms: float
    sequence_number: int

    class Config:
        arbitrary_types_allowed = True


class AnomalyMessage(BaseModel):
    """Message schema for anomaly detection results."""

    message_id: str
    timestamp: datetime
    image_id: str
    anomaly_score: float
    is_anomalous: bool
    severity_level: str
    inference_time_ms: float
    sequence_number: int

    class Config:
        arbitrary_types_allowed = True


class ControlMessage(BaseModel):
    """Message schema for control commands."""

    message_id: str
    timestamp: datetime
    target_actuator: str
    command: str
    parameters: Optional[Dict[str, Any]] = None
    priority: str = "normal"
    sequence_number: int


class StatusMessage(BaseModel):
    """Message schema for system status updates."""

    message_id: str
    timestamp: datetime
    component_id: str
    status: str
    details: Optional[Dict[str, Any]] = None
    sequence_number: int
