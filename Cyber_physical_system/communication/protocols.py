# cyber_physical_system/communication/protocols.py

import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CommunicationProtocol:
    """Base class for communication protocols."""

    def __init__(self, protocol_name: str):
        self.protocol_name = protocol_name
        self.latency_ms: float = 0.0
        logger.info(f"Protocol '{protocol_name}' initialized")

    def send(self, data: Any) -> bool:
        """Send data using the protocol."""
        raise NotImplementedError

    def receive(self) -> Any:
        """Receive data using the protocol."""
        raise NotImplementedError

    def get_latency(self) -> float:
        """Get current latency in milliseconds."""
        return self.latency_ms


class RESTProtocol(CommunicationProtocol):
    """REST API protocol simulation."""

    def __init__(self):
        super().__init__("REST")

    def send(self, data: Any) -> bool:
        """Simulate REST API request."""
        start_time = time.time()
        # Simulate network latency
        time.sleep(0.001)
        self.latency_ms = (time.time() - start_time) * 1000
        return True

    def receive(self) -> Any:
        """Simulate REST API response."""
        time.sleep(0.001)
        return {"status": "success"}


class MQTTProtocol(CommunicationProtocol):
    """MQTT protocol simulation."""

    def __init__(self):
        super().__init__("MQTT")

    def send(self, data: Any) -> bool:
        """Simulate MQTT publish."""
        start_time = time.time()
        time.sleep(0.0005)  # Faster than REST
        self.latency_ms = (time.time() - start_time) * 1000
        return True

    def receive(self) -> Any:
        """Simulate MQTT subscribe."""
        time.sleep(0.0005)
        return {"status": "received"}
