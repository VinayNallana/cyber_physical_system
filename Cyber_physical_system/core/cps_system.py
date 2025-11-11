# cyber_physical_system/core/cps_system.py

import logging
from typing import Dict, Optional, List
from datetime import datetime

# Import all components
from ..physical_layer.sensors import CameraSensor, EnvironmentalSensor
from ..physical_layer.actuators import Actuator
from ..communication.message_broker import MessageBroker
from ..communication.protocols import RESTProtocol, MQTTProtocol
from ..communication.serializers import MessageSerializer
from ..digital_layer.decision_engine import DecisionEngine
from ..digital_layer.analytics import AnalyticsEngine
from ..models.yolo_detector import YOLOv9Detector
from ..models.padim_detector import PaDiMDetector
from ..dataset.dataset_manager import DatasetManager
from ..metrics.metrics_tracker import MetricsTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CyberPhysicalSystem:
    """
    Main Cyber-Physical System integrating all layers and components.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the CPS system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_running = False

        logger.info("Initializing Cyber-Physical System...")

        # Initialize Physical Layer
        self._initialize_physical_layer()

        # Initialize Communication Layer
        self._initialize_communication_layer()

        # Initialize Digital Layer
        self._initialize_digital_layer()

        # Initialize Models
        self._initialize_models()

        # Initialize Dataset Manager
        self.dataset_manager = DatasetManager(
            base_dir=self.config.get("dataset_dir", "./datasets")
        )

        # Initialize Metrics Tracker
        self.metrics_tracker = MetricsTracker(
            window_size=self.config.get("metrics_window", 1000)
        )

        # Setup message subscriptions
        self._setup_message_subscriptions()

        logger.info("Cyber-Physical System initialized successfully")

    def _initialize_physical_layer(self) -> None:
        """Initialize physical layer components."""
        logger.info("Initializing physical layer...")

        # Camera sensors
        self.camera_sensors = {
            "cam_1": CameraSensor("cam_1", fps=30, resolution=(640, 480)),
            "cam_2": CameraSensor("cam_2", fps=30, resolution=(640, 480)),
        }

        # Environmental sensors
        self.environmental_sensors = {
            "temp_1": EnvironmentalSensor(
                "temp_1", "temperature", "°C", (20.0, 80.0)
            ),
            "pressure_1": EnvironmentalSensor(
                "pressure_1", "pressure", "PSI", (10.0, 100.0)
            ),
            "vibration_1": EnvironmentalSensor(
                "vibration_1", "vibration", "mm/s", (0.0, 50.0)
            ),
        }

        # Actuators
        self.actuators = {
            "valve_1": Actuator("valve_1", "valve"),
            "motor_1": Actuator("motor_1", "motor"),
            "alarm_1": Actuator("alarm_1", "alarm"),
        }

        num_cameras = len(self.camera_sensors)
        num_sensors = len(self.environmental_sensors)
        num_actuators = len(self.actuators)

        logger.info(
            f"Physical layer initialized: {num_cameras} cameras, "
            f"{num_sensors} sensors, {num_actuators} actuators"
        )

    def _initialize_communication_layer(self) -> None:
        """Initialize communication layer components."""
        logger.info("Initializing communication layer...")

        # Message broker
        self.message_broker = MessageBroker(max_queue_size=1000)

        # Communication protocols
        self.protocols = {"rest": RESTProtocol(), "mqtt": MQTTProtocol()}

        # Message serializer
        self.serializer = MessageSerializer()

        # Create topics
        topics = [
            "sensor/camera",
            "sensor/environmental",
            "detection/yolo",
            "detection/padim",
            "control/actuators",
            "analytics/metrics",
            "system/status",
        ]

        for topic in topics:
            self.message_broker.create_topic(topic)

        logger.info("Communication layer initialized")

    def _initialize_digital_layer(self) -> None:
        """Initialize digital layer components."""
        logger.info("Initializing digital layer...")

        # Decision engine
        self.decision_engine = DecisionEngine(
            config=self.config.get("decision_engine", {})
        )

        # Analytics engine
        self.analytics_engine = AnalyticsEngine(
            window_size=self.config.get("analytics_window", 100)
        )

        logger.info("Digital layer initialized")

    def _initialize_models(self) -> None:
        """Initialize AI models."""
        logger.info("Initializing AI models...")

        # YOLOv9 detector
        device = self.config.get("device", "auto")
        self.yolo_detector = YOLOv9Detector(device=device)

        # PaDiM detector
        backbone = self.config.get("padim_backbone", "resnet18")
        self.padim_detector = PaDiMDetector(backbone=backbone, device=device)

        logger.info("AI models initialized")

    def _setup_message_subscriptions(self) -> None:
        """Setup message subscriptions for inter-layer communication."""
        # Subscribe decision engine to detection results
        self.message_broker.subscribe(
            "detection/yolo", self._handle_detection_message
        )
        self.message_broker.subscribe(
            "detection/padim", self._handle_anomaly_message
        )

        # Subscribe analytics to all events
        self.message_broker.subscribe(
            "detection/yolo", self._handle_analytics_update
        )
        self.message_broker.subscribe(
            "detection/padim", self._handle_analytics_update
        )

        logger.info("Message subscriptions configured")

    def _handle_detection_message(self, message: Dict) -> None:
        """Handle detection messages."""
        try:
            payload = message.get("payload", {})
            if payload.get("success"):
                # Process in decision engine
                decision = self.decision_engine.process_detection_results(
                    payload
                )

                # Update metrics
                self.metrics_tracker.update_detection_metrics(payload)

                # Execute actions based on decision
                if decision.get("action") != "none":
                    self._execute_control_action(decision)

        except Exception as e:
            logger.error(f"Error handling detection message: {e}")

    def _handle_anomaly_message(self, message: Dict) -> None:
        """Handle anomaly detection messages."""
        try:
            payload = message.get("payload", {})
            if payload.get("success"):
                # Process in decision engine
                decision = self.decision_engine.process_anomaly_results(
                    payload
                )

                # Update metrics
                self.metrics_tracker.update_anomaly_metrics(payload)

                # Execute actions based on decision
                if decision.get("action") != "none":
                    self._execute_control_action(decision)

        except Exception as e:
            logger.error(f"Error handling anomaly message: {e}")

    def _handle_analytics_update(self, message: Dict) -> None:
        """Handle analytics updates."""
        try:
            payload = message.get("payload", {})
            topic = message.get("topic", "")
            if "yolo" in topic:
                self.analytics_engine.update_detection_metrics(payload)
            elif "padim" in topic:
                self.analytics_engine.update_anomaly_metrics(payload)

        except Exception as e:
            logger.error(f"Error updating analytics: {e}")

    def _execute_control_action(self, decision: Dict) -> None:
        """Execute control actions based on decisions."""
        try:
            action = decision.get("action", "none")
            priority = decision.get("priority", "low")

            logger.info(f"Executing action: {action} (priority: {priority})")

            # Map actions to actuator commands
            if action == "emergency_stop":
                for actuator in self.actuators.values():
                    actuator.execute_command("emergency_stop")
            elif action == "alert_high_activity":
                self.actuators["alarm_1"].execute_command("activate")
            elif action == "alert_and_reduce_speed":
                self.actuators["motor_1"].execute_command(
                    "reduce_speed", {"speed": 50}
                )
                self.actuators["alarm_1"].execute_command("activate")

            # Publish control message
            self.message_broker.publish(
                "control/actuators",
                {
                    "action": action,
                    "priority": priority,
                    "timestamp": datetime.now(),
                },
            )

        except Exception as e:
            logger.error(f"Error executing control action: {e}")

    def start(self) -> None:
        """Start the CPS system."""
        if self.is_running:
            logger.warning("System is already running")
            return

        logger.info("Starting Cyber-Physical System...")

        # Start message broker
        self.message_broker.start()

        # Start physical layer components
        for sensor in self.camera_sensors.values():
            sensor.start()

        for sensor in self.environmental_sensors.values():
            sensor.start()

        for actuator in self.actuators.values():
            actuator.start()

        # Start digital layer components
        self.decision_engine.start()

        # Start metrics tracking
        self.metrics_tracker.start_tracking()

        self.is_running = True
        logger.info("Cyber-Physical System started successfully")

    def stop(self) -> None:
        """Stop the CPS system."""
        if not self.is_running:
            logger.warning("System is not running")
            return

        logger.info("Stopping Cyber-Physical System...")

        # Stop physical layer components
        for sensor in self.camera_sensors.values():
            sensor.stop()

        for sensor in self.environmental_sensors.values():
            sensor.stop()

        for actuator in self.actuators.values():
            actuator.stop()

        # Stop digital layer components
        self.decision_engine.stop()

        # Stop message broker
        self.message_broker.stop()

        self.is_running = False
        logger.info("Cyber-Physical System stopped")

    def process_image_yolo(
        self, image_path: str, conf_threshold: float = 0.5
    ) -> Dict:
        """
        Process image through YOLOv9 detection pipeline.

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold

        Returns:
            Detection results
        """
        try:
            # Run detection
            result = self.yolo_detector.predict(
                image_path, conf_threshold=conf_threshold
            )

            if result.get("success"):
                # Publish to message broker
                self.message_broker.publish("detection/yolo", result)

            return result

        except Exception as e:
            logger.error(f"Error processing image with YOLO: {e}")
            return {"success": False, "error": str(e)}

    def process_image_padim(self, image_path: str) -> Dict:
        """
        Process image through PaDiM anomaly detection pipeline.

        Args:
            image_path: Path to image

        Returns:
            Anomaly detection results
        """
        try:
            # Run anomaly detection
            result = self.padim_detector.predict(image_path)

            if result.get("success"):
                # Publish to message broker
                self.message_broker.publish("detection/padim", result)

            return result

        except Exception as e:
            logger.error(f"Error processing image with PaDiM: {e}")
            return {"success": False, "error": str(e)}

    def train_yolo(
        self, data_yaml: str, epochs: int = 100, batch_size: int = 16, **kwargs
    ) -> Dict:
        """
        Train YOLOv9 model.

        Args:
            data_yaml: Path to data.yaml
            epochs: Number of epochs
            batch_size: Batch size
            **kwargs: Additional training arguments

        Returns:
            Training results
        """
        return self.yolo_detector.train(
            data_yaml, epochs, batch_size=batch_size, **kwargs
        )

    def train_padim(self, train_image_paths: List[str]) -> Dict:
        """
        Train PaDiM model.

        Args:
            train_image_paths: List of training image paths

        Returns:
            Training results
        """
        return self.padim_detector.train(train_image_paths)

    def load_yolo_model(
        self, model_path: str, class_names: Optional[List[str]] = None
    ) -> bool:
        """Load pretrained YOLOv9 model."""
        return self.yolo_detector.load_model(model_path, class_names)

    def load_padim_model(self, model_path: str) -> bool:
        """Load pretrained PaDiM model."""
        return self.padim_detector.load_model(model_path)

    def save_yolo_model(self, save_path: str) -> bool:
        """Save YOLOv9 model."""
        return self.yolo_detector.export_model(save_path)

    def save_padim_model(self, save_path: str) -> bool:
        """Save PaDiM model."""
        return self.padim_detector.save_model(save_path)

    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            "is_running": self.is_running,
            "timestamp": datetime.now(),
            "physical_layer": {
                "cameras": {
                    k: v.is_active for k, v in self.camera_sensors.items()
                },
                "sensors": {
                    k: v.is_active
                    for k, v in self.environmental_sensors.items()
                },
                "actuators": {
                    k: v.is_active for k, v in self.actuators.items()
                },
            },
            "models": {
                "yolo_trained": self.yolo_detector.is_trained,
                "padim_trained": self.padim_detector.is_trained,
            },
            "metrics": self.metrics_tracker.get_comprehensive_report(),
            "analytics": self.analytics_engine.get_system_health(),
            "communication": self.message_broker.get_communication_metrics(),
        }

    def get_metrics_report(self) -> Dict:
        """Get comprehensive metrics report."""
        return self.metrics_tracker.get_comprehensive_report()
