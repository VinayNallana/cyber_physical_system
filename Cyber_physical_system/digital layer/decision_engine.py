# cyber_physical_system/digital_layer/decision_engine.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Decision engine for processing sensor data, detection results,
    and generating control decisions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize decision engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.decision_history = deque(maxlen=1000)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.7)
        self.detection_confidence_threshold = self.config.get('detection_confidence', 0.5)
        self.is_active = False
        logger.info("Decision engine initialized")
    
    def start(self) -> None:
        """Start the decision engine."""
        self.is_active = True
        logger.info("Decision engine started")
    
    def stop(self) -> None:
        """Stop the decision engine."""
        self.is_active = False
        logger.info("Decision engine stopped")
    
    def process_detection_results(self, detection_data: Dict) -> Dict:
        """
        Process object detection results and generate decisions.
        
        Args:
            detection_data: Detection results from YOLOv9
            
        Returns:
            Decision dictionary
        """
        if not self.is_active:
            return {"error": "Decision engine not active"}
        
        try:
            detections = detection_data.get('detections', [])
            timestamp = detection_data.get('timestamp', datetime.now())
            
            # Analyze detections
            high_confidence_objects = [
                d for d in detections 
                if d.get('confidence', 0) >= self.detection_confidence_threshold
            ]
            
            # Generate decision
            decision = {
                "timestamp": timestamp,
                "decision_type": "object_detection",
                "objects_detected": len(high_confidence_objects),
                "total_detections": len(detections),
                "action": "none",
                "priority": "low",
                "details": {}
            }
            
            # Determine action based on detections
            if len(high_confidence_objects) > 5:
                decision["action"] = "alert_high_activity"
                decision["priority"] = "high"
                decision["details"] = {"reason": "High number of objects detected"}
            elif len(high_confidence_objects) > 0:
                decision["action"] = "monitor"
                decision["priority"] = "medium"
            
            self.decision_history.append(decision)
            logger.info(f"Decision made: {decision['action']}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error processing detection results: {e}")
            return {"error": str(e)}
    
    def process_anomaly_results(self, anomaly_data: Dict) -> Dict:
        """
        Process anomaly detection results and generate decisions.
        
        Args:
            anomaly_data: Anomaly detection results from PaDiM
            
        Returns:
            Decision dictionary
        """
        if not self.is_active:
            return {"error": "Decision engine not active"}
        
        try:
            anomaly_score = anomaly_data.get('anomaly_score', 0)
            is_anomalous = anomaly_data.get('is_anomalous', False)
            severity = anomaly_data.get('severity_level', 'low')
            timestamp = anomaly_data.get('timestamp', datetime.now())
            
            # Generate decision
            decision = {
                "timestamp": timestamp,
                "decision_type": "anomaly_detection",
                "anomaly_score": anomaly_score,
                "is_anomalous": is_anomalous,
                "severity": severity,
                "action": "none",
                "priority": "low",
                "details": {}
            }
            
            # Determine action based on anomaly
            if is_anomalous:
                if severity == "critical":
                    decision["action"] = "emergency_stop"
                    decision["priority"] = "critical"
                    decision["details"] = {
                        "reason": "Critical anomaly detected",
                        "recommended_actions": ["Stop system", "Alert operator", "Inspect equipment"]
                    }
                elif severity == "high":
                    decision["action"] = "alert_and_reduce_speed"
                    decision["priority"] = "high"
                    decision["details"] = {
                        "reason": "High severity anomaly detected",
                        "recommended_actions": ["Reduce operation speed", "Alert operator"]
                    }
                elif severity == "medium":
                    decision["action"] = "monitor_closely"
                    decision["priority"] = "medium"
                    decision["details"] = {
                        "reason": "Medium severity anomaly detected",
                        "recommended_actions": ["Monitor system", "Log event"]
                    }
                else:
                    decision["action"] = "log_anomaly"
                    decision["priority"] = "low"
            
            self.decision_history.append(decision)
            logger.info(f"Decision made: {decision['action']} (severity: {severity})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error processing anomaly results: {e}")
            return {"error": str(e)}
    
    def process_sensor_data(self, sensor_data: Dict) -> Dict:
        """
        Process environmental sensor data and generate decisions.
        
        Args:
            sensor_data: Sensor reading data
            
        Returns:
            Decision dictionary
        """
        if not self.is_active:
            return {"error": "Decision engine not active"}
        
        try:
            sensor_type = sensor_data.get('sensor_type', 'unknown')
            value = sensor_data.get('value', 0)
            is_normal = sensor_data.get('is_normal', True)
            timestamp = sensor_data.get('timestamp', datetime.now())
            
            decision = {
                "timestamp": timestamp,
                "decision_type": "sensor_monitoring",
                "sensor_type": sensor_type,
                "value": value,
                "is_normal": is_normal,
                "action": "none",
                "priority": "low",
                "details": {}
            }
            
            if not is_normal:
                decision["action"] = "alert_out_of_range"
                decision["priority"] = "high"
                decision["details"] = {
                    "reason": f"{sensor_type} out of normal range",
                    "value": value
                }
            
            self.decision_history.append(decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            return {"error": str(e)}
    
    def get_decision_history(self, limit: int = 100) -> List[Dict]:
        """Get decision history."""
        return list(self.decision_history)[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get decision engine statistics."""
        decisions = list(self.decision_history)
        
        if not decisions:
            return {
                "total_decisions": 0,
                "by_type": {},
                "by_priority": {},
                "by_action": {}
            }
        
        return {
            "total_decisions": len(decisions),
            "by_type": self._count_by_key(decisions, 'decision_type'),
            "by_priority": self._count_by_key(decisions, 'priority'),
            "by_action": self._count_by_key(decisions, 'action')
        }
    
    @staticmethod
    def _count_by_key(decisions: List[Dict], key: str) -> Dict:
        """Count decisions by a specific key."""
        counts = {}
        for decision in decisions:
            value = decision.get(key, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts
