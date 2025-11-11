# cyber_physical_system/physical_layer/actuators.py

class Actuator:
    """
    Simulates industrial actuators for control responses.
    """
    
    def __init__(self, actuator_id: str, actuator_type: str):
        """
        Initialize actuator.
        
        Args:
            actuator_id: Unique identifier
            actuator_type: Type of actuator (valve, motor, alarm, etc.)
        """
        self.actuator_id = actuator_id
        self.actuator_type = actuator_type
        self.is_active = False
        self.current_state = "idle"
        self.lock = Lock()
        logger.info(f"Actuator {actuator_id} ({actuator_type}) initialized")
    
    def start(self) -> None:
        """Start the actuator."""
        with self.lock:
            self.is_active = True
            logger.info(f"Actuator {self.actuator_id} started")
    
    def stop(self) -> None:
        """Stop the actuator."""
        with self.lock:
            self.is_active = False
            self.current_state = "idle"
            logger.info(f"Actuator {self.actuator_id} stopped")
    
    def execute_command(self, command: str, parameters: Optional[Dict] = None) -> Dict:
        """
        Execute actuator command.
        
        Args:
            command: Command to execute
            parameters: Optional command parameters
            
        Returns:
            Dictionary containing execution result
        """
        if not self.is_active:
            logger.warning(f"Actuator {self.actuator_id} is not active")
            return {"success": False, "message": "Actuator not active"}
        
        try:
            timestamp = datetime.now()
            
            # Simulate command execution
            if command == "activate":
                self.current_state = "active"
            elif command == "deactivate":
                self.current_state = "idle"
            elif command == "emergency_stop":
                self.current_state = "stopped"
            else:
                self.current_state = command
            
            logger.info(f"Actuator {self.actuator_id} executed command: {command}")
            
            return {
                "actuator_id": self.actuator_id,
                "command": command,
                "parameters": parameters,
                "new_state": self.current_state,
                "timestamp": timestamp,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error executing actuator command: {e}")
            return {"success": False, "message": str(e)}
    
    def get_status(self) -> Dict:
        """Get current actuator status."""
        return {
            "actuator_id": self.actuator_id,
            "actuator_type": self.actuator_type,
            "is_active": self.is_active,
            "current_state": self.current_state,
            "timestamp": datetime.now()
        }

