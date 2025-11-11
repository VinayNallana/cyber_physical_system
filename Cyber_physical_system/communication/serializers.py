# cyber_physical_system/communication/serializers.py

class MessageSerializer:
    """Handles message serialization and deserialization."""
    
    @staticmethod
    def serialize(message: Any) -> str:
        """
        Serialize message to JSON.
        
        Args:
            message: Message to serialize
            
        Returns:
            JSON string
        """
        try:
            if isinstance(message, BaseModel):
                return message.model_dump_json()
            else:
                # Handle datetime objects
                def default_converter(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                return json.dumps(message, default=default_converter)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return None
    
    @staticmethod
    def deserialize(json_str: str, model_class: Optional[type] = None) -> Any:
        """
        Deserialize JSON to message object.
        
        Args:
            json_str: JSON string
            model_class: Optional Pydantic model class
            
        Returns:
            Deserialized object
        """
        try:
            if model_class and issubclass(model_class, BaseModel):
                return model_class.model_validate_json(json_str)
            else:
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
