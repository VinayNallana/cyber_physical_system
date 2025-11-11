# cyber_physical_system/communication/serializers.py
import json
from datetime import datetime
import numpy as np
import logging
from typing import Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MessageSerializer:
    """Handles message serialization and deserialization."""

    @staticmethod
    def serialize(message: Any) -> Optional[str]:
        """
        Serialize message to JSON.

        Args:
            message: Message to serialize

        Returns:
            JSON string
        """
        try:
            if isinstance(message, BaseModel):
                # Support both pydantic v2 (model_dump_json) and v1 (json)
                if hasattr(message, "model_dump_json"):
                    return message.model_dump_json()
                if hasattr(message, "json"):
                    return message.json()
            else:
                # Handle datetime objects
                def default_converter(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError(
                        f"Object of type {type(obj)} is not JSON serializable"
                    )

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
                # Support pydantic v2 and v1 parsing APIs
                if hasattr(model_class, "model_validate_json"):
                    return model_class.model_validate_json(json_str)
                if hasattr(model_class, "parse_raw"):
                    return model_class.parse_raw(json_str)
                # Last resort: parse to dict then construct
                data = json.loads(json_str)
                return model_class(**data)
            else:
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
