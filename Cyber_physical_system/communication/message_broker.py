# cyber_physical_system/communication/message_broker.py

import queue
import threading
import json
import uuid
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


class MessageBroker:
    """
    Message broker for handling communication between layers.
    Supports publish-subscribe and request-response patterns.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize message broker.
        
        Args:
            max_queue_size: Maximum size of message queues
        """
        self.topics: Dict[str, List[queue.Queue]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Dict] = []
        self.sequence_counter = 0
        self.lock = threading.Lock()
        self.max_queue_size = max_queue_size
        self.is_running = False
        self.worker_thread = None
        logger.info("Message broker initialized")
    
    def start(self) -> None:
        """Start the message broker."""
        self.is_running = True
        logger.info("Message broker started")
    
    def stop(self) -> None:
        """Stop the message broker."""
        self.is_running = False
        logger.info("Message broker stopped")
    
    def create_topic(self, topic_name: str) -> None:
        """
        Create a new topic for pub-sub messaging.
        
        Args:
            topic_name: Name of the topic
        """
        with self.lock:
            if topic_name not in self.topics:
                self.topics[topic_name] = []
                self.subscribers[topic_name] = []
                logger.info(f"Topic '{topic_name}' created")
    
    def subscribe(self, topic_name: str, callback: Callable) -> None:
        """
        Subscribe to a topic with a callback function.
        
        Args:
            topic_name: Name of the topic to subscribe to
            callback: Function to call when message is received
        """
        with self.lock:
            if topic_name not in self.subscribers:
                self.create_topic(topic_name)
            
            self.subscribers[topic_name].append(callback)
            logger.info(f"New subscriber added to topic '{topic_name}'")
    
    def publish(self, topic_name: str, message: Any, retry_count: int = 3) -> bool:
        """
        Publish a message to a topic.
        
        Args:
            topic_name: Name of the topic
            message: Message to publish
            retry_count: Number of retry attempts
            
        Returns:
            Success status
        """
        if not self.is_running:
            logger.warning("Message broker is not running")
            return False
        
        with self.lock:
            if topic_name not in self.topics:
                self.create_topic(topic_name)
            
            # Add sequence number and message ID
            message_envelope = {
                "message_id": str(uuid.uuid4()),
                "topic": topic_name,
                "timestamp": datetime.now(),
                "sequence_number": self.sequence_counter,
                "payload": message,
                "retry_count": retry_count
            }
            self.sequence_counter += 1
            
            # Store in history
            self.message_history.append(message_envelope)
            
            # Notify all subscribers
            for callback in self.subscribers.get(topic_name, []):
                try:
                    callback(message_envelope)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
                    if retry_count > 0:
                        # Retry with exponential backoff
                        time.sleep(0.1 * (4 - retry_count))
                        return self.publish(topic_name, message, retry_count - 1)
            
            logger.debug(f"Message published to topic '{topic_name}'")
            return True
    
    def get_message_history(self, topic_name: Optional[str] = None, 
                           limit: int = 100) -> List[Dict]:
        """
        Get message history.
        
        Args:
            topic_name: Optional topic filter
            limit: Maximum number of messages to return
            
        Returns:
            List of message envelopes
        """
        with self.lock:
            if topic_name:
                filtered = [msg for msg in self.message_history 
                           if msg["topic"] == topic_name]
            else:
                filtered = self.message_history
            
            return filtered[-limit:]
    
    def get_communication_metrics(self) -> Dict:
        """
        Get communication metrics.
        
        Returns:
            Dictionary containing metrics
        """
        with self.lock:
            return {
                "total_messages": len(self.message_history),
                "active_topics": len(self.topics),
                "total_subscribers": sum(len(subs) for subs in self.subscribers.values()),
                "sequence_counter": self.sequence_counter
            }
