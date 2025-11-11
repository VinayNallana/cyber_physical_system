# cyber_physical_system/models/yolo_detector.py

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class YOLOv9Detector:
    """
    YOLOv9 object detection model wrapper with training and inference capabilities.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize YOLOv9 detector.
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.is_trained = False
        self.class_names = []
        self.num_classes = 0
        self.training_history = []
        self.inference_times = []
        
        logger.info(f"YOLOv9 detector initialized on device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model(self, model_path: str, class_names: Optional[List[str]] = None) -> bool:
        """
        Load pretrained YOLOv9 model.
        
        Args:
            model_path: Path to model weights
            class_names: Optional list of class names
            
        Returns:
            Success status
        """
        try:
            from ultralytics import YOLO
            
            self.model_path = model_path
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            if class_names:
                self.class_names = class_names
                self.num_classes = len(class_names)
            else:
                # Try to get from model
                if hasattr(self.model, 'names'):
                    self.class_names = list(self.model.names.values())
                    self.num_classes = len(self.class_names)
            
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Classes: {self.class_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640,
              batch_size: int = 16, **kwargs) -> Dict:
        """
        Train YOLOv9 model.
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            imgsz: Image size
            batch_size: Batch size
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        try:
            from ultralytics import YOLO
            
            # Initialize model for training
            if self.model is None:
                self.model = YOLO('yolov9c.pt')  # Start with pretrained weights
            
            logger.info(f"Starting training for {epochs} epochs")
            start_time = time.time()
            
            # Train the model
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=self.device,
                **kwargs
            )
            
            training_time = time.time() - start_time
            
            self.is_trained = True
            
            training_result = {
                "success": True,
                "epochs": epochs,
                "training_time_seconds": training_time,
                "final_metrics": {
                    "mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    "mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    "precision": float(results.results_dict.get('metrics/precision(B)', 0)),
                    "recall": float(results.results_dict.get('metrics/recall(B)', 0))
                }
            }
            
            self.training_history.append({
                "timestamp": datetime.now(),
                "result": training_result
            })
            
            logger.info("Training completed successfully")
            logger.info(f"Final mAP50: {training_result['final_metrics']['mAP50']:.4f}")
            
            return training_result
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, image: Union[str, np.ndarray], conf_threshold: float = 0.25,
                iou_threshold: float = 0.45) -> Dict:
        """
        Perform object detection on an image.
        
        Args:
            image: Image path or numpy array
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Detection results dictionary
        """
        if not self.is_trained or self.model is None:
            logger.error("Model not loaded or trained")
            return {
                "success": False,
                "error": "Model not ready"
            }
        
        try:
            start_time = time.time()
            
            # Run inference
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False
            )
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self.inference_times.append(inference_time)
            
            # Parse results
            detections = []
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    detection = {
                        "bbox": boxes[i].tolist(),
                        "confidence": float(confidences[i]),
                        "class_id": int(class_ids[i]),
                        "class_name": self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else f"class_{class_ids[i]}"
                    }
                    detections.append(detection)
            
            return {
                "success": True,
                "timestamp": datetime.now(),
                "detections": detections,
                "num_detections": len(detections),
                "inference_time_ms": inference_time,
                "image_shape": result.orig_shape
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict_batch(self, images: List[Union[str, np.ndarray]],
                     conf_threshold: float = 0.25) -> List[Dict]:
        """
        Perform batch prediction on multiple images.
        
        Args:
            images: List of image paths or numpy arrays
            conf_threshold: Confidence threshold
            
        Returns:
            List of detection results
        """
        results = []
        for image in images:
            result = self.predict(image, conf_threshold=conf_threshold)
            results.append(result)
        return results
    
    def validate(self, data_yaml: str) -> Dict:
        """
        Validate model on validation dataset.
        
        Args:
            data_yaml: Path to data.yaml file
            
        Returns:
            Validation metrics
        """
        if not self.is_trained or self.model is None:
            return {"success": False, "error": "Model not ready"}
        
        try:
            logger.info("Running validation...")
            results = self.model.val(data=data_yaml)
            
            return {
                "success": True,
                "metrics": {
                    "mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    "mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    "precision": float(results.results_dict.get('metrics/precision(B)', 0)),
                    "recall": float(results.results_dict.get('metrics/recall(B)', 0))
                }
            }
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return {"success": False, "error": str(e)}
    
    def export_model(self, export_path: str, format: str = 'onnx') -> bool:
        """
        Export model to different format.
        
        Args:
            export_path: Path to save exported model
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            
        Returns:
            Success status
        """
        if not self.is_trained or self.model is None:
            logger.error("Model not ready for export")
            return False
        
        try:
            self.model.export(format=format)
            logger.info(f"Model exported to {format} format")
            return True
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return False
    
    def get_metrics(self) -> Dict:
        """Get model performance metrics."""
        if not self.inference_times:
            return {
                "avg_inference_time_ms": 0,
                "min_inference_time_ms": 0,
                "max_inference_time_ms": 0,
                "total_inferences": 0
            }
        
        return {
            "avg_inference_time_ms": np.mean(self.inference_times),
            "min_inference_time_ms": np.min(self.inference_times),
            "max_inference_time_ms": np.max(self.inference_times),
            "total_inferences": len(self.inference_times)
        }
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict],
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, annotated)
        
        return annotated


