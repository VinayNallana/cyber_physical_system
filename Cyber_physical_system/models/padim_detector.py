# cyber_physical_system/models/padim_detector.py

import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.covariance import LedoitWolf
import logging
from datetime import datetime
import time
import pickle

logger = logging.getLogger(__name__)


class PaDiMDetector:
    """
    PaDiM (Patch Distribution Modeling) anomaly detection model.
    Uses pretrained ResNet features for anomaly detection.
    """

    def __init__(self, backbone: str = "resnet18", device: str = "auto"):
        """
        Initialize PaDiM detector.

        Args:
            backbone: Backbone network ('resnet18' or 'resnet50')
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.backbone_name = backbone
        self.device = self._setup_device(device)
        self.backbone = None
        self.feature_extractor = None
        self.is_trained = False

        # Training statistics
        self.mean_embeddings = None
        self.cov_embeddings = None
        self.train_feature_bank: List[torch.Tensor] = []

        # Metrics
        self.inference_times = []
        self.anomaly_scores = []

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self._initialize_backbone()
        logger.info(
            f"PaDiM detector initialized with {backbone} on {self.device}"
        )

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _initialize_backbone(self) -> None:
        """Initialize feature extraction backbone."""
        try:
            if self.backbone_name == "resnet18":
                self.backbone = models.resnet18(pretrained=True)
            elif self.backbone_name == "resnet50":
                self.backbone = models.resnet50(pretrained=True)
            else:
                raise ValueError(f"Unsupported backbone: {self.backbone_name}")

            # Remove final classification layers
            self.backbone = torch.nn.Sequential(
                *list(self.backbone.children())[:-2]
            )
            self.backbone.to(self.device)
            self.backbone.eval()

            # Freeze parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

            logger.info(f"Backbone {self.backbone_name} initialized")

        except Exception as e:
            logger.error(f"Error initializing backbone: {e}")
            raise

    def extract_features(
        self, image: Union[str, np.ndarray]
    ) -> Optional[torch.Tensor]:
        """
        Extract features from an image using the backbone.

        Args:
            image: Image path or numpy array

        Returns:
            Feature tensor
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image, np.ndarray) and len(image.shape) == 3:
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preprocess
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.backbone(image_tensor)

            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def train(
        self,
        train_image_paths: List[str],
        resize_shape: Tuple[int, int] = (256, 256),
    ) -> Dict:
        """
        Train PaDiM on normal images.

        Args:
            train_image_paths: List of paths to normal training images
            resize_shape: Target image size

        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Training PaDiM on {len(train_image_paths)} images")
            start_time = time.time()

            self.train_feature_bank = []

            # Extract features from all training images
            for i, image_path in enumerate(train_image_paths):
                if i % 10 == 0:
                    logger.info(
                        f"Processing image {i+1}/{len(train_image_paths)}"
                    )

                # Load and resize image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Failed to load image: {image_path}")
                    continue

                image = cv2.resize(image, resize_shape)

                # Extract features
                features = self.extract_features(image)
                if features is not None:
                    # Flatten spatial dimensions
                    features = features.squeeze(0)  # Remove batch dimension
                    features = features.reshape(
                        features.shape[0], -1
                    )  # [C, H*W]
                    self.train_feature_bank.append(features.cpu())

            if not self.train_feature_bank:
                return {
                    "success": False,
                    "error": "No valid training features extracted",
                }

            # Stack all features
            all_features = torch.stack(
                self.train_feature_bank, dim=0
            )  # [N, C, H*W]

            # Compute mean and covariance for each spatial location
            logger.info("Computing statistical parameters...")
            all_features = all_features.permute(2, 0, 1)  # [H*W, N, C]

            self.mean_embeddings = torch.mean(all_features, dim=1)  # [H*W, C]

            # Compute covariance matrices
            self.cov_embeddings = []
            for i in range(all_features.shape[0]):
                features_at_location = all_features[i].numpy()  # [N, C]
                # Use Ledoit-Wolf covariance estimation
                lw = LedoitWolf()
                cov = lw.fit(features_at_location).covariance_
                self.cov_embeddings.append(cov)

            self.cov_embeddings = np.array(self.cov_embeddings)  # [H*W, C, C]

            self.is_trained = True
            training_time = time.time() - start_time

            result = {
                "success": True,
                "training_images": len(train_image_paths),
                "training_time_seconds": training_time,
                "feature_dimensions": all_features.shape[-1],
                "spatial_locations": all_features.shape[0],
            }

            logger.info("Training completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {"success": False, "error": str(e)}

    def predict(
        self, image: Union[str, np.ndarray], threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect anomalies in an image.

        Args:
            image: Image path or numpy array
            threshold: Anomaly score threshold (auto-calculated if None)

        Returns:
            Anomaly detection results
        """
        if not self.is_trained:
            return {"success": False, "error": "Model not trained"}

        try:
            start_time = time.time()

            # Load image
            if isinstance(image, str):
                img = cv2.imread(image)
            else:
                img = image.copy()

            if img is None:
                return {"success": False, "error": "Failed to load image"}

            # Extract features
            features = self.extract_features(img)
            if features is None:
                return {
                    "success": False,
                    "error": "Failed to extract features",
                }

            # Compute anomaly scores
            features = features.squeeze(0)  # Remove batch dimension
            h, w = features.shape[1], features.shape[2]
            features = features.reshape(features.shape[0], -1).T  # [H*W, C]

            # Compute Mahalanobis distance for each location
            anomaly_map = np.zeros((h * w,))

            for i in range(len(features)):
                mean = self.mean_embeddings[i].numpy()
                cov = self.cov_embeddings[i]
                feature = features[i].cpu().numpy()
                try:
                    # Add small regularization to avoid singular matrix
                    cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
                    dist = mahalanobis(feature, mean, cov_inv)
                    anomaly_map[i] = dist
                except Exception as e:
                    logger.warning(
                        f"Mahalanobis computation failed at index {i}: {e}"
                    )
                    anomaly_map[i] = 0

            # Reshape to spatial dimensions
            anomaly_map = anomaly_map.reshape(h, w)

            # Apply Gaussian smoothing
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # Normalize to [0, 1]
            anomaly_map = (anomaly_map - anomaly_map.min()) / (
                anomaly_map.max() - anomaly_map.min() + 1e-8
            )

            # Compute overall anomaly score
            anomaly_score = float(np.max(anomaly_map))

            # Determine if anomalous
            if threshold is None:
                # Auto threshold at 75th percentile of historical scores
                if len(self.anomaly_scores) > 10:
                    threshold = np.percentile(self.anomaly_scores, 75)
                else:
                    threshold = 0.5

            is_anomalous = anomaly_score > threshold

            # Determine severity
            if anomaly_score > 0.9:
                severity = "critical"
            elif anomaly_score > 0.75:
                severity = "high"
            elif anomaly_score > 0.5:
                severity = "medium"
            else:
                severity = "low"

            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            self.anomaly_scores.append(anomaly_score)

            # Resize anomaly map to original image size
            anomaly_map_resized = cv2.resize(
                anomaly_map, (img.shape[1], img.shape[0])
            )

            return {
                "success": True,
                "timestamp": datetime.now(),
                "anomaly_score": anomaly_score,
                "is_anomalous": is_anomalous,
                "severity_level": severity,
                "threshold": threshold,
                "anomaly_map": anomaly_map_resized,
                "inference_time_ms": inference_time,
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"success": False, "error": str(e)}

    def predict_batch(
        self, images: List[Union[str, np.ndarray]]
    ) -> List[Dict]:
        """Batch anomaly detection."""
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results

    def save_model(self, save_path: str) -> bool:
        """
        Save trained model parameters.

        Args:
            save_path: Path to save model

        Returns:
            Success status
        """
        if not self.is_trained:
            logger.error("Model not trained")
            return False
        try:
            model_data = {
                "mean_embeddings": self.mean_embeddings,
                "cov_embeddings": self.cov_embeddings,
                "backbone_name": self.backbone_name,
            }

            with open(save_path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, load_path: str) -> bool:
        """
        Load trained model parameters.

        Args:
            load_path: Path to saved model

        Returns:
            Success status
        """
        try:
            with open(load_path, "rb") as f:
                model_data = pickle.load(f)

            self.mean_embeddings = model_data["mean_embeddings"]
            self.cov_embeddings = model_data["cov_embeddings"]
            self.backbone_name = model_data["backbone_name"]
            self.is_trained = True

            logger.info(f"Model loaded from {load_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def get_metrics(self) -> Dict:
        """Get model performance metrics."""
        if not self.inference_times:
            return {
                "avg_inference_time_ms": 0,
                "min_inference_time_ms": 0,
                "max_inference_time_ms": 0,
                "total_inferences": 0,
                "avg_anomaly_score": 0,
            }

        return {
            "avg_inference_time_ms": np.mean(self.inference_times),
            "min_inference_time_ms": np.min(self.inference_times),
            "max_inference_time_ms": np.max(self.inference_times),
            "total_inferences": len(self.inference_times),
            "avg_anomaly_score": (
                np.mean(self.anomaly_scores) if self.anomaly_scores else 0
            ),
        }

    def visualize_anomaly_map(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize anomaly map overlaid on image.

        Args:
            image: Original image
            anomaly_map: Anomaly map
            save_path: Optional path to save visualization
        Returns:
            Visualization image
        """
        # Convert anomaly map to heatmap
        heatmap = cv2.applyColorMap(
            (anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Overlay on original image
        visualization = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

        if save_path:
            cv2.imwrite(save_path, visualization)

        return visualization


# End of padim_detector module
