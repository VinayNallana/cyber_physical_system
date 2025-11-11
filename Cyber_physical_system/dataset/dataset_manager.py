# cyber_physical_system/dataset/dataset_manager.py

import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages datasets for training and testing YOLOv9 and PaDiM models.
    Supports ZIP file extraction and dataset organization.
    """

    def __init__(self, base_dir: str = "./datasets"):
        """
        Initialize dataset manager.

        Args:
            base_dir: Base directory for datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.yolo_dir = self.base_dir / "yolo"
        self.padim_dir = self.base_dir / "padim"

        self.yolo_dir.mkdir(exist_ok=True)
        self.padim_dir.mkdir(exist_ok=True)

        self.datasets = {}
        logger.info(f"Dataset manager initialized at {self.base_dir}")

    def extract_zip(
        self,
        zip_path: str,
        extract_to: Optional[str] = None,
        dataset_type: str = "yolo",
    ) -> Dict:
        """
        Extract ZIP file containing dataset.

        Args:
            zip_path: Path to ZIP file
            extract_to: Optional extraction directory
            dataset_type: Type of dataset ('yolo' or 'padim')
        Returns:
            Dictionary with extraction results
        """
        try:
            zip_path = Path(zip_path)

            if not zip_path.exists():
                logger.error(f"ZIP file not found: {zip_path}")
                return {"success": False, "error": "ZIP file not found"}

            # Determine extraction directory
            if extract_to is None:
                if dataset_type == "yolo":
                    extract_to = self.yolo_dir / zip_path.stem
                else:
                    extract_to = self.padim_dir / zip_path.stem
            else:
                extract_to = Path(extract_to)

            extract_to.mkdir(parents=True, exist_ok=True)

            # Extract ZIP
            logger.info(f"Extracting {zip_path} to {extract_to}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

            # Count files
            total_files = sum(1 for _ in extract_to.rglob("*") if _.is_file())

            # Register dataset
            dataset_info = {
                "name": zip_path.stem,
                "type": dataset_type,
                "path": str(extract_to),
                "extracted_at": datetime.now().isoformat(),
                "total_files": total_files,
            }

            self.datasets[zip_path.stem] = dataset_info

            logger.info(f"Successfully extracted {total_files} files")

            return {"success": True, "dataset_info": dataset_info}

        except Exception as e:
            logger.error(f"Error extracting ZIP: {e}")
            return {"success": False, "error": str(e)}

    def organize_yolo_dataset(self, dataset_path: str) -> Dict:
        """
        Organize YOLO dataset into standard structure.
        Expected structure:
        - images/
          - train/
          - val/
          - test/
        - labels/
          - train/
          - val/
          - test/

        Args:
            dataset_path: Path to dataset

        Returns:
            Organization result
        """
        try:
            dataset_path = Path(dataset_path)

            if not dataset_path.exists():
                return {"success": False, "error": "Dataset path not found"}

            # Check for standard YOLO structure
            required_dirs = [
                dataset_path / "images" / "train",
                dataset_path / "images" / "val",
                dataset_path / "labels" / "train",
                dataset_path / "labels" / "val",
            ]

            structure_valid = all(d.exists() for d in required_dirs)

            if structure_valid:
                # Count images and labels
                train_images = len(
                    list((dataset_path / "images" / "train").glob("*"))
                )
                val_images = len(
                    list((dataset_path / "images" / "val").glob("*"))
                )
                train_labels = len(
                    list((dataset_path / "labels" / "train").glob("*.txt"))
                )
                val_labels = len(
                    list((dataset_path / "labels" / "val").glob("*.txt"))
                )

                result = {
                    "success": True,
                    "structure": "valid",
                    "train_images": train_images,
                    "val_images": val_images,
                    "train_labels": train_labels,
                    "val_labels": val_labels,
                }
            else:
                # Try to organize into proper structure
                result = self._reorganize_yolo_files(dataset_path)

            return result

        except Exception as e:
            logger.error(f"Error organizing YOLO dataset: {e}")
            return {"success": False, "error": str(e)}

    def _reorganize_yolo_files(self, dataset_path: Path) -> Dict:
        """Reorganize YOLO files into standard structure."""
        try:
            # Create standard directories
            (dataset_path / "images" / "train").mkdir(
                parents=True, exist_ok=True
            )
            (dataset_path / "images" / "val").mkdir(
                parents=True, exist_ok=True
            )
            (dataset_path / "labels" / "train").mkdir(
                parents=True, exist_ok=True
            )
            (dataset_path / "labels" / "val").mkdir(
                parents=True, exist_ok=True
            )

            # Find all images and labels
            image_extensions = [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            ]
            all_images = []
            for ext in image_extensions:
                all_images.extend(
                    dataset_path.rglob(f"*{ext}")
                )

            # Labels are discovered per-image when copying; skip precomputing list

            # Split 80/20 for train/val
            import random

            random.shuffle(all_images)
            split_idx = int(len(all_images) * 0.8)

            train_images = all_images[:split_idx]
            val_images = all_images[split_idx:]

            # Copy files to proper locations
            for img in train_images:
                shutil.copy(img, dataset_path / "images" / "train" / img.name)
                # Try to find corresponding label
                label = img.with_suffix(".txt")
                if label.exists():
                    shutil.copy(
                        label, dataset_path / "labels" / "train" / label.name
                    )

            for img in val_images:
                shutil.copy(img, dataset_path / "images" / "val" / img.name)
                label = img.with_suffix(".txt")
                if label.exists():
                    shutil.copy(
                        label, dataset_path / "labels" / "val" / label.name
                    )

            return {
                "success": True,
                "structure": "reorganized",
                "train_images": len(train_images),
                "val_images": len(val_images),
            }

        except Exception as e:
            logger.error(f"Error reorganizing files: {e}")
            return {"success": False, "error": str(e)}

    def organize_padim_dataset(self, dataset_path: str) -> Dict:
        """
        Organize PaDiM dataset into standard structure.
        Expected structure:
        - train/
          - good/
        - test/
          - good/
          - defect_type_1/
          - defect_type_2/

        Args:
            dataset_path: Path to dataset

        Returns:
            Organization result
        """
        try:
            dataset_path = Path(dataset_path)

            if not dataset_path.exists():
                return {"success": False, "error": "Dataset path not found"}

            # Check for standard PaDiM structure
            train_dir = dataset_path / "train"
            test_dir = dataset_path / "test"

            if train_dir.exists() and test_dir.exists():
                # Count images in each category
                train_good = (
                    len(list((train_dir / "good").glob("*")))
                    if (train_dir / "good").exists()
                    else 0
                )

                test_categories = {}
                if test_dir.exists():
                    for category_dir in test_dir.iterdir():
                        if category_dir.is_dir():
                            test_categories[category_dir.name] = len(
                                list(category_dir.glob("*"))
                            )

                return {
                    "success": True,
                    "structure": "valid",
                    "train_good": train_good,
                    "test_categories": test_categories,
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid PaDiM dataset structure",
                }

        except Exception as e:
            logger.error(f"Error organizing PaDiM dataset: {e}")
            return {"success": False, "error": str(e)}

    def create_data_yaml(
        self,
        dataset_path: str,
        class_names: List[str],
        nc: Optional[int] = None,
    ) -> str:
        """
        Create data.yaml file for YOLO training.

        Args:
            dataset_path: Path to dataset
            class_names: List of class names
            nc: Number of classes (auto-detected if None)
        Returns:
            Path to created YAML file
        """
        try:
            dataset_path = Path(dataset_path)

            if nc is None:
                nc = len(class_names)

            yaml_content = {
                "path": str(dataset_path.absolute()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": nc,
                "names": class_names,
            }

            yaml_path = dataset_path / "data.yaml"

            with open(yaml_path, "w") as f:
                import yaml

                yaml.dump(yaml_content, f, default_flow_style=False)

            logger.info(f"Created data.yaml at {yaml_path}")
            return str(yaml_path)

        except Exception as e:
            logger.error(f"Error creating data.yaml: {e}")
            return None

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a registered dataset."""
        return self.datasets.get(dataset_name)

    def list_datasets(self) -> List[Dict]:
        """List all registered datasets."""
        return list(self.datasets.values())

    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset.

        Args:
            dataset_name: Name of dataset to delete

        Returns:
            Success status
        """
        try:
            if dataset_name in self.datasets:
                dataset_path = Path(self.datasets[dataset_name]["path"])
                if dataset_path.exists():
                    shutil.rmtree(dataset_path)
                del self.datasets[dataset_name]
                logger.info(f"Deleted dataset: {dataset_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            return False
