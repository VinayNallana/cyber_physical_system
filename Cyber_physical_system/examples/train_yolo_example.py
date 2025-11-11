# examples/train_yolo_example.py

from cyber_physical_system.core.cps_system import CyberPhysicalSystem


def train_yolo_example():
    """Example: Train YOLOv9 on custom dataset."""

    # Initialize system
    cps = CyberPhysicalSystem()

    # Extract dataset from ZIP
    print("Extracting dataset...")
    result = cps.dataset_manager.extract_zip(
        zip_path="path/to/your/dataset.zip", dataset_type="yolo"
    )

    if result["success"]:
        dataset_path = result["dataset_info"]["path"]
        print(f"Dataset extracted to: {dataset_path}")

        # Organize dataset
        print("Organizing dataset...")
        org_result = cps.dataset_manager.organize_yolo_dataset(dataset_path)
        print(f"Organization result: {org_result}")

        # Create data.yaml
        class_names = ["person", "car", "truck", "bicycle"]
        yaml_path = cps.dataset_manager.create_data_yaml(
            dataset_path=dataset_path, class_names=class_names
        )

        # Train model
        print("Starting training...")
        train_result = cps.train_yolo(
            data_yaml=yaml_path, epochs=50, batch_size=16, imgsz=640
        )

        if train_result["success"]:
            print("Training completed successfully!")
            print(f"Final mAP50: {train_result['final_metrics']['mAP50']:.4f}")
        else:
            print(f"Training failed: {train_result.get('error')}")
    else:
        print(f"Failed to extract dataset: {result.get('error')}")


if __name__ == "__main__":
    train_yolo_example()
