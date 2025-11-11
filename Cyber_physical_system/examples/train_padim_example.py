# examples/train_padim_example.py

from cyber_physical_system.core.cps_system import CyberPhysicalSystem
from pathlib import Path


def train_padim_example():
    """Example: Train PaDiM on normal images."""

    # Initialize system
    cps = CyberPhysicalSystem()

    # Get training images
    train_dir = Path("path/to/normal/images")
    train_images = list(train_dir.glob("*.jpg"))
    train_images.extend(list(train_dir.glob("*.png")))

    print(f"Found {len(train_images)} training images")

    # Train model
    print("Starting PaDiM training...")
    result = cps.train_padim([str(img) for img in train_images])

    if result["success"]:
        print("Training completed successfully!")
        print(f"Training time: {result['training_time_seconds']:.2f} seconds")

        # Save model
        save_path = "./models/padim_model.pkl"
        if cps.save_padim_model(save_path):
            print(f"Model saved to {save_path}")
    else:
        print(f"Training failed: {result.get('error')}")


if __name__ == "__main__":
    train_padim_example()
