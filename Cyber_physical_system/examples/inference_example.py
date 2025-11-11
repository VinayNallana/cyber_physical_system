# examples/inference_example.py

from cyber_physical_system.core.cps_system import CyberPhysicalSystem


def inference_example():
    """Example: Run inference with both models."""

    # Initialize system
    cps = CyberPhysicalSystem()

    # Load pretrained models
    print("Loading models...")
    cps.load_yolo_model("path/to/yolo/model.pt", class_names=["person", "car"])
    cps.load_padim_model("path/to/padim/model.pkl")

    # Start system
    cps.start()

    # Process image with YOLOv9
    print("\nRunning object detection...")
    image_path = "path/to/test/image.jpg"
    detection_result = cps.process_image_yolo(image_path, conf_threshold=0.5)

    if detection_result["success"]:
        print(f"Detected {detection_result['num_detections']} objects")
        print(
            f"Inference time: {detection_result['inference_time_ms']:.2f} ms"
        )

        for det in detection_result["detections"]:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")

    # Process image with PaDiM
    print("\nRunning anomaly detection...")
    anomaly_result = cps.process_image_padim(image_path)

    if anomaly_result["success"]:
        print(f"Anomaly score: {anomaly_result['anomaly_score']:.4f}")
        print(f"Is anomalous: {anomaly_result['is_anomalous']}")
        print(f"Severity: {anomaly_result['severity_level']}")
        print(f"Inference time: {anomaly_result['inference_time_ms']:.2f} ms")
    # Get system status
    print("\nSystem status:")
    status = cps.get_system_status()
    print(f"System running: {status['is_running']}")
    print(f"YOLO trained: {status['models']['yolo_trained']}")
    print(f"PaDiM trained: {status['models']['padim_trained']}")
    # Stop system
    cps.stop()


if __name__ == "__main__":
    inference_example()
