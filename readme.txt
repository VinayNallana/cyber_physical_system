# Cyber-Physical System with YOLOv9 and PaDiM

## Overview

This project implements a modular Cyber-Physical System (CPS) for industrial AI applications, integrating real-time object detection and visual anomaly detection. 

- **Object Detection:** Uses YOLOv9 for detecting multiple objects in real time.
- **Anomaly Detection:** Uses Patch Distribution Modeling (PaDiM) for detecting visual anomalies.
- **Layered Architecture:** Physical/Sensor Layer, Communication Layer, Digital Layer, Dataset Management, AI Models, Metrics, and GUI Dashboard.
- **Extensible:** Designed for deployment in industrial environments with options for cloud, local, or edge deployment.

---

## Features

- Real-time multi-object detection with YOLOv9.
- Visual anomaly detection with PaDiM.
- Robust communication layer for inter-device messaging.
- Decision engine for autonomous control based on detections.
- Analytics engine for system health and performance metrics.
- Interactive GUI dashboard built with Tkinter.
- Dataset management with ZIP handling and organization.
- Multi-layer Python architecture for clean separation of concerns. 

---

## Project Structure

```
cyber_physical_system/
├── core/                       # System integration and main entry point
├── dataset/                    # Dataset extraction and organization
├── digital_layer/              # Decision engine and analytics
├── gui/                        # GUI dashboard
├── models/                     # YOLOv9 and PaDiM model implementations
├── metrics/                    # Metrics tracking and reporting
├── physical_layer/             # Sensors and actuators interfaces
├── communication/              # Message broker and protocols
├── main.py                    # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- GPU (Optional for training but recommended)
- Dependencies installed (can use `requirements.txt`)

### Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/cyber-physical-system.git
cd cyber-physical-system
```

Install dependencies:

```
pip install -r requirements.txt
```

### Running the System

This project supports running the full pipeline locally or on Google Colab with GPU acceleration.

- **Start the system:**

```
python main.py
```

- **Or launch the GUI dashboard:**

```
from cyber_physical_system.gui.dashboard import CPSDashboard
from cyber_physical_system.core.cps_system import CyberPhysicalSystem

cps = CyberPhysicalSystem()
dashboard = CPSDashboard(cps)
dashboard.run()
```

---

## Usage

### Training Models

- Train YOLOv9 for object detection with your dataset.
- Train PaDiM for anomaly detection on normal images.

See `examples/train_yolo_example.py` and `examples/train_padim_example.py` for step-by-step training scripts.

### Inference

Use the inference script to run detection and anomaly models on test images.

---

## Deployment on Google Colab

- Clone this repository in a Colab notebook.
- Install dependencies using `pip install -r requirements.txt`.
- Mount Google Drive and set datasets path.
- Run training or inference scripts in cells.

Example notebook and step-by-step deployment instructions are provided in documentation.

---

## Contributing

Contributions are welcome! Please:

- Fork the repo.
- Create a feature branch.
- Submit pull requests with clear descriptions.

---

## License

Specify your license (e.g., MIT License).

---

## Contact

For support or inquiries, contact:

- Your Name: your.email@example.com
- GitHub: https://github.com/YOUR_USERNAME

---

## References

- YOLOv9: Ultralytics YOLO
- PaDiM: Patch Distribution Modeling paper and implementation

