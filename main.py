# main.py

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from cyber_physical_system.core.cps_system import CyberPhysicalSystem
from cyber_physical_system.gui.dashboard import CPSDashboard
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main entry point for the CPS application."""
    
    # Configuration
    config = {
        'device': 'auto',  # 'cpu', 'cuda', or 'auto'
        'padim_backbone': 'resnet18',  # 'resnet18' or 'resnet50'
        'dataset_dir': './datasets',
        'metrics_window': 1000,
        'analytics_window': 100,
        'decision_engine': {
            'anomaly_threshold': 0.7,
            'detection_confidence': 0.5
        }
    }
    
    # Initialize CPS system
    print("Initializing Cyber-Physical System...")
    cps_system = CyberPhysicalSystem(config=config)
    
    # Initialize and run GUI dashboard
    print("Launching GUI Dashboard...")
    dashboard = CPSDashboard(cps_system)
    dashboard.run()


if __name__ == "__main__":
    main()
