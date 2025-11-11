# cyber_physical_system/gui/dashboard.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from pathlib import Path
from typing import Optional, Dict
import logging
import cv2
from PIL import Image, ImageTk
import numpy as np

logger = logging.getLogger(__name__)


class CPSDashboard:
    """
    Interactive GUI Dashboard for Cyber-Physical System.
    Provides real-time monitoring, training controls, and visualization.
    """
    
    def __init__(self, cps_system):
        """
        Initialize dashboard.
        
        Args:
            cps_system: Main CPS system instance
        """
        self.cps_system = cps_system
        self.root = tk.Tk()
        self.root.title("Cyber-Physical System Dashboard - YOLOv9 & PaDiM")
        self.root.geometry("1400x900")
        
        # State variables
        self.is_monitoring = False
        self.current_image = None
        self.update_interval = 1000  # ms
        
        # Setup GUI components
        self._setup_gui()
        
        logger.info("Dashboard initialized")
    
    def _setup_gui(self) -> None:
        """Setup all GUI components."""
        # Create main container with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_detection = ttk.Frame(self.notebook)
        self.tab_anomaly = ttk.Frame(self.notebook)
        self.tab_training = ttk.Frame(self.notebook)
        self.tab_metrics = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_overview, text="System Overview")
        self.notebook.add(self.tab_detection, text="Object Detection")
        self.notebook.add(self.tab_anomaly, text="Anomaly Detection")
        self.notebook.add(self.tab_training, text="Training")
        self.notebook.add(self.tab_metrics, text="Metrics & Analytics")
        
        # Setup each tab
        self._setup_overview_tab()
        self._setup_detection_tab()
        self._setup_anomaly_tab()
        self._setup_training_tab()
        self._setup_metrics_tab()
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="System Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_overview_tab(self) -> None:
        """Setup system overview tab."""
        # Title
        title_label = tk.Label(self.tab_overview, text="Cyber-Physical System Overview", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # System status frame
        status_frame = ttk.LabelFrame(self.tab_overview, text="System Status", padding=10)
        status_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        # Create grid for status indicators
        self.status_labels = {}
        status_items = [
            ("System Status", "Idle"),
            ("YOLOv9 Status", "Not Loaded"),
            ("PaDiM Status", "Not Loaded"),
            ("Communication", "Ready"),
            ("Uptime", "0:00:00")
        ]
        
        for i, (label, initial_value) in enumerate(status_items):
            tk.Label(status_frame, text=f"{label}:", font=("Arial", 10, "bold")).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            value_label = tk.Label(status_frame, text=initial_value, font=("Arial", 10))
            value_label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.status_labels[label] = value_label
        
        # Control buttons frame
        control_frame = ttk.LabelFrame(self.tab_overview, text="System Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        btn_start = ttk.Button(control_frame, text="Start System", command=self._start_system)
        btn_start.pack(side=tk.LEFT, padx=5)
        
        btn_stop = ttk.Button(control_frame, text="Stop System", command=self._stop_system)
        btn_stop.pack(side=tk.LEFT, padx=5)
        
        btn_reset = ttk.Button(control_frame, text="Reset Metrics", command=self._reset_metrics)
        btn_reset.pack(side=tk.LEFT, padx=5)
        
        # Recent activity log
        log_frame = ttk.LabelFrame(self.tab_overview, text="Activity Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.activity_log = tk.Text(log_frame, height=15, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_frame, command=self.activity_log.yview)
        self.activity_log.configure(yscrollcommand=scrollbar.set)
        
        self.activity_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_detection_tab(self) -> None:
        """Setup object detection tab."""
        # Control panel
        control_panel = ttk.LabelFrame(self.tab_detection, text="Detection Controls", padding=10)
        control_panel.pack(fill=tk.X, padx=10, pady=5)
        
        # Image selection
        tk.Label(control_panel, text="Image:").grid(row=0, column=0, padx=5, pady=5)
        self.detection_image_path = tk.StringVar()
        entry_image = ttk.Entry(control_panel, textvariable=self.detection_image_path, width=50)
        entry_image.grid(row=0, column=1, padx=5, pady=5)
        
        btn_browse = ttk.Button(control_panel, text="Browse", command=self._browse_detection_image)
        btn_browse.grid(row=0, column=2, padx=5, pady=5)
        
        # Confidence threshold
        tk.Label(control_panel, text="Confidence:").grid(row=1, column=0, padx=5, pady=5)
        self.detection_confidence = tk.DoubleVar(value=0.5)
        scale_conf = ttk.Scale(control_panel, from_=0.1, to=1.0, variable=self.detection_confidence, 
                              orient=tk.HORIZONTAL, length=200)
        scale_conf.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        tk.Label(control_panel, textvariable=self.detection_confidence).grid(row=1, column=2, padx=5, pady=5)
        
        # Detect button
        btn_detect = ttk.Button(control_panel, text="Run Detection", command=self._run_detection)
        btn_detect.grid(row=2, column=1, padx=5, pady=10)
        
        # Results display
        results_frame = ttk.Frame(self.tab_detection)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Image display (left side)
        image_frame = ttk.LabelFrame(results_frame, text="Image", padding=5)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.detection_image_label = tk.Label(image_frame)
        self.detection_image_label.pack()
        
        # Results text (right side)
        text_frame = ttk.LabelFrame(results_frame, text="Detection Results", padding=5)
        text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.detection_results_text = tk.Text(text_frame, height=20, width=40)
        self.detection_results_text.pack(fill=tk.BOTH, expand=True)
    
    def _setup_anomaly_tab(self) -> None:
        """Setup anomaly detection tab."""
        # Control panel
        control_panel = ttk.LabelFrame(self.tab_anomaly, text="Anomaly Detection Controls", padding=10)
        control_panel.pack(fill=tk.X, padx=10, pady=5)
        
        # Image selection
        tk.Label(control_panel, text="Image:").grid(row=0, column=0, padx=5, pady=5)
        self.anomaly_image_path = tk.StringVar()
        entry_image = ttk.Entry(control_panel, textvariable=self.anomaly_image_path, width=50)
        entry_image.grid(row=0, column=1, padx=5, pady=5)
        
        btn_browse = ttk.Button(control_panel, text="Browse", command=self._browse_anomaly_image)
        btn_browse.grid(row=0, column=2, padx=5, pady=5)
        
        # Detect button
        btn_detect = ttk.Button(control_panel, text="Detect Anomalies", command=self._run_anomaly_detection)
        btn_detect.grid(row=1, column=1, padx=5, pady=10)
        
        # Results display
        results_frame = ttk.Frame(self.tab_anomaly)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Original image (left)
        orig_frame = ttk.LabelFrame(results_frame, text="Original Image", padding=5)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.anomaly_orig_label = tk.Label(orig_frame)
        self.anomaly_orig_label.pack()
        
        # Anomaly map (right)
        map_frame = ttk.LabelFrame(results_frame, text="Anomaly Map", padding=5)
        map_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.anomaly_map_label = tk.Label(map_frame)
        self.anomaly_map_label.pack()
        
        # Results text
        text_frame = ttk.LabelFrame(self.tab_anomaly, text="Anomaly Results", padding=5)
        text_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.anomaly_results_text = tk.Text(text_frame, height=8)
        self.anomaly_results_text.pack(fill=tk.BOTH, expand=True)
    
    def _setup_training_tab(self) -> None:
        """Setup training tab."""
        # YOLOv9 Training
        yolo_frame = ttk.LabelFrame(self.tab_training, text="YOLOv9 Training", padding=10)
        yolo_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        tk.Label(yolo_frame, text="Dataset Path:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.yolo_dataset_path = tk.StringVar()
        ttk.Entry(yolo_frame, textvariable=self.yolo_dataset_path, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(yolo_frame, text="Browse", command=self._browse_yolo_dataset).grid(row=0, column=2, padx=5, pady=5)
        
        tk.Label(yolo_frame, text="Epochs:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.yolo_epochs = tk.IntVar(value=100)
        ttk.Spinbox(yolo_frame, from_=1, to=500, textvariable=self.yolo_epochs, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        tk.Label(yolo_frame, text="Batch Size:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.yolo_batch = tk.IntVar(value=16)
        ttk.Spinbox(yolo_frame, from_=1, to=64, textvariable=self.yolo_batch, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(yolo_frame, text="Start Training", command=self._train_yolo).grid(row=3, column=1, padx=5, pady=10)
        
        # PaDiM Training
        padim_frame = ttk.LabelFrame(self.tab_training, text="PaDiM Training", padding=10)
        padim_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        tk.Label(padim_frame, text="Training Images:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.padim_dataset_path = tk.StringVar()
        ttk.Entry(padim_frame, textvariable=self.padim_dataset_path, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(padim_frame, text="Browse", command=self._browse_padim_dataset).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(padim_frame, text="Start Training", command=self._train_padim).grid(row=1, column=1, padx=5, pady=10)
        
        # Training progress
        progress_frame = ttk.LabelFrame(self.tab_training, text="Training Progress", padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.training_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.training_progress.pack(fill=tk.X, pady=5)
        
        self.training_log = tk.Text(progress_frame, height=15)
        self.training_log.pack(fill=tk.BOTH, expand=True)
    
    def _setup_metrics_tab(self) -> None:
        """Setup metrics and analytics tab."""
        # Metrics display
        metrics_frame = ttk.LabelFrame(self.tab_metrics, text="System Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        # Detection metrics
        det_frame = ttk.Frame(metrics_frame)
        det_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(det_frame, text="Detection Metrics", font=("Arial", 12, "bold")).pack()
        self.detection_metrics_text = tk.Text(det_frame, height=8)
        self.detection_metrics_text.pack(fill=tk.X)
        
        # Anomaly metrics
        anom_frame = ttk.Frame(metrics_frame)
        anom_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(anom_frame, text="Anomaly Metrics", font=("Arial", 12, "bold")).pack()
        self.anomaly_metrics_text = tk.Text(anom_frame, height=8)
        self.anomaly_metrics_text.pack(fill=tk.X)
        
        # Charts frame
        charts_frame = ttk.LabelFrame(self.tab_metrics, text="Performance Charts", padding=10)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        ttk.Button(self.tab_metrics, text="Refresh Metrics", command=self._refresh_metrics).pack(pady=5)
    
    def _browse_detection_image(self) -> None:
        """Browse for detection image."""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if filename:
            self.detection_image_path.set(filename)
    
    def _browse_anomaly_image(self) -> None:
        """Browse for anomaly detection image."""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if filename:
            self.anomaly_image_path.set(filename)
    
    def _browse_yolo_dataset(self) -> None:
        """Browse for YOLO dataset."""
        dirname = filedialog.askdirectory(title="Select YOLO Dataset Directory")
        if dirname:
            self.yolo_dataset_path.set(dirname)
    
    def _browse_padim_dataset(self) -> None:
        """Browse for PaDiM dataset."""
        dirname = filedialog.askdirectory(title="Select PaDiM Training Images Directory")
        if dirname:
            self.padim_dataset_path.set(dirname)
    
    def _run_detection(self) -> None:
        """Run object detection."""
        image_path = self.detection_image_path.get()
        if not image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            self.status_bar.config(text="Running detection...")
            self.root.update()
            
            # Run detection
            confidence = self.detection_confidence.get()
            result = self.cps_system.yolo_detector.predict(image_path, conf_threshold=confidence)
            
            if result.get("success"):
                # Display image with detections
                image = cv2.imread(image_path)
                annotated = self.cps_system.yolo_detector.visualize_detections(image, result["detections"])
                self._display_image(annotated, self.detection_image_label, max_size=(600, 400))
                
                # Display results text
                self.detection_results_text.delete(1.0, tk.END)
                self.detection_results_text.insert(tk.END, f"Detections: {result['num_detections']}\n")
                self.detection_results_text.insert(tk.END, f"Inference Time: {result['inference_time_ms']:.2f} ms\n\n")
                
                for i, det in enumerate(result["detections"], 1):
                    self.detection_results_text.insert(tk.END, 
                        f"{i}. {det['class_name']}: {det['confidence']:.2f}\n")
                
                self.status_bar.config(text="Detection completed")
            else:
                messagebox.showerror("Error", result.get("error", "Detection failed"))
                self.status_bar.config(text="Detection failed")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_bar.config(text="Error occurred")
    
    def _run_anomaly_detection(self) -> None:
        """Run anomaly detection."""
        image_path = self.anomaly_image_path.get()
        if not image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            self.status_bar.config(text="Running anomaly detection...")
            self.root.update()
            
            # Run anomaly detection
            result = self.cps_system.padim_detector.predict(image_path)
            
            if result.get("success"):
                # Display original image
                image = cv2.imread(image_path)
                self._display_image(image, self.anomaly_orig_label, max_size=(400, 400))
                
                # Display anomaly map
                anomaly_map = result["anomaly_map"]
                visualization = self.cps_system.padim_detector.visualize_anomaly_map(image, anomaly_map)
                self._display_image(visualization, self.anomaly_map_label, max_size=(400, 400))
                
                # Display results text
                self.anomaly_results_text.delete(1.0, tk.END)
                self.anomaly_results_text.insert(tk.END, f"Anomaly Score: {result['anomaly_score']:.4f}\n")
                self.anomaly_results_text.insert(tk.END, f"Is Anomalous: {result['is_anomalous']}\n")
                self.anomaly_results_text.insert(tk.END, f"Severity: {result['severity_level']}\n")
                self.anomaly_results_text.insert(tk.END, f"Inference Time: {result['inference_time_ms']:.2f} ms\n")
                
                self.status_bar.config(text="Anomaly detection completed")
            else:
                messagebox.showerror("Error", result.get("error", "Detection failed"))
                self.status_bar.config(text="Detection failed")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_bar.config(text="Error occurred")
    
    def _display_image(self, image: np.ndarray, label: tk.Label, max_size: tuple = (600, 400)) -> None:
        """Display image in label widget."""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize if needed
        h, w = image_rgb.shape[:2]
        max_w, max_h = max_size
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(image_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        label.config(image=img_tk)
        label.image = img_tk  # Keep reference
    
    def _train_yolo(self) -> None:
        """Start YOLOv9 training."""
        dataset_path = self.yolo_dataset_path.get()
        if not dataset_path:
            messagebox.showerror("Error", "Please select a dataset")
            return
        
        # Start training in separate thread
        def train_thread():
            self.training_progress.start()
            self._log_training("Starting YOLOv9 training...")
            
            try:
                # Create data.yaml if needed
                data_yaml = Path(dataset_path) / "data.yaml"
                if not data_yaml.exists():
                    messagebox.showwarning("Warning", "data.yaml not found. Please create it first.")
                    return
                
                result = self.cps_system.yolo_detector.train(
                    data_yaml=str(data_yaml),
                    epochs=self.yolo_epochs.get(),
                    batch_size=self.yolo_batch.get()
                )
                
                if result.get("success"):
                    self._log_training("Training completed successfully!")
                    self._log_training(f"Final mAP50: {result['final_metrics']['mAP50']:.4f}")
                    messagebox.showinfo("Success", "Training completed successfully")
                else:
                    self._log_training(f"Training failed: {result.get('error')}")
                    messagebox.showerror("Error", result.get("error"))
                    
            except Exception as e:
                self._log_training(f"Error: {str(e)}")
                messagebox.showerror("Error", str(e))
            finally:
                self.training_progress.stop()
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def _train_padim(self) -> None:
        """Start PaDiM training."""
        dataset_path = self.padim_dataset_path.get()
        if not dataset_path:
            messagebox.showerror("Error", "Please select a dataset")
            return
        
        # Start training in separate thread
        def train_thread():
            self.training_progress.start()
            self._log_training("Starting PaDiM training...")
            
            try:
                # Get all image paths
                image_paths = list(Path(dataset_path).glob("*.jpg"))
                image_paths.extend(list(Path(dataset_path).glob("*.png")))
                
                if not image_paths:
                    messagebox.showerror("Error", "No images found in directory")
                    return
                
                result = self.cps_system.padim_detector.train([str(p) for p in image_paths])
                
                if result.get("success"):
                    self._log_training("Training completed successfully!")
                    self._log_training(f"Trained on {result['training_images']} images")
                    messagebox.showinfo("Success", "Training completed successfully")
                else:
                    self._log_training(f"Training failed: {result.get('error')}")
                    messagebox.showerror("Error", result.get("error"))
                    
            except Exception as e:
                self._log_training(f"Error: {str(e)}")
                messagebox.showerror("Error", str(e))
            finally:
                self.training_progress.stop()
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def _log_training(self, message: str) -> None:
        """Log training message."""
        self.training_log.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.training_log.see(tk.END)
    
    def _start_system(self) -> None:
        """Start the CPS system."""
        try:
            self.cps_system.start()
            self._log_activity("System started")
            self.status_bar.config(text="System running")
            self.is_monitoring = True
            self._update_status()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _stop_system(self) -> None:
        """Stop the CPS system."""
        try:
            self.cps_system.stop()
            self._log_activity("System stopped")
            self.status_bar.config(text="System stopped")
            self.is_monitoring = False
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _reset_metrics(self) -> None:
        """Reset system metrics."""
        if messagebox.askyesno("Confirm", "Reset all metrics?"):
            self.cps_system.metrics_tracker.reset_metrics()
            self._log_activity("Metrics reset")
            self._refresh_metrics()
    
    def _refresh_metrics(self) -> None:
        """Refresh metrics display."""
        try:
            # Get metrics
            detection_metrics = self.cps_system.metrics_tracker.get_detection_summary()
            anomaly_metrics = self.cps_system.metrics_tracker.get_anomaly_summary()
            
            # Update detection metrics
            self.detection_metrics_text.delete(1.0, tk.END)
            self.detection_metrics_text.insert(tk.END, f"Total Detections: {detection_metrics['total_detections']}\n")
            self.detection_metrics_text.insert(tk.END, f"Avg Detections/Frame: {detection_metrics['avg_detections_per_frame']:.2f}\n")
            self.detection_metrics_text.insert(tk.END, f"Avg Inference Time: {detection_metrics['avg_inference_time_ms']:.2f} ms\n")
            self.detection_metrics_text.insert(tk.END, f"Avg Confidence: {detection_metrics['avg_confidence']:.2f}\n")
            
            # Update anomaly metrics
            self.anomaly_metrics_text.delete(1.0, tk.END)
            self.anomaly_metrics_text.insert(tk.END, f"Total Predictions: {anomaly_metrics['total_predictions']}\n")
            self.anomaly_metrics_text.insert(tk.END, f"Anomaly Count: {anomaly_metrics['anomaly_count']}\n")
            self.anomaly_metrics_text.insert(tk.END, f"Anomaly Rate: {anomaly_metrics['anomaly_rate']:.2%}\n")
            self.anomaly_metrics_text.insert(tk.END, f"Avg Score: {anomaly_metrics['avg_anomaly_score']:.4f}\n")
            
            # Update charts
            self._update_charts()
            
        except Exception as e:
            logger.error(f"Error refreshing metrics: {e}")
    
    def _update_charts(self) -> None:
        """Update performance charts."""
        self.fig.clear()
        
        detection_metrics = self.cps_system.metrics_tracker.detection_metrics
        anomaly_metrics = self.cps_system.metrics_tracker.anomaly_metrics
        
        # Detection inference times
        ax1 = self.fig.add_subplot(131)
        if detection_metrics["inference_times"]:
            ax1.plot(list(detection_metrics["inference_times"]))
            ax1.set_title("Detection Inference Time")
            ax1.set_ylabel("Time (ms)")
        
        # Anomaly scores
        ax2 = self.fig.add_subplot(132)
        if anomaly_metrics["anomaly_scores"]:
            ax2.plot(list(anomaly_metrics["anomaly_scores"]))
            ax2.set_title("Anomaly Scores")
            ax2.set_ylabel("Score")
        
        # Severity distribution
        ax3 = self.fig.add_subplot(133)
        severity_counts = anomaly_metrics["severity_counts"]
        if sum(severity_counts.values()) > 0:
            ax3.bar(severity_counts.keys(), severity_counts.values())
            ax3.set_title("Anomaly Severity Distribution")
            ax3.set_ylabel("Count")
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _update_status(self) -> None:
        """Update status labels."""
        if not self.is_monitoring:
            return
        
        try:
            # Update uptime
            system_metrics = self.cps_system.metrics_tracker.get_system_summary()
            self.status_labels["Uptime"].config(text=system_metrics["uptime_formatted"])
            
            # Update model status
            if self.cps_system.yolo_detector.is_trained:
                self.status_labels["YOLOv9 Status"].config(text="Ready", fg="green")
            
            if self.cps_system.padim_detector.is_trained:
                self.status_labels["PaDiM Status"].config(text="Ready", fg="green")
            
            # Schedule next update
            self.root.after(self.update_interval, self._update_status)
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def _log_activity(self, message: str) -> None:
        """Log activity message."""
        self.activity_log.config(state=tk.NORMAL)
        self.activity_log.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.activity_log.see(tk.END)
        self.activity_log.config(state=tk.DISABLED)
    
    def run(self) -> None:
        """Run the dashboard."""
        logger.info("Starting dashboard...")
        self.root.mainloop()


