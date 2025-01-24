from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
import psutil
import logging
from typing import Optional
import threading
import time

logger = logging.getLogger(__name__)

class MLOpsMetrics:
    def __init__(self):
        # Create a custom registry for this instance
        self.registry = CollectorRegistry()
        
        # Training metrics
        self.training_steps = Counter(
            'training_steps_total',
            'Total number of training steps completed',
            registry=self.registry
        )
        self.training_loss = Gauge(
            'training_loss',
            'Current training loss',
            registry=self.registry
        )
        self.validation_loss = Gauge(
            'validation_loss',
            'Current validation loss',
            registry=self.registry
        )
        self.epoch_progress = Gauge(
            'epoch_progress',
            'Current epoch progress',
            registry=self.registry
        )
        
        # Data validation metrics
        self.validation_checks = Counter(
            'data_validation_checks_total',
            'Total number of data validation checks performed',
            ['check_name', 'status'],
            registry=self.registry
        )
        
        # Model metrics
        self.inference_latency = Histogram(
            'model_inference_latency_seconds',
            'Time taken for model inference',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
            registry=self.registry
        )
        
        # System metrics
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory currently in use',
            registry=self.registry
        )
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        self.ram_usage = Gauge(
            'ram_usage_bytes',
            'RAM memory currently in use',
            registry=self.registry
        )
        
        # Start system metrics collection
        self._start_system_metrics_collection()
    
    def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while True:
            try:
                # CPU usage
                self.cpu_usage.set(psutil.cpu_percent())
                
                # RAM usage
                ram = psutil.virtual_memory()
                self.ram_usage.set(ram.used)
                
                # GPU metrics (if available)
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated()
                        self.gpu_memory_used.set(memory_allocated)
                except ImportError:
                    pass
                
                time.sleep(15)  # Collect every 15 seconds
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _start_system_metrics_collection(self):
        """Start system metrics collection in background thread"""
        thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True
        )
        thread.start()
    
    def record_validation_check(self, check_name: str, success: bool):
        """Record a data validation check result"""
        status = "success" if success else "failure"
        self.validation_checks.labels(check_name=check_name, status=status).inc()
    
    def record_training_step(self, loss: Optional[float] = None):
        """Record a training step completion"""
        self.training_steps.inc()
        if loss is not None:
            self.training_loss.set(loss)
    
    def record_validation_loss(self, loss: float):
        """Record validation loss"""
        self.validation_loss.set(loss)
    
    def record_epoch_progress(self, progress: float):
        """Record epoch progress (0-1)"""
        self.epoch_progress.set(progress)
    
    def time_inference(self):
        """Context manager to time model inference"""
        return self.inference_latency.time() 