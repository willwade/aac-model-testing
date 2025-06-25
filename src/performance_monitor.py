"""
Performance Monitor

This module provides comprehensive performance monitoring for model testing,
including memory usage, CPU utilization, response time, and system metrics.
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PerformanceSnapshot:
    """A snapshot of system performance metrics at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for AAC model testing.
    
    This class monitors system resources during model execution to provide
    insights into performance characteristics of different models.
    """
    
    def __init__(self, sampling_interval: float = 0.5):
        """
        Initialize the performance monitor.
        
        Args:
            sampling_interval: How often to sample metrics (in seconds)
        """
        self.sampling_interval = sampling_interval
        self.logger = logging.getLogger("AAC_Testing.PerformanceMonitor")
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._snapshots: List[PerformanceSnapshot] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        
        # Initial system state
        self._initial_snapshot: Optional[PerformanceSnapshot] = None
        
        # Process tracking
        self._process = psutil.Process()
        
        self.logger.info("Performance monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self._monitoring:
            self.logger.warning("Performance monitoring already active")
            return
        
        self.logger.info("Starting performance monitoring")
        
        # Reset state
        self._snapshots.clear()
        self._start_time = time.time()
        self._end_time = None
        
        # Take initial snapshot
        self._initial_snapshot = self._take_snapshot()
        
        # Start monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop performance monitoring and return metrics.
        
        Returns:
            Dictionary containing comprehensive performance metrics
        """
        if not self._monitoring:
            self.logger.warning("Performance monitoring not active")
            return {}
        
        self.logger.info("Stopping performance monitoring")
        
        # Stop monitoring
        self._monitoring = False
        self._end_time = time.time()
        
        # Wait for monitor thread to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        # Calculate and return metrics
        metrics = self._calculate_metrics()
        
        self.logger.info(f"Performance monitoring completed. Duration: {metrics.get('duration', 0):.2f}s")
        
        return metrics
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self._monitoring:
            try:
                snapshot = self._take_snapshot()
                self._snapshots.append(snapshot)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.sampling_interval)
    
    def _take_snapshot(self) -> PerformanceSnapshot:
        """Take a snapshot of current system performance."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            memory_percent = memory_info.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
            network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0
            
            return PerformanceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb
            )
            
        except Exception as e:
            self.logger.error(f"Failed to take performance snapshot: {str(e)}")
            # Return a default snapshot
            return PerformanceSnapshot(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics from snapshots."""
        if not self._snapshots or not self._initial_snapshot:
            return {}
        
        duration = (self._end_time or time.time()) - (self._start_time or 0)
        
        # Extract metric series
        cpu_values = [s.cpu_percent for s in self._snapshots]
        memory_mb_values = [s.memory_mb for s in self._snapshots]
        memory_percent_values = [s.memory_percent for s in self._snapshots]
        
        # Calculate statistics
        metrics = {
            # Timing
            "duration": duration,
            "start_time": datetime.fromtimestamp(self._start_time).isoformat() if self._start_time else None,
            "end_time": datetime.fromtimestamp(self._end_time).isoformat() if self._end_time else None,
            "sample_count": len(self._snapshots),
            "sampling_interval": self.sampling_interval,
            
            # Response time (same as duration for now)
            "response_time": duration,
            
            # CPU metrics
            "cpu_usage": {
                "average": self._safe_mean(cpu_values),
                "peak": max(cpu_values) if cpu_values else 0,
                "minimum": min(cpu_values) if cpu_values else 0,
                "std_dev": self._calculate_std_dev(cpu_values)
            },
            
            # Memory metrics (absolute)
            "memory_usage_mb": {
                "average": self._safe_mean(memory_mb_values),
                "peak": max(memory_mb_values) if memory_mb_values else 0,
                "minimum": min(memory_mb_values) if memory_mb_values else 0,
                "std_dev": self._calculate_std_dev(memory_mb_values)
            },
            
            # Memory metrics (percentage)
            "memory_usage_percent": {
                "average": self._safe_mean(memory_percent_values),
                "peak": max(memory_percent_values) if memory_percent_values else 0,
                "minimum": min(memory_percent_values) if memory_percent_values else 0,
                "std_dev": self._calculate_std_dev(memory_percent_values)
            },
            
            # Peak memory for easy access
            "peak_memory_mb": max(memory_mb_values) if memory_mb_values else 0,
            
            # I/O metrics (delta from start to end)
            "io_metrics": self._calculate_io_metrics(),
            
            # System info
            "system_info": self._get_system_info()
        }
        
        return metrics
    
    def _calculate_io_metrics(self) -> Dict[str, Any]:
        """Calculate I/O metrics from snapshots."""
        if not self._snapshots or len(self._snapshots) < 2:
            return {}
        
        first_snapshot = self._snapshots[0]
        last_snapshot = self._snapshots[-1]
        
        return {
            "disk_read_mb": last_snapshot.disk_io_read_mb - first_snapshot.disk_io_read_mb,
            "disk_write_mb": last_snapshot.disk_io_write_mb - first_snapshot.disk_io_write_mb,
            "network_sent_mb": last_snapshot.network_sent_mb - first_snapshot.network_sent_mb,
            "network_recv_mb": last_snapshot.network_recv_mb - first_snapshot.network_recv_mb
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "platform": psutil.WINDOWS if hasattr(psutil, 'WINDOWS') else "unknown"
            }
        except Exception as e:
            self.logger.error(f"Failed to get system info: {str(e)}")
            return {}
    
    def _safe_mean(self, values: List[float]) -> float:
        """Safely calculate mean of a list of values."""
        return sum(values) / len(values) if values else 0.0
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = self._safe_mean(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics without stopping monitoring.
        
        Returns:
            Dictionary containing current performance snapshot
        """
        if not self._monitoring:
            return {}
        
        current_snapshot = self._take_snapshot()
        
        return {
            "timestamp": datetime.fromtimestamp(current_snapshot.timestamp).isoformat(),
            "cpu_percent": current_snapshot.cpu_percent,
            "memory_mb": current_snapshot.memory_mb,
            "memory_percent": current_snapshot.memory_percent,
            "monitoring_duration": current_snapshot.timestamp - (self._start_time or 0)
        }
    
    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active."""
        return self._monitoring
    
    def reset(self):
        """Reset the monitor state."""
        if self._monitoring:
            self.stop_monitoring()
        
        self._snapshots.clear()
        self._start_time = None
        self._end_time = None
        self._initial_snapshot = None
        
        self.logger.info("Performance monitor reset")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return self.stop_monitoring()
