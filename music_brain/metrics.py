"""
Metrics collection for monitoring and observability.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Simple in-memory metrics collector.
    
    In production, this should be replaced with Prometheus, StatsD, or similar.
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()
        self._max_points_per_metric = 1000
    
    def increment(self, metric_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(metric_name, tags)
            self._counters[key] += value
    
    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                tags=tags or {}
            )
            key = self._make_key(metric_name, tags)
            self._metrics[key].append(point)
            
            # Keep only recent points
            if len(self._metrics[key]) > self._max_points_per_metric:
                self._metrics[key] = self._metrics[key][-self._max_points_per_metric:]
    
    def histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        with self._lock:
            key = self._make_key(metric_name, tags)
            self._histograms[key].append(value)
            
            # Keep only recent values
            if len(self._histograms[key]) > self._max_points_per_metric:
                self._histograms[key] = self._histograms[key][-self._max_points_per_metric:]
    
    def timer(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return Timer(self, metric_name, tags)
    
    def _make_key(self, metric_name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a key from metric name and tags."""
        if not tags:
            return metric_name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric_name}[{tag_str}]"
    
    def get_counter(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value."""
        with self._lock:
            key = self._make_key(metric_name, tags)
            return self._counters.get(key, 0)
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics."""
        with self._lock:
            summary = {
                "counters": dict(self._counters),
                "gauges": {},
                "histograms": {},
            }
            
            # Calculate gauge averages
            for key, points in self._metrics.items():
                if points:
                    values = [p.value for p in points]
                    summary["gauges"][key] = {
                        "current": values[-1],
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }
            
            # Calculate histogram statistics
            for key, values in self._histograms.items():
                if values:
                    summary["histograms"][key] = {
                        "count": len(values),
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99),
                    }
            
            return summary
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def reset(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._histograms.clear()


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, tags: Optional[Dict[str, str]]):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.histogram(self.metric_name, duration, self.tags)
        return False


# Global metrics collector instance
metrics = MetricsCollector()
