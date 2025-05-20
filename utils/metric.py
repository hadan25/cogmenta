"""
Performance metrics for evaluating the cognitive architecture.
Provides functions to measure accuracy, response time, memory usage, and other metrics.
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class PerformanceMetrics:
    def __init__(self):
        """Initialize performance metrics tracking"""
        # Timing metrics
        self.response_times = []
        self.module_times = defaultdict(list)
        
        # Accuracy metrics
        self.predictions = []  # (predicted, actual, confidence)
        
        # Memory usage metrics
        self.memory_snapshots = []
        
        # Resource utilization
        self.cpu_usage = []
        self.memory_usage = []
        
        # Component-specific metrics
        self.component_metrics = defaultdict(lambda: defaultdict(list))
        
        # Meta-reasoning metrics
        self.strategy_usage = defaultdict(int)
        self.strategy_success = defaultdict(list)
        
        # Hallucination metrics
        self.hallucination_scores = []
    
    def record_response_time(self, module_name, duration):
        """
        Record response time for a module
        
        Args:
            module_name: Name of the module
            duration: Time taken in seconds
        """
        self.response_times.append(duration)
        self.module_times[module_name].append(duration)
    
    def start_timer(self):
        """
        Start a timer for measuring execution time
        
        Returns:
            Start time
        """
        return time.time()
    
    def end_timer(self, start_time, module_name=None):
        """
        End a timer and record duration
        
        Args:
            start_time: Start time from start_timer()
            module_name: Optional module name to categorize timing
            
        Returns:
            Duration in seconds
        """
        duration = time.time() - start_time
        
        # Record overall response time
        self.response_times.append(duration)
        
        # Record per-module time if provided
        if module_name:
            self.module_times[module_name].append(duration)
            
        return duration
    
    def record_accuracy(self, predicted, actual, confidence=1.0):
        """
        Record prediction accuracy
        
        Args:
            predicted: Predicted value
            actual: Actual value
            confidence: Confidence in prediction (0-1)
        """
        self.predictions.append((predicted, actual, confidence))
    
    def record_component_metric(self, component, metric_name, value):
        """
        Record component-specific metric
        
        Args:
            component: Component name
            metric_name: Name of the metric
            value: Metric value
        """
        self.component_metrics[component][metric_name].append(value)
    
    def record_strategy_usage(self, strategy_name, success_score):
        """
        Record usage and success of a reasoning strategy
        
        Args:
            strategy_name: Name of the strategy
            success_score: Score indicating success (0-1)
        """
        self.strategy_usage[strategy_name] += 1
        self.strategy_success[strategy_name].append(success_score)
    
    def record_hallucination_score(self, score):
        """
        Record hallucination score
        
        Args:
            score: Hallucination score (higher means more hallucination)
        """
        self.hallucination_scores.append(score)
    
    def calculate_metrics(self):
        """
        Calculate summary metrics from recorded data
        
        Returns:
            Dict of calculated metrics
        """
        metrics = {}
        
        # Response time metrics
        if self.response_times:
            metrics["response_time"] = {
                "mean": np.mean(self.response_times),
                "median": np.median(self.response_times),
                "min": min(self.response_times),
                "max": max(self.response_times),
                "std": np.std(self.response_times)
            }
            
        # Per-module timing
        metrics["module_timing"] = {}
        for module, times in self.module_times.items():
            if times:
                metrics["module_timing"][module] = {
                    "mean": np.mean(times),
                    "median": np.median(times),
                    "min": min(times),
                    "max": max(times),
                    "std": np.std(times),
                    "count": len(times)
                }
                
        # Accuracy metrics
        if self.predictions:
            # Calculate overall accuracy
            correct = sum(1 for p, a, _ in self.predictions if p == a)
            total = len(self.predictions)
            
            metrics["accuracy"] = {
                "overall": correct / total if total > 0 else 0,
                "count": total
            }
            
            # Calculate confidence-weighted accuracy
            weighted_correct = sum(c for p, a, c in self.predictions if p == a)
            total_confidence = sum(c for _, _, c in self.predictions)
            
            metrics["accuracy"]["confidence_weighted"] = weighted_correct / total_confidence if total_confidence > 0 else 0
            
            # Calculate high-confidence accuracy
            high_conf_preds = [(p, a) for p, a, c in self.predictions if c >= 0.8]
            high_conf_correct = sum(1 for p, a in high_conf_preds if p == a)
            
            metrics["accuracy"]["high_confidence"] = high_conf_correct / len(high_conf_preds) if high_conf_preds else 0
            
        # Strategy usage metrics
        if self.strategy_usage:
            metrics["strategies"] = {
                "usage_counts": dict(self.strategy_usage)
            }
            
            # Calculate strategy success rates
            strategy_success_rates = {}
            for strategy, scores in self.strategy_success.items():
                if scores:
                    strategy_success_rates[strategy] = np.mean(scores)
                    
            metrics["strategies"]["success_rates"] = strategy_success_rates
            
        # Hallucination metrics
        if self.hallucination_scores:
            metrics["hallucination"] = {
                "mean_score": np.mean(self.hallucination_scores),
                "max_score": max(self.hallucination_scores),
                "min_score": min(self.hallucination_scores)
            }
            
        # Component-specific metrics
        if self.component_metrics:
            metrics["component_metrics"] = {}
            
            for component, component_data in self.component_metrics.items():
                metrics["component_metrics"][component] = {}
                
                for metric_name, values in component_data.items():
                    if values:
                        metrics["component_metrics"][component][metric_name] = {
                            "mean": np.mean(values),
                            "median": np.median(values),
                            "min": min(values),
                            "max": max(values),
                            "std": np.std(values),
                            "count": len(values)
                        }
                        
        return metrics
    
    def visualize_metrics(self, metrics=None, save_path=None):
        """
        Generate visualizations of metrics
        
        Args:
            metrics: Pre-calculated metrics (if None, calculate)
            save_path: Path to save visualizations
            
        Returns:
            Dict of figure objects
        """
        if metrics is None:
            metrics = self.calculate_metrics()
            
        figures = {}
        
        # Response time distribution
        if self.response_times:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.response_times, bins=20, alpha=0.7, color='blue')
            ax.set_title('Response Time Distribution')
            ax.set_xlabel('Response Time (seconds)')
            ax.set_ylabel('Frequency')
            
            figures["response_time_distribution"] = fig
            
            if save_path:
                fig.savefig(f"{save_path}/response_time_distribution.png")
                
        # Module timing comparison
        if self.module_times:
            modules = list(self.module_times.keys())
            mean_times = [np.mean(self.module_times[m]) for m in modules]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(modules, mean_times, alpha=0.7, color='green')
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom')
                
            ax.set_title('Average Processing Time by Module')
            ax.set_xlabel('Module')
            ax.set_ylabel('Average Time (seconds)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            figures["module_timing"] = fig
            
            if save_path:
                fig.savefig(f"{save_path}/module_timing.png")
                
        # Strategy usage and success
        if self.strategy_usage and self.strategy_success:
            strategies = list(self.strategy_usage.keys())
            usage_counts = [self.strategy_usage[s] for s in strategies]
            success_rates = [np.mean(self.strategy_success[s]) if self.strategy_success[s] else 0 for s in strategies]
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Strategy')
            ax1.set_ylabel('Usage Count', color=color)
            bars1 = ax1.bar(strategies, usage_counts, alpha=0.7, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Add second y-axis for success rates
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Success Rate', color=color)
            ax2.plot(strategies, success_rates, 'o-', color=color, linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add values on top of bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', color='blue')
                
            plt.title('Strategy Usage and Success Rates')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            figures["strategy_metrics"] = fig
            
            if save_path:
                fig.savefig(f"{save_path}/strategy_metrics.png")
                
        # Hallucination score over time
        if self.hallucination_scores:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(self.hallucination_scores)), self.hallucination_scores, 
                   marker='o', linestyle='-', color='purple')
            ax.set_title('Hallucination Score Over Time')
            ax.set_xlabel('Observation Index')
            ax.set_ylabel('Hallucination Score')
            ax.grid(True, alpha=0.3)
            
            figures["hallucination_trend"] = fig
            
            if save_path:
                fig.savefig(f"{save_path}/hallucination_trend.png")
                
        return figures
    
    def save_metrics(self, filename):
        """
        Save metrics to JSON file
        
        Args:
            filename: Output filename
            
        Returns:
            Success flag
        """
        try:
            metrics = self.calculate_metrics()
            
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            return True
        except Exception as e:
            print(f"[Metrics] Error saving metrics: {e}")
            return False
    
    def load_metrics(self, filename):
        """
        Load metrics from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded metrics dict
        """
        try:
            with open(filename, 'r') as f:
                metrics = json.load(f)
                
            return metrics
        except Exception as e:
            print(f"[Metrics] Error loading metrics: {e}")
            return None
    
    def reset(self):
        """Reset all metrics"""
        self.response_times = []
        self.module_times = defaultdict(list)
        self.predictions = []
        self.memory_snapshots = []
        self.cpu_usage = []
        self.memory_usage = []
        self.component_metrics = defaultdict(lambda: defaultdict(list))
        self.strategy_usage = defaultdict(int)
        self.strategy_success = defaultdict(list)
        self.hallucination_scores = []
        
    def get_summary(self):
        """
        Get a text summary of key metrics
        
        Returns:
            String with metrics summary
        """
        metrics = self.calculate_metrics()
        
        summary_lines = ["=== Performance Metrics Summary ==="]
        
        # Response time summary
        if "response_time" in metrics:
            rt = metrics["response_time"]
            summary_lines.append("\nResponse Time:")
            summary_lines.append(f"  Mean: {rt['mean']:.4f} seconds")
            summary_lines.append(f"  Median: {rt['median']:.4f} seconds")
            summary_lines.append(f"  Min/Max: {rt['min']:.4f} / {rt['max']:.4f} seconds")
            
        # Accuracy summary
        if "accuracy" in metrics:
            acc = metrics["accuracy"]
            summary_lines.append("\nAccuracy:")
            summary_lines.append(f"  Overall: {acc['overall']*100:.2f}% ({acc['count']} samples)")
            if "confidence_weighted" in acc:
                summary_lines.append(f"  Confidence-weighted: {acc['confidence_weighted']*100:.2f}%")
            if "high_confidence" in acc:
                summary_lines.append(f"  High-confidence: {acc['high_confidence']*100:.2f}%")
                
        # Strategy summary
        if "strategies" in metrics:
            strat = metrics["strategies"]
            summary_lines.append("\nStrategy Usage:")
            for strategy, count in strat["usage_counts"].items():
                success_rate = strat["success_rates"].get(strategy, 0) * 100
                summary_lines.append(f"  {strategy}: {count} uses, {success_rate:.2f}% success")
                
        # Hallucination summary
        if "hallucination" in metrics:
            hall = metrics["hallucination"]
            summary_lines.append("\nHallucination:")
            summary_lines.append(f"  Mean score: {hall['mean_score']:.4f}")
            summary_lines.append(f"  Min/Max: {hall['min_score']:.4f} / {hall['max_score']:.4f}")
            
        # Module timing summary
        if "module_timing" in metrics:
            summary_lines.append("\nModule Timing (Average seconds):")
            for module, timing in metrics["module_timing"].items():
                summary_lines.append(f"  {module}: {timing['mean']:.4f}")
                
        return "\n".join(summary_lines)