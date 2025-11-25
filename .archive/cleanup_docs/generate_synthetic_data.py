#!/usr/bin/env python3
"""
Generate Synthetic RCAEval-like Data for Testing

Creates realistic synthetic failure cases with:
- Metrics time series (CPU, memory, latency, etc.)
- Log entries with templates
- Trace graphs (service dependencies)
- Ground truth labels

Usage:
    python scripts/generate_synthetic_data.py --n_cases 50 --output data/synthetic
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Tuple
import networkx as nx


class SyntheticDataGenerator:
    """Generate synthetic multimodal failure data"""

    def __init__(self, random_seed: int = 42):
        """Initialize generator with random seed"""
        np.random.seed(random_seed)
        self.random_seed = random_seed

        # Fault types
        self.fault_types = ['CPU', 'MEM', 'DISK', 'DELAY', 'LOSS', 'SOCKET']

        # Service names (microservice architecture)
        self.services = [
            'frontend', 'api-gateway', 'auth-service', 'user-service',
            'order-service', 'payment-service', 'inventory-service',
            'shipping-service', 'notification-service', 'database',
            'cache-service', 'search-service', 'recommendation-service'
        ]

        # Metric types per service
        self.metric_types = [
            'cpu_usage', 'memory_usage', 'disk_usage', 'network_in',
            'network_out', 'latency_p50', 'latency_p90', 'latency_p99',
            'request_rate', 'error_rate', 'queue_size'
        ]

        # Log templates
        self.log_templates = [
            "INFO: Request processed successfully",
            "INFO: Database query executed in <NUM>ms",
            "WARN: High memory usage detected: <NUM>%",
            "ERROR: Connection timeout to <SERVICE>",
            "ERROR: Database connection failed",
            "INFO: Cache hit rate: <NUM>%",
            "WARN: Request queue size: <NUM>",
            "ERROR: Service <SERVICE> returned 500",
            "INFO: Authentication successful for user <ID>",
            "ERROR: Payment processing failed for order <ID>"
        ]

    def generate_metrics(
        self,
        n_timesteps: int = 60,
        fault_service: str = None,
        fault_type: str = None,
        fault_start: int = 40
    ) -> pd.DataFrame:
        """
        Generate synthetic metrics time series

        Args:
            n_timesteps: Number of timesteps (default 60 = 5-min intervals for 5 hours)
            fault_service: Service where fault occurs
            fault_type: Type of fault (CPU, MEM, etc.)
            fault_start: Timestep where fault begins

        Returns:
            DataFrame with columns: [timestamp, service_metric, ...]
        """
        data = {}

        # Generate timestamp index
        timestamps = pd.date_range('2024-01-01', periods=n_timesteps, freq='5min')
        data['timestamp'] = timestamps

        for service in self.services:
            for metric in self.metric_types:
                col_name = f"{service}_{metric}"

                # Generate base time series with some autocorrelation
                base_value = np.random.uniform(20, 60)
                noise = np.random.randn(n_timesteps) * 5

                # Add trend
                trend = np.linspace(0, np.random.uniform(-10, 10), n_timesteps)

                # Add seasonality (daily pattern)
                seasonal = 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 12)

                values = base_value + trend + seasonal + noise

                # Inject fault if this is the fault service
                if service == fault_service and fault_type:
                    if fault_type == 'CPU' and 'cpu' in metric:
                        # CPU spike
                        values[fault_start:] += np.linspace(0, 40, n_timesteps - fault_start)
                    elif fault_type == 'MEM' and 'memory' in metric:
                        # Memory leak
                        values[fault_start:] += np.linspace(0, 35, n_timesteps - fault_start)
                    elif fault_type == 'DELAY' and 'latency' in metric:
                        # Latency spike
                        values[fault_start:] *= 3
                    elif fault_type == 'DISK' and 'disk' in metric:
                        # Disk usage increase
                        values[fault_start:] += 30

                # Clip to realistic ranges
                if 'usage' in metric or 'rate' in metric:
                    values = np.clip(values, 0, 100)
                elif 'latency' in metric:
                    values = np.clip(values, 1, 5000)
                else:
                    values = np.clip(values, 0, 1000)

                data[col_name] = values

        return pd.DataFrame(data)

    def generate_logs(
        self,
        n_entries: int = 1000,
        fault_service: str = None,
        fault_type: str = None,
        fault_start_time: pd.Timestamp = None
    ) -> pd.DataFrame:
        """
        Generate synthetic log entries

        Args:
            n_entries: Number of log entries
            fault_service: Service where fault occurs
            fault_type: Type of fault
            fault_start_time: When fault begins

        Returns:
            DataFrame with columns: [timestamp, service, level, message]
        """
        logs = []

        base_time = pd.Timestamp('2024-01-01')
        time_range = pd.Timedelta(hours=5)

        for i in range(n_entries):
            # Random timestamp
            timestamp = base_time + pd.Timedelta(seconds=np.random.uniform(0, time_range.total_seconds()))

            # Random service
            service = np.random.choice(self.services)

            # Determine log level based on fault
            if fault_start_time and timestamp >= fault_start_time and service == fault_service:
                # More errors during fault
                level = np.random.choice(['ERROR', 'WARN', 'INFO'], p=[0.4, 0.3, 0.3])
            else:
                # Normal operation
                level = np.random.choice(['ERROR', 'WARN', 'INFO'], p=[0.05, 0.15, 0.80])

            # Random template
            template = np.random.choice(self.log_templates)

            # Fill in template
            message = template
            message = message.replace('<NUM>', str(np.random.randint(1, 100)))
            message = message.replace('<SERVICE>', np.random.choice(self.services))
            message = message.replace('<ID>', str(np.random.randint(1000, 9999)))

            logs.append({
                'timestamp': timestamp,
                'service': service,
                'level': level,
                'message': message
            })

        df = pd.DataFrame(logs).sort_values('timestamp').reset_index(drop=True)
        return df

    def generate_service_graph(self) -> nx.DiGraph:
        """
        Generate synthetic service dependency graph

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add all services as nodes
        for service in self.services:
            G.add_node(service)

        # Create realistic dependencies
        # Frontend calls API gateway
        G.add_edge('frontend', 'api-gateway')

        # API gateway calls various services
        for service in ['auth-service', 'user-service', 'order-service', 'search-service']:
            G.add_edge('api-gateway', service)

        # Services call each other
        G.add_edge('order-service', 'payment-service')
        G.add_edge('order-service', 'inventory-service')
        G.add_edge('order-service', 'shipping-service')
        G.add_edge('payment-service', 'notification-service')
        G.add_edge('user-service', 'auth-service')
        G.add_edge('search-service', 'recommendation-service')

        # Many services call database and cache
        for service in ['user-service', 'order-service', 'inventory-service', 'auth-service']:
            G.add_edge(service, 'database')
            G.add_edge(service, 'cache-service')

        return G

    def generate_traces(
        self,
        n_traces: int = 5000,
        service_graph: nx.DiGraph = None
    ) -> pd.DataFrame:
        """
        Generate synthetic distributed traces

        Args:
            n_traces: Number of trace spans
            service_graph: Service dependency graph

        Returns:
            DataFrame with columns: [span_id, parent_span, service, latency, ...]
        """
        if service_graph is None:
            service_graph = self.generate_service_graph()

        traces = []
        trace_id = 0

        # Generate traces by doing random walks through service graph
        for _ in range(n_traces // 10):  # Each trace has ~10 spans
            trace_id += 1

            # Start from frontend
            current_service = 'frontend'
            parent_span = None
            span_counter = 0

            # Walk through the graph
            for depth in range(5):
                span_id = f"trace_{trace_id}_span_{span_counter}"

                # Generate span data
                latency = np.random.exponential(50)  # Exponential distribution for latency

                traces.append({
                    'trace_id': trace_id,
                    'span_id': span_id,
                    'parent_span': parent_span,
                    'service': current_service,
                    'latency_ms': latency,
                    'status': 'ok' if np.random.random() > 0.05 else 'error'
                })

                # Move to next service
                successors = list(service_graph.successors(current_service))
                if successors:
                    current_service = np.random.choice(successors)
                    parent_span = span_id
                    span_counter += 1
                else:
                    break

        return pd.DataFrame(traces)

    def generate_failure_case(
        self,
        case_id: int,
        system: str = 'SyntheticSystem'
    ) -> Dict:
        """
        Generate one complete failure case

        Args:
            case_id: Unique case identifier
            system: System name

        Returns:
            Dictionary with all modalities and metadata
        """
        # Random fault configuration
        fault_service = np.random.choice(self.services)
        fault_type = np.random.choice(self.fault_types)
        fault_start = 40  # Fault starts at timestep 40

        # Generate all modalities
        metrics = self.generate_metrics(
            fault_service=fault_service,
            fault_type=fault_type,
            fault_start=fault_start
        )

        fault_start_time = metrics['timestamp'].iloc[fault_start]

        logs = self.generate_logs(
            fault_service=fault_service,
            fault_type=fault_type,
            fault_start_time=fault_start_time
        )

        service_graph = self.generate_service_graph()
        traces = self.generate_traces(service_graph=service_graph)

        return {
            'case_id': f"{system}_case_{case_id:03d}",
            'system': system,
            'fault_type': fault_type,
            'root_cause_service': fault_service,
            'root_cause_indicator': f"{fault_service}_{fault_type.lower()}_usage",
            'metrics': metrics,
            'logs': logs,
            'traces': traces,
            'service_graph': service_graph
        }

    def save_case(self, case: Dict, output_dir: Path):
        """Save a failure case to disk"""
        case_dir = output_dir / case['case_id']
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        case['metrics'].to_csv(case_dir / 'metrics.csv', index=False)

        # Save logs
        case['logs'].to_csv(case_dir / 'logs.csv', index=False)

        # Save traces
        case['traces'].to_csv(case_dir / 'traces.csv', index=False)

        # Save metadata
        metadata = {
            'case_id': case['case_id'],
            'system': case['system'],
            'fault_type': case['fault_type'],
            'root_cause_service': case['root_cause_service'],
            'root_cause_indicator': case['root_cause_indicator']
        }

        with open(case_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved: {case['case_id']}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic RCAEval-like data')
    parser.add_argument('--n_cases', type=int, default=50, help='Number of failure cases')
    parser.add_argument('--output', type=str, default='data/synthetic', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print("=" * 80)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 80)
    print(f"\nGenerating {args.n_cases} failure cases...")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = SyntheticDataGenerator(random_seed=args.seed)

    # Generate cases
    for i in range(args.n_cases):
        case = generator.generate_failure_case(case_id=i, system='Synthetic')
        generator.save_case(case, output_dir)

    print("\n" + "=" * 80)
    print("✅ GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated {args.n_cases} cases in: {output_dir}")
    print("\nTo use this data:")
    print(f"  1. Update data loader to point to: {output_dir}")
    print(f"  2. Run tests: python scripts/test_encoders.py --n_cases 10")
    print(f"  3. Run experiments: python scripts/run_experiments.py")


if __name__ == '__main__':
    main()
