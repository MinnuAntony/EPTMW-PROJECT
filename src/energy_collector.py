# energy_collector.py content

import pandas as pd
import random
import time

def collect_metrics(workload_file, output_file):
    """Simulate power metrics collection based on workloads."""
    workloads = pd.read_csv(workload_file)
    results = []
    for _, row in workloads.iterrows():
        time.sleep(0.5)  # Simulate delay
        power = (row['cpu_load'] * 100) + (row['memory_load'] * 50) + (row['disk_load'] * 30) + random.uniform(0, 20)
        results.append({
            'cpu_load': row['cpu_load'],
            'memory_load': row['memory_load'],
            'disk_load': row['disk_load'],
            'power': power
        })
    pd.DataFrame(results).to_csv(output_file, index=False)
    print("Simulated power metrics collected.")

if __name__ == "__main__":
    collect_metrics('data/workloads.csv', 'data/metrics.csv')

