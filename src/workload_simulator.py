# workload_simulator.py content
import numpy as np
import pandas as pd
from pyDOE import lhs

def generate_workloads(output_file, num_samples=50):
    """Generate mixed workloads using LHS (CPU, memory, disk)."""
    samples = lhs(3, samples=num_samples)
    workloads = pd.DataFrame(samples, columns=['cpu_load', 'memory_load', 'disk_load'])
    workloads.to_csv(output_file, index=False)
    print("Mixed workloads generated using LHS.")

if __name__ == "__main__":
    generate_workloads('data/workloads.csv')


""" import numpy as np
import pandas as pd
from pyDOE import lhs

def generate_realistic_workloads(output_file, num_samples=50):
    # Generate mixed workloads using LHS with realistic constraints.
    
    # Define workload categories
    workload_types = ["cpu_intensive", "memory_intensive", "disk_intensive", "mixed"]
    
    # Scaling values for realistic loads
    max_cpu = 100       # CPU usage in %
    max_memory = 16     # Memory usage in GB (assuming a 16GB system)
    max_disk = 500      # Disk I/O in MB/s
    
    # Generate samples using LHS
    samples = lhs(3, samples=num_samples)
    
    workloads = []
    
    for i in range(num_samples):
        workload_type = np.random.choice(workload_types)  # Assign a workload type
        
        # Assign loads based on workload type
        if workload_type == "cpu_intensive":
            cpu_load = samples[i, 0] * max_cpu * 0.8 + 20   # High CPU (20%-100%)
            memory_load = samples[i, 1] * max_memory * 0.5  # Moderate Memory
            disk_load = samples[i, 2] * max_disk * 0.2      # Low Disk
        
        elif workload_type == "memory_intensive":
            cpu_load = samples[i, 0] * max_cpu * 0.5        # Moderate CPU
            memory_load = samples[i, 1] * max_memory * 0.8 + 2  # High Memory (2GB-16GB)
            disk_load = samples[i, 2] * max_disk * 0.3      # Moderate Disk
        
        elif workload_type == "disk_intensive":
            cpu_load = samples[i, 0] * max_cpu * 0.3        # Low CPU
            memory_load = samples[i, 1] * max_memory * 0.3  # Low Memory
            disk_load = samples[i, 2] * max_disk * 0.8 + 50  # High Disk (50MB/s-500MB/s)
        
        else:  # Mixed workloads
            cpu_load = samples[i, 0] * max_cpu  # Any CPU load
            memory_load = samples[i, 1] * max_memory  # Any Memory load
            disk_load = samples[i, 2] * max_disk  # Any Disk load
        
        workloads.append([workload_type, cpu_load, memory_load, disk_load])
    
    # Convert to DataFrame and save
    df = pd.DataFrame(workloads, columns=['workload_type', 'cpu_load', 'memory_load', 'disk_load'])
    df.to_csv(output_file, index=False)
    
    print(f"Realistic mixed workloads generated and saved to {output_file}")

if __name__ == "__main__":
    generate_realistic_workloads('data/realistic_workloads.csv') 
    
"""

