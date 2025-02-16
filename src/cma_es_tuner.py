
import cma
import numpy as np
import joblib
from src.workload_detector import detect_workload
from src.kernel_param_adjuster import apply_kernel_params

THROUGHPUT_MODEL_FILE = "data/workload_throughput_model.pkl"
ENERGY_MODEL_FILE = "data/energy_efficiency_model.pkl"
PERFORMANCE_MODEL_FILE = "data/performance_model.pkl"

PARAMETER_BOUNDS = {
    "vm.swappiness": (0, 100),
    "kernel.sched_min_granularity_ns": (100000, 5000000),
    "kernel.sched_wakeup_granularity_ns": (100000, 5000000),
    "vm.dirty_ratio": (5, 50),
    "vm.dirty_background_ratio": (1, 20),
    "block/read_ahead_kb": (32, 512),
    "block/queue_depth": (16, 128),
}

def objective_function(params):
    throughput_model = joblib.load(THROUGHPUT_MODEL_FILE)
    energy_model = joblib.load(ENERGY_MODEL_FILE)
    performance_model = joblib.load(PERFORMANCE_MODEL_FILE)
    
    sample_input = np.array(params).reshape(1, -1)
    predicted_throughput = throughput_model.predict(sample_input)[0]
    predicted_efficiency = energy_model.predict(sample_input)[0]
    predicted_performance = performance_model.predict(sample_input)[0]
    
    penalty_factor = 2.0
    reward_factor = 1.0
    
    if predicted_performance < predicted_throughput:
        penalty = penalty_factor * abs(predicted_throughput - predicted_performance)
    else:
        penalty = 0
    
    score = (reward_factor * predicted_efficiency) - penalty
    return -score

def optimize_parameters(n_trials=30):
    initial_guess = [np.mean(bound) for bound in PARAMETER_BOUNDS.values()]
    options = {
        "bounds": [list(b[0] for b in PARAMETER_BOUNDS.values()), 
                   list(b[1] for b in PARAMETER_BOUNDS.values())],
        "popsize": 10,
        "maxiter": n_trials,
    }
    
    es = cma.CMAEvolutionStrategy(initial_guess, 0.5, options)
    es.optimize(objective_function)
    
    best_params = es.result.xbest
    print(f"Best parameters found: {best_params}")
    
    tuned_params = dict(zip(PARAMETER_BOUNDS.keys(), best_params))
    print(f"Tuned Kernel Parameters: {tuned_params}")
    
    apply_kernel_params()
    return tuned_params

if __name__ == "__main__":
    print("Optimizing parameters using CMA-ES...")
    best_config = optimize_parameters(n_trials=30)
    print("Optimization complete.")
