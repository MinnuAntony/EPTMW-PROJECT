
import src.workload_simulator as ws
import src.energy_collector as ec
import src.model_trainer as mt
import src.kernel_param_adjuster as ka
import src.workload_detector as wd
import src.cma_es_tuner as ce

def run_eptmw():
    print("\n===== Step 1: Generate Workloads =====")
    ws.generate_workloads('data/workloads.csv')

    print("\n===== Step 2: Collect Simulated Power Metrics =====")
    ec.collect_metrics('data/workloads.csv', 'data/metrics.csv')

    print("\n===== Step 3: Train Models =====")
    mt.train_models('data/metrics.csv', 'data/eptmw_model')

    print("\n===== Step 4: Train Workload Detection Models =====")
    wd.train_workload_models('data/performance_counters.csv')

    print("\n===== Step 5: Apply Linux Kernel Parameter Tuning =====")
    ka.apply_kernel_params()

    print("\n===== Step 6: Detect Workload Type and Compute Weight Vector =====")
    sample_counters = [0.85, 0.12, 0.07, 150, 250, 60]
    wd.detect_workload(sample_counters)

    print("\n===== Step 7: Optimize Parameters Using CMA-ES =====")
    ce.optimize_parameters(n_trials=30)

    print("\n===== ✅ EPTMW Workflow Completed Successfully! =====")

if __name__ == "__main__":
    run_eptmw()
