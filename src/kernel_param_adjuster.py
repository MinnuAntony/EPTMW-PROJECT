
import subprocess

KERNEL_PARAMS = {
    "cpu": {
        "vm.swappiness": 10,
        "kernel.sched_min_granularity_ns": 1000000,
        "kernel.sched_wakeup_granularity_ns": 1500000,
        "kernel.sched_migration_cost_ns": 500000,
    },
    "memory": {
        "vm.min_free_kbytes": 65536,
        "vm.dirty_background_ratio": 5,
        "vm.dirty_ratio": 10,
        "vm.dirty_writeback_centisecs": 100,
        "vm.dirty_expire_centisecs": 200,
        "vm.zone_reclaim_mode": 0,
        "kernel.randomize_va_space": 2,
    },
    "disk": {
        "vm.dirty_expire_centisecs": 200,
        "vm.dirty_writeback_centisecs": 100,
        "block/scheduler": "mq-deadline",
        "block/max_sectors_kb": 512,
        "block/queue_depth": 32,
        "block/nr_requests": 128,
        "block/read_ahead_kb": 128,
    }
}

def set_sysctl(param, value):
    try:
        subprocess.run(["sudo", "sysctl", f"{param}={value}"], check=True)
        print(f"Set {param} to {value}")
    except Exception as e:
        print(f"Failed to set {param}: {e}")

def set_block_param(param, value):
    try:
        with open(f"/sys/block/sda/queue/{param}", "w") as f:
            f.write(str(value))
        print(f"Set /sys/block/sda/queue/{param} to {value}")
    except Exception as e:
        print(f"Failed to set /sys/block/sda/queue/{param}: {e}")

def apply_kernel_params():
    for category, params in KERNEL_PARAMS.items():
        for param, value in params.items():
            if "block/" not in param:
                set_sysctl(param, value)

    for param, value in KERNEL_PARAMS["disk"].items():
        if param.startswith("block/"):
            set_block_param(param.replace("block/", ""), value)

if __name__ == "__main__":
    print("Applying Linux kernel parameters for EPTMW...")
    apply_kernel_params()
    print("Kernel parameters applied successfully.")
