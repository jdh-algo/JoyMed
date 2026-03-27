#!/usr/bin/env python3
"""
Comprehensive evaluation pipeline for multi-modal models.
This pipeline handles both inference and evaluation phases for local and remote models.
Supports background daemon execution with status tracking and management.
"""

import os
import sys
import yaml
import time
import argparse
import logging
import subprocess
import json
import signal
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import shutil
from datetime import datetime
from enum import Enum
import pwd
import importlib.util
import requests
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_service.vllm_manager import VLLMServiceManager
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

# Import dataset configurations
from data_configs.data_configs import DATASET_CONFIGS


class PipelineState(Enum):
    """Pipeline execution states."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class EvaluationPipeline:
    """Main evaluation pipeline class with daemon support."""

    def __init__(self, config_path: str, state_file: str = "pipeline_state.json", phase: str = "all"):
        """Initialize the pipeline with configuration."""
        self.config_path = config_path
        self.state_file = state_file
        self.phase = phase  # Phase to run: 'infer', 'eval', or 'all'
        self.config = self._load_config()
        self.vllm_manager = VLLMServiceManager()
        self.logger = None
        self.current_log_file = None  # Initialize log file attribute
        self._setup_directories()
        self.child_processes = []  # Track child processes
        self.state_lock = threading.Lock()  # Lock for thread-safe state updates
        self._cleanup_in_progress = False  # Flag to prevent recursive cleanup
        self._shutdown_requested = False  # Flag to prevent multiple signal handling

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        # Handle relative paths
        if not os.path.isabs(self.config_path):
            # First try relative to current directory
            if os.path.exists(self.config_path):
                config_path = self.config_path
            else:
                # Try relative to script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(script_dir, self.config_path)
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found at {self.config_path} or {config_path}")
        else:
            config_path = self.config_path

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_directories(self):
        """Setup necessary directories."""
        # Create log directory
        self.log_dir = self.config["pipeline"].get("log_dir", "./pipeline_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Create vLLM logs directory
        vllm_log_dir = "./vllm_logs"
        os.makedirs(vllm_log_dir, exist_ok=True)

        # Create work directory
        os.makedirs(self.config["work_dir"], exist_ok=True)

        # Create PID directory
        self.pid_dir = "./pipeline_pids"
        os.makedirs(self.pid_dir, exist_ok=True)

    def _setup_logging(self, daemon: bool = False):
        """Setup logging configuration for foreground mode."""
        if daemon:
            # Daemon mode now handles its own logging setup
            return

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # For normal mode, log to both file and console
        log_file = os.path.join(self.log_dir, f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
        log_level = getattr(logging, self.config["pipeline"]["log_level"])

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],  # Explicitly use stdout
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline initialized with config: {self.config_path}")

        # Save log file path in state
        self.current_log_file = log_file

    def _save_state(self, state: Dict[str, Any]):
        """Save pipeline state to file (thread-safe)."""
        with self.state_lock:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

    def _load_state(self) -> Dict[str, Any]:
        """Load pipeline state from file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "status": PipelineState.IDLE.value,
            "pid": None,
            "pgid": None,  # Process group ID
            "start_time": None,
            "end_time": None,
            "current_model": None,
            "current_phase": None,
            "phase": "all",  # Which phase(s) to run: 'infer', 'eval', or 'all'
            "completed_models": [],
            "failed_models": [],
            "log_file": None,
            "total_models": 0,
            "completed_count": 0,
            "child_pids": [],  # Track child processes
            "error": None,  # Track error message
        }

    def _save_pid(self, pid: int, pgid: int = None):
        """Save process PID and PGID for management."""
        pid_file = os.path.join(self.pid_dir, "pipeline.pid")
        with open(pid_file, "w") as f:
            f.write(f"{pid}\n")
            if pgid:
                f.write(f"{pgid}\n")

    def _get_pid_info(self) -> Tuple[Optional[int], Optional[int]]:
        """Get saved PID and PGID if exists."""
        pid_file = os.path.join(self.pid_dir, "pipeline.pid")
        if os.path.exists(pid_file):
            with open(pid_file, "r") as f:
                lines = f.readlines()
                try:
                    pid = int(lines[0].strip())
                    pgid = int(lines[1].strip()) if len(lines) > 1 else None
                    return pid, pgid
                except (ValueError, IndexError) as e:
                    if self.logger:
                        self.logger.error(f"Error parsing PID file: {str(e)}\n{traceback.format_exc()}")
                    return None, None
        return None, None

    def _remove_pid(self):
        """Remove PID file."""
        pid_file = os.path.join(self.pid_dir, "pipeline.pid")
        if os.path.exists(pid_file):
            os.remove(pid_file)

    def _track_child_process(self, pid: int):
        """Track a child process PID."""
        state = self._load_state()
        if "child_pids" not in state:
            state["child_pids"] = []
        if pid not in state["child_pids"]:
            state["child_pids"].append(pid)
        self._save_state(state)

    def _get_all_datasets(self) -> List[str]:
        """Get all datasets from configuration."""
        return self.config["datasets"]

    def _get_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all models categorized by type."""
        return {
            "local": self.config["models"].get("local_models", []),
            "remote": self.config["models"].get("remote_models", []),
        }

    def _prepare_model_config(self, model: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Prepare model configuration for evaluation."""
        eval_config = model["eval_config"].copy()

        # Determine if this is a remote model
        is_remote = model.get("type") == "remote"

        # Set nproc based on model type
        if is_remote:
            # Use max_remote_concurrency for remote models
            nproc = self.config["evaluation"].get("max_remote_concurrency", 20)
        else:
            # For local models: multiply nproc by number of instances
            base_nproc = self.config["evaluation"]["nproc"]
            if "vllm_config" not in model:
                raise ValueError(f"Local model {model['name']} is missing vllm_config")
            instances = model["vllm_config"]["instances"]
            nproc = base_nproc * instances
            self.logger.info(
                f"Local model {model['name']}: {base_nproc} nproc × {instances} instances = {nproc} total workers"
            )

        # Create configuration structure similar to yaml configs
        config = {
            "work_dir": self.config["work_dir"],
            "eval_backend": self.config["eval_backend"],
            "eval_config": {
                "model": [eval_config],
                "data": self._get_all_datasets(),
                "mode": mode,
                "limit": self.config["evaluation"].get("limit"),
                "reuse": self.config["evaluation"]["reuse"],
                "nproc": nproc,
            },
        }

        # Add judge model configuration if in evaluation mode
        if mode == "all":
            judge_config = self._get_judge_config()
            config["eval_config"].update(judge_config)

        # Set output directory with consistent transformation: "/" -> "_" and "." -> "_"
        model_name = model["name"].replace("/", "_").replace(".", "_")
        config["use_cache"] = os.path.join(self.config["work_dir"], model_name)

        return config

    def _get_judge_config(self) -> Dict[str, Any]:
        """Get judge model configuration."""
        # Detect which judge model type is configured (local or remote)
        judge_models = self.config.get("judge_models", {})

        if "local" in judge_models:
            judge_type = "local"
        elif "remote" in judge_models:
            judge_type = "remote"
        else:
            raise ValueError("No valid judge model configuration found. Expected 'local' or 'remote' in judge_models")

        judge_config = judge_models[judge_type]["eval_config"]
        return judge_config

    def _save_temp_config(self, config: Dict[str, Any]) -> str:
        """Save configuration to a temporary file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f, default_flow_style=False)
            return f.name

    def _wait_for_vllm_services(self, proxy_port: int, max_wait: int = 1800, check_interval: int = 30) -> bool:
        """Wait for vLLM services to be ready by checking their status.

        Args:
            proxy_port: Port number of the proxy service
            max_wait: Maximum time to wait in seconds (default: 30 minutes)
            check_interval: Interval between checks in seconds

        Returns:
            True if services are ready, False if timeout
        """
        start_time = time.time()
        proxy_ready = False
        all_instances_ready = False

        self.logger.info(f"Waiting for vLLM services to be ready (max {max_wait}s)...")

        while time.time() - start_time < max_wait:
            elapsed = time.time() - start_time

            # Check proxy health
            if not proxy_ready:
                try:
                    response = requests.get(f"http://localhost:{proxy_port}/health", timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        if health_data.get("status") == "healthy":
                            proxy_ready = True
                            self.logger.info(f"✓ Proxy is ready on port {proxy_port}")
                except Exception as e:
                    if elapsed > 60:  # Only log after first minute
                        self.logger.debug(f"  Waiting for proxy ({elapsed:.0f}s elapsed)...")
                    # Connection errors are expected while service is starting up
                    pass

            # Check instance status through proxy stats
            if proxy_ready and not all_instances_ready:
                try:
                    response = requests.get(f"http://localhost:{proxy_port}/stats", timeout=5)
                    if response.status_code == 200:
                        stats = response.json()
                        total_instances = stats.get("total_instances", 0)
                        healthy_instances = stats.get("healthy_instances", 0)

                        if total_instances > 0 and healthy_instances == total_instances:
                            all_instances_ready = True
                            self.logger.info(f"✓ All {total_instances} instances are ready")
                        else:
                            self.logger.info(f"  Instances ready: {healthy_instances}/{total_instances}")
                except Exception as e:
                    self.logger.debug(f"  Waiting for instances to be ready ({elapsed:.0f}s elapsed)...")
                    pass

            # Check if everything is ready
            if proxy_ready and all_instances_ready:
                self.logger.info(f"✅ All vLLM services are ready (took {elapsed:.1f}s)")
                return True

            # Wait before next check
            time.sleep(check_interval)

            # Provide periodic status updates
            if elapsed % 60 < check_interval:  # Log every minute
                self.logger.info(f"Still waiting for services to be ready ({elapsed:.0f}s elapsed)...")
                self.logger.info(f"  Proxy ready: {'✓' if proxy_ready else '⨯'}")
                self.logger.info(f"  Instances ready: {'✓' if all_instances_ready else '⨯'}")

        # Timeout reached
        self.logger.error(f"❌ Timeout waiting for vLLM services after {max_wait}s")
        self.logger.error("  Final status:")
        self.logger.error(f"    Proxy ready: {'✓' if proxy_ready else '⨯'}")
        self.logger.error(f"    Instances ready: {'✓' if all_instances_ready else '⨯'}")
        return False

    def _start_local_model(self, model: Dict[str, Any]) -> bool:
        """Start a local model using vLLM manager with smart retry for failed instances."""
        self.logger.info(f"Starting local model: {model['name']}")
        vllm_config = model["vllm_config"].copy()  # Make a copy to avoid modifying original

        # Initial attempt with regular config
        success = self._start_instances_with_smart_retry(model, vllm_config)

        if success:
            # Start proxy after all instances are ready
            proxy_success = self.vllm_manager.start_proxy(
                port=vllm_config["proxy_port"], strategy=vllm_config["strategy"]
            )

            if proxy_success:
                # Wait for all services to be ready
                services_ready = self._wait_for_vllm_services(proxy_port=vllm_config["proxy_port"])

                if services_ready:
                    # Log final status
                    try:
                        response = requests.get(f"http://localhost:{vllm_config['proxy_port']}/stats", timeout=2)
                        if response.status_code == 200:
                            stats = response.json()
                            self.logger.info(
                                f"Proxy stats - Total instances: {stats.get('total_instances', 0)}, "
                                f"Healthy: {stats.get('healthy_instances', 0)}"
                            )
                    except:
                        pass

                    self.logger.info(f"Successfully started {model['name']}")
                    return True
                else:
                    self.logger.error(f"Services did not become ready in time for {model['name']}")
                    # Clean up on failure
                    try:
                        self.vllm_manager.stop_proxy()
                        self.vllm_manager.stop_instances()
                    except Exception as cleanup_e:
                        self.logger.warning(f"Warning during cleanup: {str(cleanup_e)}")
                    return False
            else:
                self.logger.error(f"Failed to start proxy for {model['name']}")
                # Clean up on failure
                try:
                    self.vllm_manager.stop_instances()
                except Exception as cleanup_e:
                    self.logger.warning(f"Warning during cleanup: {str(cleanup_e)}")
                return False
        else:
            self.logger.error(f"Failed to start instances for {model['name']}")
            return False

    def _start_instances_with_smart_retry(self, model: Dict[str, Any], vllm_config: Dict[str, Any]) -> bool:
        """Start instances with smart retry logic that only retries on compilation errors."""
        model_name = model["name"]
        max_retries = 2

        # Clean up any existing services first
        try:
            self.vllm_manager.stop_proxy()
            self.vllm_manager.stop_instances()
        except Exception as e:
            self.logger.warning(f"Warning during cleanup before starting {model_name}: {str(e)}")

        # Additional cleanup for all configured ports to handle config mismatches
        # This ensures we clean up ALL instances from all models in config
        instance_ports, proxy_ports = self._get_all_configured_ports()
        all_ports = instance_ports + proxy_ports

        if all_ports:
            try:
                self.logger.info(f"Performing comprehensive port cleanup for ports: {all_ports}")
                self.vllm_manager.cleanup_processes(all_ports)
            except Exception as e:
                self.logger.warning(f"Warning during comprehensive port cleanup: {str(e)}")

        # Wait a moment for ports to be released
        time.sleep(3)

        # Also ensure GPU memory is cleaned up
        self._cleanup_gpu_memory()

        # Track which instances need retry
        instances_to_start = list(range(vllm_config["instances"]))
        compilation_config = vllm_config.get("compilation_config", None)
        base_port = vllm_config["base_port"]

        for attempt in range(max_retries):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {model_name}")
                self.logger.info(f"Retrying instances: {[base_port + i for i in instances_to_start]}")
                compilation_config = 0  # Disable compilation on retry
                time.sleep(10)

            try:
                # Start instances (either all on first attempt or only failed ones on retry)
                if attempt == 0:
                    # First attempt: start all instances
                    success = self.vllm_manager.start_instances(
                        model=vllm_config["model"],
                        gpus=vllm_config["gpus"],
                        instances=vllm_config["instances"],
                        base_port=vllm_config["base_port"],
                        delay=vllm_config["delay"],
                        compilation_config=compilation_config,
                        is_retry=False,
                        additional_args=vllm_config.get("additional_args"),
                    )
                else:
                    # Retry: start only failed instances with custom GPU assignment
                    gpu_groups = self._calculate_gpu_groups_for_instances(vllm_config, instances_to_start)
                    custom_gpu_assignment = [",".join(map(str, group)) for group in gpu_groups]

                    # Start only the failed instances
                    success = self._start_specific_instances(vllm_config, instances_to_start, compilation_config)

                if success:
                    self.logger.info(f"All instances started successfully for {model_name}")
                    return True
                else:
                    # Check which instances failed and if they have compilation errors
                    if attempt == 0:
                        failed_instances = self._check_failed_instances_for_compilation_errors(vllm_config)
                        if failed_instances:
                            self.logger.warning(
                                f"Compilation errors detected for {model_name} on instances: {failed_instances}"
                            )
                            # Stop only the failed instances
                            self._stop_specific_instances(failed_instances, base_port)
                            instances_to_start = failed_instances
                            continue  # Retry with compilation disabled
                        else:
                            self.logger.error(f"Failed to start instances for {model_name} (attempt {attempt + 1})")
                            self._log_startup_failure_reasons(vllm_config)
                    else:
                        self.logger.error(f"Failed to restart instances for {model_name} (attempt {attempt + 1})")

            except Exception as e:
                self.logger.error(f"Error starting {model_name} (attempt {attempt + 1}): {str(e)}")
                if attempt == 0:
                    self.logger.error(f"Full traceback: {traceback.format_exc()}")

            # Only retry if it's the first attempt and we detected compilation errors
            if attempt < max_retries - 1 and instances_to_start:
                pass
            else:
                break

        # All retries failed
        self.logger.error(f"Failed to start {model_name} after {max_retries} attempts")
        return False

    def _check_failed_instances_for_compilation_errors(self, vllm_config: Dict[str, Any]) -> List[int]:
        """Check which instances failed due to compilation errors.

        Returns:
            List of instance indices (0-based) that have compilation errors
        """
        log_dir = "./vllm_logs"
        base_port = vllm_config["base_port"]
        instances = vllm_config["instances"]
        model_name = vllm_config["model"]

        failed_instances = []

        # First, check which instances are not responding
        for i in range(instances):
            port = base_port + i
            try:
                response = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
                if response.status_code != 200:
                    # Instance not responding properly
                    if self._check_instance_log_for_compilation_error(port):
                        failed_instances.append(i)
                    else:
                        # Log why this instance failed but won't be retried
                        self._log_non_retriable_failure_reason(port, model_name)
            except:
                # Instance not responding at all
                if self._check_instance_log_for_compilation_error(port):
                    failed_instances.append(i)
                else:
                    # Log why this instance failed but won't be retried
                    self._log_non_retriable_failure_reason(port, model_name)

        return failed_instances

    def _log_non_retriable_failure_reason(self, port: int, model_name: str):
        """Log the reason why an instance failure is not retriable."""
        log_dir = "./vllm_logs"

        for log_file in os.listdir(log_dir):
            if f"port_{port}" in log_file and model_name in log_file and log_file.endswith(".log"):
                log_path = os.path.join(log_dir, log_file)
                try:
                    with open(log_path, "r") as f:
                        content = f.read()

                        if "No available memory for the cache blocks" in content:
                            self.logger.info(f"Instance on port {port} failed due to GPU memory (not retriable)")
                        elif "OutOfMemoryError" in content:
                            self.logger.info(f"Instance on port {port} failed due to CUDA OOM (not retriable)")
                        elif "gpu_memory_utilization" in content:
                            self.logger.info(f"Instance on port {port} failed due to GPU memory config (not retriable)")
                        else:
                            self.logger.info(f"Instance on port {port} failed (reason unknown, not retriable)")
                        return
                except:
                    pass

        self.logger.info(f"Instance on port {port} failed (log not found, not retriable)")

    def _calculate_gpu_groups_for_instances(
        self, vllm_config: Dict[str, Any], instance_indices: List[int]
    ) -> List[List[int]]:
        """Calculate GPU groups for specific instance indices."""
        total_gpus = vllm_config["gpus"]
        total_instances = vllm_config["instances"]
        gpus_per_instance = total_gpus // total_instances

        gpu_groups = []
        for idx in instance_indices:
            start_gpu = idx * gpus_per_instance
            end_gpu = start_gpu + gpus_per_instance
            gpu_groups.append(list(range(start_gpu, end_gpu)))

        return gpu_groups

    def _start_specific_instances(
        self, vllm_config: Dict[str, Any], instance_indices: List[int], compilation_config: int
    ) -> bool:
        """Start specific instances by index."""
        if not instance_indices:
            return True

        base_port = vllm_config["base_port"]
        model_name = vllm_config["model"]
        delay = vllm_config["delay"]

        # Calculate GPU assignments for the specific instances
        gpu_groups = self._calculate_gpu_groups_for_instances(vllm_config, instance_indices)

        self.logger.info(f"Starting {len(instance_indices)} specific instances for {model_name}")

        # Create log directory
        log_dir = "./vllm_logs"
        os.makedirs(log_dir, exist_ok=True)

        # Start each instance
        for i, (instance_idx, gpu_group) in enumerate(zip(instance_indices, gpu_groups)):
            port = base_port + instance_idx
            log_file = os.path.join(log_dir, f"vllm_instance_{instance_idx+1}_port_{port}_{model_name}_retry.log")

            self.logger.info(f"Starting instance {instance_idx+1} on port {port} with GPUs {gpu_group}")

            process = self.vllm_manager.launch_vllm_instance(
                model_name,
                gpu_group,
                port,
                log_file,
                compilation_config,
                "",  # Don't pass "_retry" suffix since it's already in log_file
                vllm_config.get("additional_args"),  # Pass additional args from config
            )

            if process:
                self.logger.info(f"  → Started successfully (PID: {process.pid})")
                self.logger.info(f"  → Log file: {os.path.basename(log_file)}")
            else:
                self.logger.error(f"  → Failed to start")
                return False

            # Add delay between instances except for the last one
            if i < len(instance_indices) - 1:
                self.logger.info(f"Waiting {delay} seconds before launching next instance")
                time.sleep(delay)

        # Wait for the specific instances to be ready
        max_wait = 600  # 10 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            all_responding = True
            for instance_idx in instance_indices:
                port = base_port + instance_idx
                try:
                    response = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
                    if response.status_code != 200:
                        all_responding = False
                        break
                except:
                    all_responding = False
                    break

            if all_responding:
                self.logger.info(f"✅ All {len(instance_indices)} instances are now responding")
                return True

            time.sleep(5)

        self.logger.error(f"❌ Some instances did not start properly within {max_wait}s")
        return False

    def _stop_specific_instances(self, instance_indices: List[int], base_port: int):
        """Stop specific instances by index."""
        self.logger.info(f"Stopping {len(instance_indices)} specific instances")

        for instance_idx in instance_indices:
            port = base_port + instance_idx
            self.logger.info(f"Stopping instance on port {port}")

            # Find and kill the process on this port
            pid = self.vllm_manager.find_pid_by_port(port)
            if pid:
                try:
                    os.killpg(pid, signal.SIGTERM)
                    self.logger.info(f"  → Sent SIGTERM to PID {pid}")
                except:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        self.logger.info(f"  → Sent SIGTERM to PID {pid}")
                    except:
                        pass

        # Wait for processes to stop
        time.sleep(5)

        # Clean up the ports
        ports_to_clean = [base_port + idx for idx in instance_indices]
        self.vllm_manager.cleanup_processes(ports_to_clean)

    def _check_instance_health(self, vllm_config: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        """Check which instances are healthy and which have failed.

        Returns:
            Tuple of (working_ports, failed_ports)
        """
        base_port = vllm_config["base_port"]
        instances = vllm_config["instances"]

        working_instances = []
        failed_instances = []

        for i in range(instances):
            port = base_port + i
            try:
                response = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
                if response.status_code == 200:
                    working_instances.append(port)
                    self.logger.debug(f"Instance on port {port} is healthy")
                else:
                    failed_instances.append(port)
                    self.logger.warning(f"Instance on port {port} returned status {response.status_code}")
            except Exception as e:
                failed_instances.append(port)
                self.logger.warning(f"Instance on port {port} is not responding: {str(e)}")

        return working_instances, failed_instances

    def _check_logs_for_compilation_errors(self, vllm_config: Dict[str, Any]) -> bool:
        """Check all instance logs for compilation errors."""
        log_dir = "./vllm_logs"
        base_port = vllm_config["base_port"]
        instances = vllm_config["instances"]
        model_name = vllm_config["model"]

        compilation_error_found = False

        for i in range(instances):
            port = base_port + i
            # Check for log files for this instance
            for log_file in os.listdir(log_dir):
                if f"port_{port}" in log_file and model_name in log_file and log_file.endswith(".log"):
                    log_path = os.path.join(log_dir, log_file)
                    try:
                        with open(log_path, "r") as f:
                            content = f.read()
                            # Check for compilation-related errors
                            compilation_error_patterns = [
                                "failed to get the hash of the compiled graph",
                                "torch.compile",
                                "compilation failed",
                                "AssertionError",
                            ]
                            for pattern in compilation_error_patterns:
                                if pattern in content.lower():
                                    self.logger.warning(f"Compilation error detected in {log_file}")
                                    compilation_error_found = True
                                    break
                    except Exception as e:
                        self.logger.debug(f"Could not read log file {log_path}: {str(e)}")

                    if compilation_error_found:
                        break

            if compilation_error_found:
                break

        return compilation_error_found

    def _log_startup_failure_reasons(self, vllm_config: Dict[str, Any]):
        """Log detailed reasons for startup failures."""
        log_dir = "./vllm_logs"
        base_port = vllm_config["base_port"]
        instances = vllm_config["instances"]
        model_name = vllm_config["model"]

        self.logger.error("Analyzing startup failure reasons:")

        for i in range(instances):
            port = base_port + i
            instance_num = i + 1

            # Find the most recent log file for this instance
            latest_log = None
            latest_mtime = 0

            for log_file in os.listdir(log_dir):
                if f"port_{port}" in log_file and model_name in log_file and log_file.endswith(".log"):
                    log_path = os.path.join(log_dir, log_file)
                    try:
                        mtime = os.path.getmtime(log_path)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_log = log_path
                    except:
                        pass

            if latest_log:
                self.logger.error(f"  Instance {instance_num} (port {port}):")
                try:
                    with open(latest_log, "r") as f:
                        content = f.read()

                        # Check for specific error types
                        if "OutOfMemoryError" in content:
                            self.logger.error(f"    → CUDA out of memory error")
                        elif "No available memory for the cache blocks" in content:
                            self.logger.error(
                                f"    → GPU memory error: Insufficient memory for cache blocks (increase gpu_memory_utilization)"
                            )
                        elif "gpu_memory_utilization" in content:
                            self.logger.error(f"    → GPU memory configuration issue")
                        elif "failed to get the hash of the compiled graph" in content:
                            self.logger.error(f"    → Compilation error (torch.compile issue)")
                        elif "RuntimeError" in content:
                            # Extract the specific runtime error
                            lines = content.split("\n")
                            for line in lines:
                                if "RuntimeError:" in line:
                                    self.logger.error(f"    → {line.strip()}")
                                    break
                        elif "Application startup complete" in content:
                            self.logger.warning(
                                f"    → Instance appears to have started successfully (may need more time)"
                            )
                        elif "Loading model" in content:
                            self.logger.warning(f"    → Instance still loading model (may need more time)")
                        else:
                            self.logger.error(f"    → Unknown failure (check {os.path.basename(latest_log)})")
                except Exception as e:
                    self.logger.error(f"    → Could not analyze log: {str(e)}")
            else:
                self.logger.error(f"  Instance {instance_num} (port {port}): No log file found")

    def _check_instance_log_for_compilation_error(self, port: int) -> bool:
        """Check if a specific instance log indicates a compilation error."""
        log_dir = "./vllm_logs"

        # Find log files for this port
        for log_file in os.listdir(log_dir):
            if f"port_{port}" in log_file and log_file.endswith(".log"):
                log_path = os.path.join(log_dir, log_file)
                try:
                    with open(log_path, "r") as f:
                        content = f.read()
                        # Check for compilation-related errors
                        # Be more specific to avoid false positives
                        compilation_error_patterns = [
                            "failed to get the hash of the compiled graph",
                            "torch.compile failed",
                            "compilation failed",
                            "AssertionError: failed to get the hash of the compiled graph",
                        ]
                        for pattern in compilation_error_patterns:
                            if pattern in content:
                                self.logger.warning(f"Compilation error detected in {log_file}")
                                return True

                        # Explicitly check that it's NOT a GPU memory error
                        memory_error_patterns = [
                            "No available memory for the cache blocks",
                            "gpu_memory_utilization",
                            "OutOfMemoryError",
                            "CUDA out of memory",
                        ]
                        for pattern in memory_error_patterns:
                            if pattern in content:
                                self.logger.info(f"GPU memory error (not compilation error) detected in {log_file}")
                                return False
                except Exception as e:
                    self.logger.debug(f"Could not read log file {log_path}: {str(e)}")

        return False

    def _log_failure_reasons(self, failed_ports: List[int]):
        """Log detailed reasons for failed instances."""
        log_dir = "./vllm_logs"

        for failed_port in failed_ports:
            self.logger.error(f"Instance on port {failed_port} failed")

            # Try to find and analyze the log file
            for log_file in os.listdir(log_dir):
                if f"port_{failed_port}" in log_file and log_file.endswith(".log"):
                    log_path = os.path.join(log_dir, log_file)
                    try:
                        with open(log_path, "r") as f:
                            content = f.read()

                            # Check for specific error types
                            if "OutOfMemoryError" in content:
                                self.logger.error(f"  → CUDA out of memory error detected")
                            elif "compiled graph" in content.lower() and "error" in content.lower():
                                self.logger.error(f"  → Compilation error detected")
                            elif "RuntimeError" in content:
                                # Extract the specific runtime error
                                lines = content.split("\n")
                                for i, line in enumerate(lines):
                                    if "RuntimeError:" in line:
                                        self.logger.error(f"  → {line.strip()}")
                                        break
                            else:
                                # Check if still loading
                                if "Loading model" in content and "Application startup complete" not in content:
                                    self.logger.warning(f"  → Instance may still be loading (no errors found)")
                                else:
                                    self.logger.error(f"  → Unknown failure reason (check {log_file})")
                    except Exception as e:
                        self.logger.debug(f"Could not analyze log file {log_path}: {str(e)}")

    def _stop_local_model(self):
        """Stop all local model instances."""
        if self.logger:
            self.logger.info("Stopping local model instances")

        # Stop services with better error handling
        try:
            self.vllm_manager.stop_proxy()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Warning while stopping proxy: {str(e)}")

        try:
            self.vllm_manager.stop_instances()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Warning while stopping instances: {str(e)}")

        # Additional cleanup for all configured ports to ensure nothing is left running
        # This handles cases where config might be out of sync with actual running instances
        instance_ports, proxy_ports = self._get_all_configured_ports()
        all_ports = instance_ports + proxy_ports

        if all_ports:
            try:
                if self.logger:
                    self.logger.info(f"Performing comprehensive cleanup of all configured ports: {all_ports}")
                self.vllm_manager.cleanup_processes(all_ports)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Warning during comprehensive cleanup: {str(e)}")

        # Give processes time to clean up
        time.sleep(2)

        # Clean up GPU memory after stopping
        self._cleanup_gpu_memory()

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory by forcing garbage collection and clearing CUDA cache."""
        try:
            import torch
            import gc

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Log GPU memory status
                if self.logger:
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        self.logger.debug(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        except ImportError:
            # torch not available, skip
            pass
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error during GPU memory cleanup: {str(e)}")

    def _start_judge_model(self) -> bool:
        """Start the judge model if it's local."""
        # Detect which judge model type is configured
        judge_models = self.config.get("judge_models", {})

        if "local" in judge_models:
            judge_type = "local"
        elif "remote" in judge_models:
            judge_type = "remote"
        else:
            self.logger.error("No valid judge model configuration found")
            return False

        judge_config = judge_models[judge_type]

        if judge_config["type"] == "local":
            if self.logger:
                self.logger.info("Starting local judge model")
            vllm_config = judge_config["vllm_config"]

            try:
                success = self.vllm_manager.start_instances(
                    model=vllm_config["model"],
                    gpus=vllm_config["gpus"],
                    instances=vllm_config["instances"],
                    base_port=vllm_config["base_port"],
                    delay=vllm_config["delay"],
                    additional_args=vllm_config.get("additional_args"),
                )

                if success:
                    # Start proxy for judge model
                    proxy_success = self.vllm_manager.start_proxy(
                        port=vllm_config["proxy_port"], strategy=vllm_config["strategy"]
                    )

                    if proxy_success:
                        # Wait for all services to be ready
                        services_ready = self._wait_for_vllm_services(
                            proxy_port=vllm_config["proxy_port"], max_wait=1800, check_interval=5  # 30 minutes max wait
                        )

                        if services_ready:
                            self.logger.info("Successfully started judge model")
                            return True
                        else:
                            self.logger.error("Judge model services did not become ready in time")
                            return False
                    else:
                        self.logger.error("Failed to start proxy for judge model")
                        return False
                else:
                    self.logger.error("Failed to start instances for judge model")
                    return False

            except Exception as e:
                self.logger.error(f"Error starting judge model: {str(e)}\n{traceback.format_exc()}")
                return False

        return True  # Remote judge model doesn't need starting

    def _stop_judge_model(self):
        """Stop the judge model if it's local."""
        try:
            # Detect which judge model type is configured
            judge_models = self.config.get("judge_models", {})

            if "local" in judge_models:
                judge_type = "local"
            elif "remote" in judge_models:
                judge_type = "remote"
            else:
                # No judge model configured, nothing to stop
                return

            judge_config = judge_models[judge_type]

            if judge_config["type"] == "local":
                if self.logger:
                    self.logger.info("Stopping local judge model")
                # vLLM manager handles graceful shutdown with proper waits
                try:
                    self.vllm_manager.stop_proxy()
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Warning stopping proxy: {str(e)}")

                try:
                    self.vllm_manager.stop_instances()
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Warning stopping instances: {str(e)}")
                # No additional sleep needed - vLLM manager already waits appropriately
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in _stop_judge_model: {str(e)}")
            # Don't re-raise to prevent recursion issues

    def _run_evaluation_task(self, model: Dict[str, Any], mode: str) -> bool:
        """Run evaluation task for a single model."""
        try:
            # Prepare configuration
            config = self._prepare_model_config(model, mode)
            config_file = self._save_temp_config(config)

            # Run evaluation
            self.logger.info(f"Running {mode} for {model['name']}")
            analysis_report = self.config["evaluation"]["evaluate"]["analysis_report"] if mode == "all" else False

            # Set environment variables for clean subprocess handling
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            env["MULTIPROCESSING_METHOD"] = "spawn"

            # Use subprocess to run eval_V.py
            eval_script_path = os.path.join(os.path.dirname(__file__), "eval_V.py")
            cmd = [sys.executable, eval_script_path, "--config", config_file, "--analysis_report", str(analysis_report)]

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,  # Create new session
            )

            # Track the child process
            self._track_child_process(process.pid)

            # Read output
            output = []
            for line in process.stdout:
                output.append(line)
                self.logger.info(line.strip())

            return_code = process.wait()

            if return_code == 0:
                self.logger.info(f"Successfully completed {mode} for {model['name']}")
                success = True
            else:
                self.logger.error(f"Failed {mode} for {model['name']}: return code {return_code}")
                if output:
                    self.logger.error(f"Output: {''.join(output[-20:])}")  # Last 20 lines
                success = False

            # Clean up temporary config file
            os.unlink(config_file)
            return success

        except Exception as e:
            self.logger.error(f"Error running {mode} for {model['name']}: {str(e)}\n{traceback.format_exc()}")
            return False

    def _run_remote_models_sequentially(self, models: List[Dict[str, Any]], mode: str, state: Dict[str, Any]):
        """Run remote models sequentially (to be called in background thread)."""
        for model in models:
            self.logger.info(f"\nProcessing remote model: {model['name']}")

            # Check if results already exist (only for inference mode)
            if mode == "infer" and self._check_inference_results_exist(model):
                self.logger.info(f"✅ Skipping {model['name']} - inference results already exist")
                with self.state_lock:
                    if model["name"] not in state["completed_models"]:
                        state["completed_models"].append(model["name"])
                    state["completed_count"] += 1
                self._save_state(state)
                continue

            # Run the evaluation task
            success = self._run_evaluation_task(model, mode)

            # Thread-safe state updates
            with self.state_lock:
                if success:
                    if model["name"] not in state["completed_models"]:
                        state["completed_models"].append(model["name"])
                else:
                    if model["name"] not in state["failed_models"]:
                        state["failed_models"].append(model["name"])
                state["completed_count"] += 1

            # Save state after updating
            self._save_state(state)

    def _run_pipeline(self):
        """Run the complete evaluation pipeline (internal method for daemon)."""
        state = self._load_state()
        state["status"] = PipelineState.RUNNING.value
        state["start_time"] = datetime.now().isoformat()
        state["log_file"] = self.current_log_file if self.current_log_file else None
        state["pid"] = os.getpid()
        state["pgid"] = os.getpgid(0)
        state["phase"] = self.phase  # Track which phase(s) are being run
        self._save_state(state)

        # Log which phase is being run
        if self.logger:
            self.logger.info(f"Starting pipeline with phase: {self.phase}")
            if self.phase == "infer":
                self.logger.info("Running INFERENCE phase only")
            elif self.phase == "eval":
                self.logger.info("Running EVALUATION phase only")
            else:
                self.logger.info("Running both INFERENCE and EVALUATION phases")

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            # Prevent multiple signal handling
            if self._shutdown_requested:
                return

            self._shutdown_requested = True
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self._cleanup()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Phase 1: Inference (if requested)
            if self.phase in ["all", "infer"]:
                self._run_inference_phase(state)

            # Phase 2: Evaluation (if requested)
            if self.phase in ["all", "eval"]:
                self._run_evaluation_phase(state)

            # Generate summary report (only if evaluation was run)
            if self.phase in ["all", "eval"]:
                self._generate_summary_report()

            # Update state
            state["status"] = PipelineState.COMPLETED.value
            state["end_time"] = datetime.now().isoformat()
            self._save_state(state)

            elapsed_time = (
                datetime.fromisoformat(state["end_time"]) - datetime.fromisoformat(state["start_time"])
            ).total_seconds()
            self.logger.info(f"\nPipeline completed in {elapsed_time/60:.2f} minutes")

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}\n{traceback.format_exc()}")
            state["status"] = PipelineState.FAILED.value
            state["error"] = str(e)
            state["end_time"] = datetime.now().isoformat()
            self._save_state(state)
            self._cleanup()
            raise
        finally:
            # Always cleanup on exit
            self._cleanup()

    def _run_inference_phase(self, state: Dict[str, Any]):
        """Run the inference phase for all models and datasets."""
        self.logger.info("=" * 60)
        self.logger.info("Starting INFERENCE PHASE")
        self.logger.info("=" * 60)

        state["current_phase"] = "inference"
        state["completed_count"] = 0  # Reset counter at start
        state["completed_models"] = []  # Reset completed models
        state["failed_models"] = []  # Reset failed models
        self._save_state(state)

        models = self._get_all_models()
        total_models = len(models["local"]) + len(models["remote"])
        state["total_models"] = total_models
        self.logger.info(f"Total models to process: {total_models}")
        self._save_state(state)

        # Start remote models in background thread
        remote_future = None
        executor = None
        if models["remote"]:
            self.logger.info(f"Starting {len(models['remote'])} remote models in background...")
            executor = ThreadPoolExecutor(max_workers=1)
            remote_future = executor.submit(self._run_remote_models_sequentially, models["remote"], "infer", state)

        # Run local models in foreground
        for model in models["local"]:
            self.logger.info(f"\nProcessing local model: {model['name']}")
            state["current_model"] = model["name"]
            self._save_state(state)

            try:
                # Check if results already exist
                if self._check_inference_results_exist(model):
                    self.logger.info(f"✅ Skipping {model['name']} - inference results already exist")
                    if model["name"] not in state["completed_models"]:
                        state["completed_models"].append(model["name"])
                    state["completed_count"] = min(state["completed_count"] + 1, total_models)
                    self._save_state(state)
                    continue

                # Start the model
                if not self._start_local_model(model):
                    self.logger.error(f"Failed to start {model['name']}, skipping...")
                    if model["name"] not in state["failed_models"]:
                        state["failed_models"].append(model["name"])
                    state["completed_count"] = min(state["completed_count"] + 1, total_models)
                    self._save_state(state)

                    # Ensure thorough cleanup on failure before continuing
                    self.logger.info(f"Performing thorough cleanup after failure to start {model['name']}")
                    self._stop_local_model()
                    time.sleep(5)  # Give extra time for cleanup
                    continue

                # Run inference
                success = False
                try:
                    success = self._run_evaluation_task(model, "infer")
                    if success:
                        if model["name"] not in state["completed_models"]:
                            state["completed_models"].append(model["name"])
                        self.logger.info(f"✅ Successfully completed inference for {model['name']}")
                    else:
                        if model["name"] not in state["failed_models"]:
                            state["failed_models"].append(model["name"])
                        self.logger.error(f"❌ Inference failed for {model['name']}")
                except Exception as task_e:
                    self.logger.error(f"Exception during inference for {model['name']}: {str(task_e)}")
                    if model["name"] not in state["failed_models"]:
                        state["failed_models"].append(model["name"])

                state["completed_count"] = min(state["completed_count"] + 1, total_models)
                self._save_state(state)

            except Exception as e:
                self.logger.error(f"Unexpected error processing {model['name']}: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                if model["name"] not in state["failed_models"]:
                    state["failed_models"].append(model["name"])
                state["completed_count"] = min(state["completed_count"] + 1, total_models)
                self._save_state(state)

            finally:
                # Always try to stop the model, even if there were errors
                try:
                    self._stop_local_model()
                except Exception as stop_e:
                    self.logger.warning(f"Warning while stopping {model['name']}: {str(stop_e)}")

                # Brief pause between models to allow for cleanup
                time.sleep(3)

        # Wait for remote models to complete
        if remote_future:
            self.logger.info("\nWaiting for remote models to complete...")
            try:
                remote_future.result()  # This will raise any exceptions from the background thread
                self.logger.info("Remote models completed")
            except Exception as e:
                self.logger.error(f"Error in remote model processing: {str(e)}\n{traceback.format_exc()}")
            finally:
                if executor:
                    executor.shutdown(wait=True)

        # Ensure final count is correct
        state["completed_count"] = min(len(state["completed_models"]) + len(state["failed_models"]), total_models)
        self._save_state(state)

        self.logger.info("\nInference phase completed!")
        self.logger.info(f"Completed models: {len(state['completed_models'])}")
        self.logger.info(f"Failed models: {len(state['failed_models'])}")
        self.logger.info(f"Total processed: {state['completed_count']}/{total_models}")

    def _run_evaluation_phase(self, state: Dict[str, Any]):
        """Run the evaluation phase for all models and datasets."""
        self.logger.info("=" * 60)
        self.logger.info("Starting EVALUATION PHASE")
        self.logger.info("=" * 60)

        state["current_phase"] = "evaluation"
        self._save_state(state)

        # If running eval-only, check that inference results exist
        if self.phase == "eval":
            models = self._get_all_models()
            all_models = models["remote"] + models["local"]
            missing_results = []

            for model in all_models:
                # Use the same transformation as in _check_inference_results_exist
                model_name = model["name"].replace("/", "_").replace(".", "_")
                model_output_dir = os.path.join(self.config["work_dir"], model_name)

                if not os.path.exists(model_output_dir):
                    missing_results.append(model["name"])

            if missing_results:
                self.logger.error(
                    f"Cannot run evaluation phase - missing inference results for {len(missing_results)} models:"
                )
                for model in missing_results[:5]:  # Show first 5
                    self.logger.error(f"  - {model}")
                if len(missing_results) > 5:
                    self.logger.error(f"  ... and {len(missing_results) - 5} more")
                self.logger.error("Please run inference phase first with --phase infer or --phase all")
                raise RuntimeError("Missing inference results for evaluation phase")

        # Start judge model if needed
        if not self._start_judge_model():
            self.logger.error("Failed to start judge model, aborting evaluation phase")
            return

        models = self._get_all_models()
        all_models = models["remote"] + models["local"]

        # Set reuse to True for evaluation phase
        self.config["evaluation"]["reuse"] = True

        # Filter out failed models
        remote_models_to_eval = [m for m in models["remote"] if m["name"] not in state["failed_models"]]
        local_models_to_eval = [m for m in models["local"] if m["name"] not in state["failed_models"]]

        # Start remote models in background thread
        remote_future = None
        executor = None
        if remote_models_to_eval:
            self.logger.info(f"Starting evaluation for {len(remote_models_to_eval)} remote models in background...")
            executor = ThreadPoolExecutor(max_workers=1)
            remote_future = executor.submit(self._run_remote_models_sequentially, remote_models_to_eval, "all", state)

        # Run evaluation for local models in foreground
        for model in local_models_to_eval:
            self.logger.info(f"\nEvaluating local model: {model['name']}")
            state["current_model"] = model["name"]
            self._save_state(state)

            self._run_evaluation_task(model, "all")

        # Wait for remote models to complete
        if remote_future:
            self.logger.info("\nWaiting for remote model evaluations to complete...")
            try:
                remote_future.result()  # This will raise any exceptions from the background thread
                self.logger.info("Remote model evaluations completed")
            except Exception as e:
                self.logger.error(f"Error in remote model evaluation: {str(e)}\n{traceback.format_exc()}")
            finally:
                if executor:
                    executor.shutdown(wait=True)

        # Stop judge model if it's local
        self._stop_judge_model()

        self.logger.info("\nEvaluation phase completed!")

    def _generate_summary_report(self):
        """Generate a summary report of all evaluations."""
        self.logger.info("Generating summary report...")

        output_dir = self.config["work_dir"]
        summary_data = {}

        # Collect results from all models
        models = self._get_all_models()
        all_models = models["remote"] + models["local"]

        for model in all_models:
            # Use the same transformation as in _check_inference_results_exist
            model_name = model["name"].replace("/", "_").replace(".", "_")
            model_output_dir = os.path.join(output_dir, model_name)

            if os.path.exists(model_output_dir):
                # Look for result files
                for file in os.listdir(model_output_dir):
                    if file.endswith("_acc.csv") or file.endswith("_score.csv"):
                        file_path = os.path.join(model_output_dir, file)
                        try:
                            import pandas as pd

                            df = pd.read_csv(file_path)
                            dataset_name = file.split("_")[0]

                            if model_name not in summary_data:
                                summary_data[model_name] = {}
                            summary_data[model_name][dataset_name] = df.to_dict()
                        except Exception as e:
                            self.logger.warning(f"Failed to read {file_path}: {str(e)}\n{traceback.format_exc()}")

        # Save summary report
        summary_file = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        self.logger.info(f"Summary report saved to: {summary_file}")

    def _cleanup(self):
        """Clean up any running services."""
        # Prevent recursive cleanup calls
        if self._cleanup_in_progress:
            return

        self._cleanup_in_progress = True

        try:
            if self.logger:
                self.logger.info("Cleaning up services...")

            # Stop services
            try:
                self._stop_judge_model()
            except Exception as e:
                error_msg = str(e)
                if self.logger and "I/O operation on closed file" not in error_msg:
                    self.logger.error(f"Error stopping judge model: {error_msg}")

            try:
                self._stop_local_model()
            except Exception as e:
                error_msg = str(e)
                if self.logger and "I/O operation on closed file" not in error_msg:
                    self.logger.error(f"Error stopping local model: {error_msg}")

            # Final cleanup of any remaining processes on all configured ports
            instance_ports, proxy_ports = self._get_all_configured_ports()
            all_ports = instance_ports + proxy_ports

            if all_ports:
                try:
                    self.vllm_manager.cleanup_processes(all_ports)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during process cleanup: {str(e)}")
        except Exception as e:
            # If logger is closed, try to print to stderr
            try:
                if self.logger:
                    self.logger.error(f"Error during cleanup: {str(e)}")
                else:
                    print(f"Error during cleanup: {str(e)}", file=sys.stderr)
            except:
                pass
        finally:
            # Reset flag only after cleanup is truly complete
            self._cleanup_in_progress = False

    def cleanup(self):
        """Clean up all pipeline resources and reset state."""
        print("Cleaning up pipeline resources...")

        # Stop any running services
        try:
            self.vllm_manager.stop_proxy()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error stopping proxy during cleanup: {str(e)}")

        try:
            self.vllm_manager.stop_instances()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error stopping instances during cleanup: {str(e)}")

        # Reset state
        state = {
            "status": PipelineState.IDLE.value,
            "pid": None,
            "pgid": None,
            "start_time": None,
            "end_time": None,
            "current_model": None,
            "current_phase": None,
            "phase": "all",
            "completed_models": [],
            "failed_models": [],
            "log_file": None,
            "total_models": 0,
            "completed_count": 0,
            "child_pids": [],
            "error": None,
        }
        self._save_state(state)

        # Remove PID file
        try:
            self._remove_pid()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error removing PID file: {str(e)}")

        print("✅ Cleanup completed")

    def start(self, daemon: bool = True):
        """Start the evaluation pipeline."""
        # Check if already running
        state = self._load_state()
        if state["status"] == PipelineState.RUNNING.value:
            pid = state.get("pid")
            if pid and self._is_process_running(pid):
                print(f"Pipeline is already running (PID: {pid})")
                return

        if daemon:
            print("Starting evaluation pipeline as daemon...")

            # Set environment variables for clean process handling
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            env["MULTIPROCESSING_METHOD"] = "spawn"

            # Create log directory if it doesn't exist
            os.makedirs(self.log_dir, exist_ok=True)
            log_file = os.path.join(self.log_dir, f"pipeline_daemon_{time.strftime('%Y%m%d_%H%M%S')}.log")

            # Fork to create daemon process
            try:
                pid = os.fork()
                if pid == 0:
                    # Child process
                    try:
                        os.setsid()  # Create new session
                        pid = os.fork()
                        if pid == 0:
                            # Grandchild process (actual daemon)
                            try:
                                # Close all file descriptors
                                os.closerange(0, 1024)

                                # Open log file for stdout and stderr
                                log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                                os.dup2(log_fd, 1)  # stdout
                                os.dup2(log_fd, 2)  # stderr
                                os.close(log_fd)

                                # Open /dev/null for stdin
                                null_fd = os.open(os.devnull, os.O_RDONLY)
                                os.dup2(null_fd, 0)  # stdin
                                os.close(null_fd)

                                # Setup logging for daemon
                                logging.basicConfig(
                                    level=getattr(logging, self.config["pipeline"]["log_level"]),
                                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                    handlers=[
                                        logging.FileHandler(log_file),
                                        logging.StreamHandler(),  # This will now go to the log file
                                    ],
                                )
                                self.logger = logging.getLogger(__name__)
                                self.current_log_file = log_file  # Set the current log file
                                self.logger.info("Daemon process started successfully")

                                # Save PID and PGID
                                self._save_pid(os.getpid(), os.getpgid(0))

                                # Run pipeline
                                self._run_pipeline()
                            except Exception as e:
                                # Log any startup errors
                                with open(log_file, "a") as f:
                                    f.write(f"Failed to start daemon: {e}\n{traceback.format_exc()}\n")
                            finally:
                                # Clean up PID file
                                self._remove_pid()
                                os._exit(0)
                        else:
                            # First child exits
                            os._exit(0)
                    except Exception as e:
                        self.logger.error(f"Failed to create daemon: {e}\n{traceback.format_exc()}")
                        os._exit(1)
                else:
                    # Parent waits for first child
                    os.waitpid(pid, 0)

                    # Wait a moment for daemon to start
                    time.sleep(2)

                    # Check if daemon started successfully
                    state = self._load_state()
                    if state["status"] == PipelineState.RUNNING.value:
                        print(f"✅ Pipeline started successfully (PID: {state['pid']})")
                        print(f"   Log file: {log_file}")
                        print(f"   Use 'python {sys.argv[0]} status' to check progress")
                        print(f"   Use 'python {sys.argv[0]} logs' to view logs")
                        print(f"   Use 'python {sys.argv[0]} stop' to stop the pipeline")
                    else:
                        print("❌ Failed to start pipeline")
                        # Show any startup errors from the log
                        if os.path.exists(log_file):
                            with open(log_file, "r") as f:
                                print("\nStartup log:")
                                print(f.read())

            except Exception as e:
                self.logger.error(f"Failed to fork process: {e}\n{traceback.format_exc()}")
                raise
        else:
            # Run in foreground
            print("Starting evaluation pipeline in foreground...")
            self._setup_logging(daemon=False)
            self._run_pipeline()

    def stop(self):
        """Stop the running pipeline with graceful shutdown."""
        state = self._load_state()

        if state["status"] != PipelineState.RUNNING.value:
            print("Pipeline is not running")
            return

        pid, pgid = self._get_pid_info()
        if not pid:
            pid = state.get("pid")
            pgid = state.get("pgid")

        if not pid:
            print("No PID found for running pipeline")
            return

        print(f"Stopping pipeline gracefully...")
        print(f"Main process PID: {pid}")
        if pgid:
            print(f"Process Group ID: {pgid}")

        try:
            # Phase 1: Send SIGINT (Ctrl+C) for graceful shutdown
            print("Phase 1: Sending SIGINT for graceful shutdown...")
            if pgid:
                try:
                    os.killpg(pgid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            else:
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass

            # Wait for graceful shutdown
            print("Waiting for graceful shutdown...")
            for i in range(10):
                if not self._is_process_running(pid):
                    print("✅ Pipeline stopped gracefully")
                    break
                time.sleep(1)
                if i % 3 == 0:
                    print(f"  Still waiting... ({10-i}s remaining)")

            # Phase 2: Send SIGTERM if still running
            if self._is_process_running(pid):
                print("\nPhase 2: Sending SIGTERM...")
                if pgid:
                    try:
                        os.killpg(pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

                # Wait for termination
                for i in range(5):
                    if not self._is_process_running(pid):
                        print("✅ Pipeline terminated")
                        break
                    time.sleep(1)

            # Phase 3: Force kill if still running
            if self._is_process_running(pid):
                print("\nPhase 3: Force killing process...")
                if pgid:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                print("✅ Pipeline force killed")

            # Clean up any child processes
            child_pids = state.get("child_pids", [])
            if child_pids:
                print(f"\nCleaning up {len(child_pids)} child processes...")
                for cpid in child_pids:
                    try:
                        os.kill(cpid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                time.sleep(2)

                # Force kill remaining children
                for cpid in child_pids:
                    try:
                        os.kill(cpid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            # Update state
            state["status"] = PipelineState.STOPPED.value
            state["end_time"] = datetime.now().isoformat()
            state["child_pids"] = []
            self._save_state(state)

            # Clean up PID file
            self._remove_pid()

            # Clean up any running vLLM services
            print("\nCleaning up vLLM services...")
            self.vllm_manager.stop_proxy()
            self.vllm_manager.stop_instances()

            print("\n✅ Pipeline stopped successfully")

        except ProcessLookupError:
            print("Pipeline process not found (may have already stopped)")
            state["status"] = PipelineState.STOPPED.value
            state["child_pids"] = []
            self._save_state(state)
            self._remove_pid()
        except Exception as e:
            print(f"Error stopping pipeline: {e}\n{traceback.format_exc()}")

    def status(self):
        """Show pipeline status."""
        state = self._load_state()

        print("=" * 60)
        print("Evaluation Pipeline Status")
        print("=" * 60)

        print(f"Status: {state['status']}")
        print(f"Configuration: {self.config_path}")
        print(f"Pipeline Phase: {state.get('phase', 'all')}")

        if state["pid"]:
            running = self._is_process_running(state["pid"])
            print(f"PID: {state['pid']} {'(running)' if running else '(not running)'}")
            if state.get("pgid"):
                print(f"Process Group ID: {state['pgid']}")

        if state["start_time"]:
            print(f"Start time: {state['start_time']}")

        if state["end_time"]:
            print(f"End time: {state['end_time']}")

        if state["current_phase"]:
            print(f"Current phase: {state['current_phase']}")

        if state["current_model"]:
            print(f"Current model: {state['current_model']}")

        if state.get("error"):
            print(f"\nError: {state['error']}")

        if state["total_models"] > 0:
            print(f"\nProgress: {state['completed_count']}/{state['total_models']} models")
            progress = (state["completed_count"] / state["total_models"]) * 100
            print(f"Progress: {progress:.1f}%")

        if state["completed_models"]:
            print(f"\nCompleted models ({len(state['completed_models'])}):")
            for model in state["completed_models"][-5:]:  # Show last 5
                print(f"  ✅ {model}")
            if len(state["completed_models"]) > 5:
                print(f"  ... and {len(state['completed_models']) - 5} more")

        if state["failed_models"]:
            print(f"\nFailed models ({len(state['failed_models'])}):")
            for model in state["failed_models"]:
                print(f"  ❌ {model}")

        if state.get("child_pids"):
            active_children = [p for p in state["child_pids"] if self._is_process_running(p)]
            if active_children:
                print(f"\nActive child processes: {len(active_children)}")

        if state["log_file"]:
            print(f"\nLog file: {state['log_file']}")

        # Check vLLM services
        print("\n" + "=" * 60)
        print("vLLM Services Status")
        print(f"{'='*60}")
        self.vllm_manager.status()

    def logs(self, tail: bool = False, lines: int = 50):
        """Show pipeline logs."""
        state = self._load_state()

        if not state["log_file"] or not os.path.exists(state["log_file"]):
            # Try to find latest log file
            log_files = sorted(Path(self.log_dir).glob("pipeline*.log"), key=os.path.getmtime)
            if log_files:
                log_file = str(log_files[-1])
            else:
                print("No log files found")
                return
        else:
            log_file = state["log_file"]

        print(f"=" * 60)
        print(f"Pipeline Logs: {log_file}")
        print(f"=" * 60)

        if tail:
            # Follow log file
            subprocess.run(["tail", "-f", log_file])
        else:
            # Show last N lines
            subprocess.run(["tail", f"-{lines}", log_file])

        # Also show vLLM logs if any services are running
        if state["status"] == PipelineState.RUNNING.value:
            print(f"\n{'='*60}")
            print("vLLM Service Logs")
            print(f"{'='*60}")
            self.vllm_manager.logs("all", tail=False)

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running."""
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def cleanup_ipc(self, dry_run: bool = False, force: bool = False):
        """Clean up leaked IPC resources (semaphores and shared memory)."""

        def run_cmd(cmd):
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip(), result.stderr.strip()

        user = pwd.getpwuid(os.getuid()).pw_name
        print(f"Scanning for leaked IPC resources (user: {user})...")

        # Check shared memory
        stdout, _ = run_cmd("ipcs -m")
        orphaned_shm = []
        for line in stdout.split("\n"):
            parts = line.split()
            if len(parts) >= 6 and parts[2] == user:
                nattch = int(parts[5]) if parts[5].isdigit() else 0
                if nattch == 0:  # No processes attached
                    orphaned_shm.append(parts[1])  # shmid

        # Check semaphores
        stdout, _ = run_cmd("ipcs -s")
        user_sem = []
        for line in stdout.split("\n"):
            parts = line.split()
            if len(parts) >= 5 and parts[2] == user:
                user_sem.append(parts[1])  # semid

        print(f"Found {len(orphaned_shm)} orphaned shared memory segments")
        print(f"Found {len(user_sem)} semaphore arrays")

        if not orphaned_shm and not user_sem:
            print("✅ No leaked resources found!")
            return

        if dry_run:
            print("\nDRY RUN MODE - would remove:")
            if orphaned_shm:
                print(f"  - {len(orphaned_shm)} shared memory segments")
            if user_sem and force:
                print(f"  - {len(user_sem)} semaphore arrays")
            print("\nRun without --dry-run to actually clean up")
            return

        # Clean up shared memory
        if orphaned_shm:
            print(f"\nCleaning up {len(orphaned_shm)} shared memory segments...")
            cleaned = 0
            for shmid in orphaned_shm:
                _, stderr = run_cmd(f"ipcrm -m {shmid}")
                if not stderr:
                    cleaned += 1
            print(f"✅ Removed {cleaned} shared memory segments")

        # Clean up semaphores (only with --force)
        if user_sem and force:
            print(f"\nCleaning up {len(user_sem)} semaphore arrays...")
            cleaned = 0
            for semid in user_sem:
                _, stderr = run_cmd(f"ipcrm -s {semid}")
                if not stderr:
                    cleaned += 1
            print(f"✅ Removed {cleaned} semaphore arrays")
        elif user_sem and not force:
            print("\n⚠️  Found semaphore arrays but not removing (use --force)")

        print("\n✅ IPC cleanup complete!")

    def _get_all_configured_ports(self) -> Tuple[List[int], List[int]]:
        """Get all instance and proxy ports configured across all models.

        Returns:
            Tuple of (instance_ports, proxy_ports)
        """
        instance_ports = set()
        proxy_ports = set()

        # Collect ports from local models
        local_models = self.config["models"].get("local_models", [])
        for model in local_models:
            if "vllm_config" in model:
                vllm_config = model["vllm_config"]
                base_port = vllm_config.get("base_port", 7800)
                instances = vllm_config.get("instances", 1)
                proxy_port = vllm_config.get("proxy_port", 7888)

                # Add all instance ports for this model
                for i in range(instances):
                    instance_ports.add(base_port + i)

                # Add proxy port
                proxy_ports.add(proxy_port)

        # Collect ports from judge model if it's local
        judge_models = self.config.get("judge_models", {})
        if "local" in judge_models:
            judge_config = judge_models["local"]
            if "vllm_config" in judge_config:
                vllm_config = judge_config["vllm_config"]
                base_port = vllm_config.get("base_port", 7900)
                instances = vllm_config.get("instances", 1)
                proxy_port = vllm_config.get("proxy_port", 7988)

                # Add all instance ports for judge model
                for i in range(instances):
                    instance_ports.add(base_port + i)

                # Add proxy port
                proxy_ports.add(proxy_port)

        return sorted(list(instance_ports)), sorted(list(proxy_ports))

    def _check_inference_results_exist(self, model: Dict[str, Any]) -> bool:
        """Check if inference results already exist for a model.

        Args:
            model: Model configuration

        Returns:
            True if results exist for ALL datasets, False otherwise
        """
        # Transform model name: replace "/" with "_" and "." with "_"
        model_name = model["name"].replace("/", "_").replace(".", "_")
        model_output_dir = os.path.join(self.config["work_dir"], model_name)

        if not os.path.exists(model_output_dir):
            return False

        # Check for the model subdirectory (which also has dots replaced with underscores)
        model_subdir = os.path.join(model_output_dir, model_name)
        if not os.path.exists(model_subdir):
            return False

        # Get all configured datasets
        datasets = self._get_all_datasets()

        # Check if there are result files for each dataset
        try:
            files = os.listdir(model_subdir)
            # Look for Excel result files (typical output format)
            xlsx_files = [f for f in files if f.endswith(".xlsx")]

            # Check if we have results for all datasets
            missing_datasets = []
            found_datasets = []

            for dataset in datasets:
                # Check if any xlsx file contains this dataset name
                dataset_found = False
                for xlsx_file in xlsx_files:
                    if dataset in xlsx_file:
                        dataset_found = True
                        found_datasets.append(dataset)
                        break

                if not dataset_found:
                    missing_datasets.append(dataset)

            if missing_datasets:
                self.logger.info(f"Incomplete inference results for {model['name']} in {model_subdir}")
                self.logger.info(f"  Found results for: {found_datasets}")
                self.logger.info(f"  Missing results for: {missing_datasets}")
                return False
            else:
                self.logger.info(f"Found complete inference results for {model['name']} in {model_subdir}")
                self.logger.info(f"  Result files: {len(xlsx_files)} files covering all {len(datasets)} datasets")
                return True

        except Exception as e:
            self.logger.warning(f"Error checking results for {model['name']}: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation pipeline for multi-modal models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start pipeline as background daemon (both phases)
  python eval_pipeline.py start --config all_config.yaml
  
  # Start pipeline in foreground
  python eval_pipeline.py start --config all_config.yaml --foreground
  
  # Run inference phase only
  python eval_pipeline.py start --config all_config.yaml --phase infer
  
  # Run evaluation phase only (requires existing inference results)
  python eval_pipeline.py start --config all_config.yaml --phase eval
  
  # Check pipeline status
  python eval_pipeline.py status
  
  # View logs
  python eval_pipeline.py logs
  python eval_pipeline.py logs --tail
  
  # Stop pipeline gracefully
  python eval_pipeline.py stop
  
  # Clean up resources
  python eval_pipeline.py cleanup
  
  # Clean up IPC resources
  python eval_pipeline.py cleanup-ipc --dry-run
  python eval_pipeline.py cleanup-ipc --force
        """,
    )

    parser.add_argument("--state-file", type=str, default="pipeline_state.json", help="Path to pipeline state file")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the evaluation pipeline")
    start_parser.add_argument(
        "--config", type=str, default="all_config.yaml", help="Path to the comprehensive configuration file"
    )
    start_parser.add_argument(
        "--foreground", action="store_true", help="Run pipeline in foreground instead of as daemon"
    )
    start_parser.add_argument(
        "--phase",
        choices=["all", "infer", "eval"],
        default="all",
        help="Phase to run: 'infer' (inference only), 'eval' (evaluation only), or 'all' (both phases)",
    )

    # Stop command
    subparsers.add_parser("stop", help="Stop the running pipeline")

    # Status command
    subparsers.add_parser("status", help="Show pipeline status")

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show pipeline logs")
    logs_parser.add_argument("--tail", action="store_true", help="Follow log output")
    logs_parser.add_argument("--lines", type=int, default=50, help="Number of lines to show (default: 50)")

    # Cleanup command
    subparsers.add_parser("cleanup", help="Clean up all pipeline resources")

    # Cleanup IPC resources
    cleanup_ipc_parser = subparsers.add_parser("cleanup-ipc", help="Clean up leaked IPC resources")
    cleanup_ipc_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be cleaned without doing it"
    )
    cleanup_ipc_parser.add_argument(
        "--force", action="store_true", help="Also remove semaphore arrays (use with caution)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle commands
    if args.command == "start":
        # Create pipeline instance with config from start command
        pipeline = EvaluationPipeline(args.config, args.state_file, phase=args.phase)
        pipeline.start(daemon=not args.foreground)
        return

    # For other commands, use default config
    pipeline = EvaluationPipeline("all_config.yaml", args.state_file)

    if args.command == "stop":
        pipeline.stop()
    elif args.command == "status":
        pipeline.status()
    elif args.command == "logs":
        pipeline.logs(tail=args.tail, lines=args.lines)
    elif args.command == "cleanup":
        pipeline.cleanup()
    elif args.command == "cleanup-ipc":
        pipeline.cleanup_ipc(dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
