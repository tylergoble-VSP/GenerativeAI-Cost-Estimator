"""
DGX Spark Performance Monitoring Module

This module provides reusable functions for monitoring NVIDIA DGX Spark systems,
including SSH connection management, system metrics collection, and GPU monitoring.

Key Features:
- SSH connection management with automatic reconnection
- System metrics (CPU, RAM, disk, network)
- GPU metrics via nvidia-smi and NVML
- VPN connection monitoring
- Data collection and export
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class SSHConnection:
    """
    Manages SSH connection to remote DGX Spark system.
    
    This class handles SSH connections with automatic reconnection,
    command execution, and connection health monitoring.
    """
    
    def __init__(self, hostname: str, username: str, 
                 key_path: Optional[str] = None, 
                 password: Optional[str] = None,
                 port: int = 22):
        """
        Initialize SSH connection parameters.
        
        Args:
            hostname: SSH hostname or IP address
            username: SSH username
            key_path: Path to SSH private key file (optional)
            password: SSH password (optional, less secure than key)
            port: SSH port (default: 22)
        """
        self.hostname = hostname
        self.username = username
        self.key_path = key_path
        self.password = password
        self.port = port
        self.client: Optional[paramiko.SSHClient] = None
        self.connected = False
        self.last_connection_time: Optional[float] = None
    
    def connect(self, timeout: int = 10) -> bool:
        """
        Establish SSH connection to remote host.
        
        Args:
            timeout: Connection timeout in seconds
        
        Returns:
            True if connection successful, False otherwise
        """
        if not PARAMIKO_AVAILABLE:
            print("Warning: paramiko not available. Cannot establish SSH connection.")
            return False
        
        try:
            # Create SSH client
            self.client = paramiko.SSHClient()
            
            # Automatically add host key (for convenience, in production use known_hosts)
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Load SSH key if provided
            pkey = None
            if self.key_path:
                try:
                    pkey = paramiko.RSAKey.from_private_key_file(self.key_path)
                except Exception as e:
                    print(f"Warning: Could not load SSH key from {self.key_path}: {e}")
            
            # Connect to remote host
            self.client.connect(
                hostname=self.hostname,
                username=self.username,
                pkey=pkey,
                password=self.password,
                port=self.port,
                timeout=timeout
            )
            
            self.connected = True
            self.last_connection_time = time.time()
            return True
            
        except Exception as e:
            print(f"SSH connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close SSH connection."""
        if self.client:
            try:
                self.client.close()
            except:
                pass
        self.connected = False
        self.client = None
    
    def is_connected(self) -> bool:
        """Check if SSH connection is active."""
        if not self.connected or not self.client:
            return False
        
        # Try to check if connection is still alive
        try:
            transport = self.client.get_transport()
            if transport and transport.is_active():
                return True
        except:
            pass
        
        self.connected = False
        return False
    
    def execute_command(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """
        Execute a command on the remote host via SSH.
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
        
        Returns:
            Tuple of (success: bool, stdout: str, stderr: str)
        """
        if not self.is_connected():
            # Try to reconnect
            if not self.connect():
                return False, "", "Not connected and reconnection failed"
        
        try:
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
            
            # Read output
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            # Check exit status
            exit_status = stdout.channel.recv_exit_status()
            success = exit_status == 0
            
            return success, stdout_text, stderr_text
            
        except Exception as e:
            return False, "", f"Command execution error: {str(e)}"
    
    def reconnect(self, max_attempts: int = 3, backoff: float = 2.0) -> bool:
        """
        Reconnect to SSH with exponential backoff.
        
        Args:
            max_attempts: Maximum reconnection attempts
            backoff: Initial backoff time in seconds (doubles each attempt)
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self.disconnect()
        
        for attempt in range(max_attempts):
            wait_time = backoff * (2 ** attempt)
            if attempt > 0:
                print(f"Reconnection attempt {attempt + 1}/{max_attempts} after {wait_time:.1f}s...")
                time.sleep(wait_time)
            
            if self.connect():
                print("SSH reconnection successful!")
                return True
        
        print("SSH reconnection failed after all attempts")
        return False


def check_vpn_status(ssh: Optional[SSHConnection] = None) -> Dict[str, Any]:
    """
    Check VPN connection status (Tailscale or other).
    
    Args:
        ssh: Optional SSH connection to remote host (if None, checks local)
    
    Returns:
        Dictionary with VPN status information
    """
    status = {
        "vpn_available": False,
        "vpn_type": None,
        "connected": False,
        "status_message": ""
    }
    
    # Check for Tailscale
    if ssh:
        # Check remote Tailscale status
        success, stdout, stderr = ssh.execute_command("which tailscale && tailscale status 2>/dev/null || echo 'not_found'")
        if success and "not_found" not in stdout:
            status["vpn_available"] = True
            status["vpn_type"] = "tailscale"
            if "100." in stdout or "connected" in stdout.lower():
                status["connected"] = True
                status["status_message"] = "Tailscale connected"
            else:
                status["status_message"] = "Tailscale installed but not connected"
    else:
        # Check local Tailscale status
        try:
            result = subprocess.run(
                ["tailscale", "status"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                status["vpn_available"] = True
                status["vpn_type"] = "tailscale"
                if "100." in result.stdout or "connected" in result.stdout.lower():
                    status["connected"] = True
                    status["status_message"] = "Tailscale connected"
                else:
                    status["status_message"] = "Tailscale installed but not connected"
        except FileNotFoundError:
            status["status_message"] = "Tailscale not found"
        except Exception as e:
            status["status_message"] = f"Error checking Tailscale: {e}"
    
    return status


def get_gpu_metrics_nvidia_smi(ssh: Optional[SSHConnection] = None) -> List[Dict[str, Any]]:
    """
    Get GPU metrics using nvidia-smi command.
    
    This function parses nvidia-smi output to get GPU information.
    Works both locally and remotely via SSH.
    
    Args:
        ssh: Optional SSH connection to remote host
    
    Returns:
        List of dictionaries, one per GPU, with metrics
    """
    # Build nvidia-smi command with JSON output for easier parsing
    command = "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits"
    
    if ssh:
        success, stdout, stderr = ssh.execute_command(command)
        if not success:
            return []
        output = stdout
    else:
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return []
            output = result.stdout
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")
            return []
    
    gpus = []
    for line in output.strip().split('\n'):
        if not line.strip():
            continue
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 9:
            try:
                gpu = {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu": float(parts[2]),
                    "utilization_memory": float(parts[3]),
                    "memory_used_mb": float(parts[4]),
                    "memory_total_mb": float(parts[5]),
                    "temperature_c": float(parts[6]),
                    "power_draw_w": float(parts[7]),
                    "power_limit_w": float(parts[8]),
                    "memory_used_pct": (float(parts[4]) / float(parts[5])) * 100 if float(parts[5]) > 0 else 0
                }
                gpus.append(gpu)
            except (ValueError, IndexError) as e:
                print(f"Error parsing GPU data: {e}")
                continue
    
    return gpus


def get_gpu_metrics_nvml(ssh: Optional[SSHConnection] = None) -> List[Dict[str, Any]]:
    """
    Get GPU metrics using NVML (NVIDIA Management Library).
    
    This provides programmatic access to GPU metrics.
    Note: NVML must be available on the system where this runs.
    
    Args:
        ssh: Not used for NVML (must run locally)
    
    Returns:
        List of dictionaries, one per GPU, with metrics
    """
    if not PYNVML_AVAILABLE:
        return []
    
    if ssh:
        # NVML can't be used remotely, fall back to nvidia-smi
        return get_gpu_metrics_nvidia_smi(ssh)
    
    try:
        # Initialize NVML
        pynvml.nvmlInit()
        
        gpus = []
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get GPU name
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Get utilization rates
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Get power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
            except:
                power = 0.0
                power_limit = 0.0
            
            gpu = {
                "index": i,
                "name": name,
                "utilization_gpu": util.gpu,
                "utilization_memory": util.memory,
                "memory_used_mb": mem_info.used / (1024 * 1024),
                "memory_total_mb": mem_info.total / (1024 * 1024),
                "temperature_c": temp,
                "power_draw_w": power,
                "power_limit_w": power_limit,
                "memory_used_pct": (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
            }
            gpus.append(gpu)
        
        return gpus
        
    except Exception as e:
        print(f"Error getting GPU metrics via NVML: {e}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def get_system_metrics(ssh: Optional[SSHConnection] = None) -> Dict[str, Any]:
    """
    Get system metrics (CPU, RAM, disk) using psutil.
    
    Args:
        ssh: Optional SSH connection (if None, gets local metrics)
    
    Returns:
        Dictionary with system metrics
    """
    if not PSUTIL_AVAILABLE:
        return {}
    
    if ssh:
        # For remote, we need to get metrics via SSH commands
        # This is a simplified version - could be enhanced
        success, stdout, stderr = ssh.execute_command(
            "top -bn1 | head -5 && df -h && free -h"
        )
        # Parse output would go here - for now return basic structure
        return {
            "cpu_percent": 0.0,
            "cpu_count": 0,
            "memory_total_gb": 0.0,
            "memory_used_gb": 0.0,
            "memory_percent": 0.0,
            "disk_usage": {}
        }
    
    # Local metrics using psutil
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = {}
        disk_io = psutil.disk_io_counters()
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    "total_gb": usage.total / (1024**3),
                    "used_gb": usage.used / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "percent": usage.percent
                }
            except PermissionError:
                # Skip partitions we can't access
                continue
        
        return {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "cpu_per_core": cpu_per_core,
            "load_avg_1min": load_avg[0],
            "load_avg_5min": load_avg[1],
            "load_avg_15min": load_avg[2],
            "memory_total_gb": memory.total / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "swap_total_gb": swap.total / (1024**3),
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent,
            "disk_usage": disk_usage,
            "disk_read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
            "disk_write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0
        }
    except Exception as e:
        print(f"Error getting system metrics: {e}")
        return {}


def get_network_metrics(ssh: Optional[SSHConnection] = None) -> Dict[str, Any]:
    """
    Get network interface metrics.
    
    Args:
        ssh: Optional SSH connection
    
    Returns:
        Dictionary with network metrics
    """
    if not PSUTIL_AVAILABLE:
        return {}
    
    if ssh:
        # Remote network metrics via SSH
        success, stdout, stderr = ssh.execute_command("cat /proc/net/dev")
        # Parse would go here
        return {}
    
    try:
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        
        # Get per-interface stats
        interfaces = {}
        for interface, addrs in psutil.net_if_addrs().items():
            stats = psutil.net_if_stats().get(interface)
            if stats:
                interfaces[interface] = {
                    "isup": stats.isup,
                    "speed_mbps": stats.speed,
                    "mtu": stats.mtu
                }
        
        return {
            "bytes_sent_mb": net_io.bytes_sent / (1024**2),
            "bytes_recv_mb": net_io.bytes_recv / (1024**2),
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "connections": net_connections,
            "interfaces": interfaces
        }
    except Exception as e:
        print(f"Error getting network metrics: {e}")
        return {}


def collect_all_metrics(ssh: Optional[SSHConnection] = None,
                        use_nvml: bool = False) -> Dict[str, Any]:
    """
    Collect all metrics (GPU, system, network) in one call.
    
    Args:
        ssh: Optional SSH connection
        use_nvml: Whether to use NVML for GPU metrics (local only)
    
    Returns:
        Dictionary with all collected metrics and timestamp
    """
    timestamp = datetime.now().isoformat()
    timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get GPU metrics
    if use_nvml and not ssh:
        gpu_metrics = get_gpu_metrics_nvml(ssh)
    else:
        gpu_metrics = get_gpu_metrics_nvidia_smi(ssh)
    
    # Get system metrics
    system_metrics = get_system_metrics(ssh)
    
    # Get network metrics
    network_metrics = get_network_metrics(ssh)
    
    # Get VPN status
    vpn_status = check_vpn_status(ssh)
    
    return {
        "timestamp": timestamp,
        "timestamp_readable": timestamp_readable,
        "gpu_metrics": gpu_metrics,
        "system_metrics": system_metrics,
        "network_metrics": network_metrics,
        "vpn_status": vpn_status,
        "ssh_connected": ssh.is_connected() if ssh else False
    }


def export_metrics_to_json(metrics_list: List[Dict[str, Any]], filepath: Path):
    """
    Export collected metrics to JSON file.
    
    Args:
        metrics_list: List of metric dictionaries from collect_all_metrics()
        filepath: Path to save JSON file
    """
    if isinstance(filepath, Path):
        filepath = str(filepath)
    
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "export_timestamp_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(metrics_list),
        "metrics": metrics_list
    }
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)


def export_metrics_to_csv(metrics_list: List[Dict[str, Any]], filepath: Path):
    """
    Export collected metrics to CSV file (flattened format).
    
    Args:
        metrics_list: List of metric dictionaries
        filepath: Path to save CSV file
    """
    import pandas as pd
    
    if isinstance(filepath, Path):
        filepath = str(filepath)
    
    # Flatten metrics for CSV
    rows = []
    for metric in metrics_list:
        row = {
            "timestamp": metric.get("timestamp"),
            "timestamp_readable": metric.get("timestamp_readable"),
            "ssh_connected": metric.get("ssh_connected", False)
        }
        
        # Add GPU metrics (one row per GPU)
        gpus = metric.get("gpu_metrics", [])
        if gpus:
            for gpu in gpus:
                gpu_row = row.copy()
                gpu_row.update({f"gpu_{k}": v for k, v in gpu.items()})
                rows.append(gpu_row)
        else:
            # No GPUs, just add system metrics
            row.update({f"system_{k}": v for k, v in metric.get("system_metrics", {}).items() if not isinstance(v, (dict, list))})
            row.update({f"network_{k}": v for k, v in metric.get("network_metrics", {}).items() if not isinstance(v, dict)})
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    else:
        # Create empty CSV with headers
        with open(filepath, 'w') as f:
            f.write("timestamp,timestamp_readable,ssh_connected\n")

