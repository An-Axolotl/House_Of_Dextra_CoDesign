import yaml
import os
import numpy as np
from typing import Dict, Any

class Config:
    """Configuration loader and manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default=None):
        """Get config value using dot notation (e.g., 'dynamixel.baudrate')."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set config value using dot notation."""
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    # Convenience properties for commonly used values
    @property
    def policy_default_path(self) -> str:
        return self.get('policy.default_path')
    
    @property
    def act_moving_avg(self) -> float:
        return self.get('control.act_moving_avg')
    
    @property
    def control_hz(self) -> float:
        return self.get('control.control_hz')
    
    @property
    def default_joint_positions_isaac(self) -> np.ndarray:
        return np.array(self.get('joint_positions.default_isaac_order'), dtype=np.float32)
    
    @property
    def dynamixel_device_name(self) -> str:
        return self.get('dynamixel.device_name')
    
    @property
    def dynamixel_baudrate(self) -> int:
        return self.get('dynamixel.baudrate')
    
    @property
    def dynamixel_protocol_version(self) -> float:
        return self.get('dynamixel.protocol_version')
    
    @property
    def motor_ids(self) -> list:
        return self.get('dynamixel.motor_ids')
    
    @property
    def palm_ids(self) -> set:
        return set(self.get('dynamixel.palm_ids'))
    
    @property
    def limit_offsets(self) -> list:
        return self.get('limit_offsets')
    
    @property
    def pid_gains(self) -> list:
        return self.get('pid_gains')
    
    @property
    def object_one_hot(self) -> np.ndarray:
        return np.array(self.get('observation.object_one_hot'), dtype=np.float32)
    
    @property
    def history_frames(self) -> int:
        return self.get('observation.history_frames')
    
    def get_dynamixel_address(self, name: str) -> int:
        return self.get(f'dynamixel.addresses.{name}')
    
    def get_data_length(self, name: str) -> int:
        return self.get(f'dynamixel.data_lengths.{name}')