"""
Hardware conversion utilities for converting between simulation values and hardware ticks.
"""
import numpy as np


class HardwareConverter:
    """Handles conversion between simulation radians and hardware ticks."""
    
    def __init__(self, config):
        """
        Initialize converter with configuration.
        
        Args:
            config: Config object containing simulation ranges and motor settings
        """
        self.config = config
        self.palm_ids = config.palm_ids
        
        # Set up per-joint simulation ranges
        palm_range = config.get('simulation.palm_range')
        other_range = config.get('simulation.other_range')
        
        self.sim_min_mj = np.full(20, other_range[0], dtype=np.float32)
        self.sim_max_mj = np.full(20, other_range[1], dtype=np.float32)
        
        for mid in self.palm_ids:
            self.sim_min_mj[mid-1] = palm_range[0]
            self.sim_max_mj[mid-1] = palm_range[1]
            
        self.sim_span_mj = self.sim_max_mj - self.sim_min_mj
    
    def ticks_to_sim_rad(self, motor_limits, motor_id: int, ticks: int) -> float:
        """
        Convert hardware ticks to simulation radians for a specific motor.
        
        Args:
            motor_limits: Dictionary containing motor limit information
            motor_id: Motor ID (1-20)
            ticks: Hardware tick value
            
        Returns:
            float: Position in simulation radians
        """
        lo = motor_limits[motor_id]['tight_lo']
        hi = motor_limits[motor_id]['tight_hi']
        idx = motor_id - 1
        
        # Clamp ticks to valid range
        t = min(max(int(ticks), lo), hi)
        
        # Convert to fraction [0, 1]
        frac = (t - lo) / max(1, (hi - lo))
        
        # Map to simulation range
        return float(self.sim_min_mj[idx] + frac * self.sim_span_mj[idx])
    
    def sim_rad_to_ticks(self, motor_limits, motor_id: int, sim_r: float) -> int:
        """
        Convert simulation radians to hardware ticks for a specific motor.
        
        Args:
            motor_limits: Dictionary containing motor limit information
            motor_id: Motor ID (1-20)
            sim_r: Position in simulation radians
            
        Returns:
            int: Hardware tick value
        """
        lo = motor_limits[motor_id]['tight_lo']
        hi = motor_limits[motor_id]['tight_hi']
        idx = motor_id - 1
        
        # Clamp to simulation range
        s = min(max(float(sim_r), self.sim_min_mj[idx]), self.sim_max_mj[idx])
        
        # Convert to fraction [0, 1]
        frac = (s - self.sim_min_mj[idx]) / max(1e-8, self.sim_span_mj[idx])
        
        # Map to tick range
        return int(round(lo + frac * (hi - lo)))
    
    @property
    def sim_min(self) -> np.ndarray:
        """Get minimum simulation values for all joints."""
        return self.sim_min_mj
    
    @property
    def sim_max(self) -> np.ndarray:
        """Get maximum simulation values for all joints."""
        return self.sim_max_mj
    
    @property
    def sim_span(self) -> np.ndarray:
        """Get simulation range spans for all joints."""
        return self.sim_span_mj