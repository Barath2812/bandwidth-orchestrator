import numpy as np
import random
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class NetworkEnvironment:
    def __init__(self, config):
        self.config = config
        network_cfg = config['network']
        self.classes = network_cfg['classes']
        self.class_names = [c['name'] for c in self.classes]
        self.total_bandwidth = network_cfg['total_bandwidth']
        self.device_criticality = [c['priority'] for c in self.classes]
        self.min_bandwidth = [c['min_bandwidth'] for c in self.classes]
        self.demand_ranges = [c['demand_range'] for c in self.classes]
        self.state_dim = len(self.classes) * 2
        self.reset()
        self.active_events = []
    
    def reset(self):
        self.demands = np.array([
            random.uniform(r[0], r[1]) for r in self.demand_ranges
        ])
        self.allocations = np.zeros(len(self.classes))
        return self._get_state()
    
    def _get_state(self):
        return np.concatenate((self.demands, self.allocations))
    
    def create_event(self, class_idx, severity=1.0, duration=50):
        self.active_events.append({
            'class': class_idx,
            'severity': severity,
            'duration': duration,
            'original_demand': float(self.demands[class_idx])
        })
    
    def update_priority(self, class_idx, new_priority):
        self.device_criticality[class_idx] = float(new_priority)
    
    def step(self, action):
        # Convert action to bandwidth allocation
        allocation = self._action_to_allocation(action)
        
        # Apply minimum bandwidth guarantees
        for i in range(len(self.classes)):
            allocation[i] = max(self.min_bandwidth[i], 
                               min(allocation[i], self.demands[i]))
        
        # Calculate reward
        reward = 0
        unmet_penalty = 0
        
        for i in range(len(self.classes)):
            fraction_met = min(1, allocation[i] / max(1, self.demands[i]))
            reward += self.device_criticality[i] * fraction_met
            
            # Additional reward for critical traffic
            if i == 0 and fraction_met > 0.95:
                reward += 1.0
            if i == 1 and fraction_met > 0.90:
                reward += 0.5
                
            # Penalty for unmet demands
            unmet = max(0, self.demands[i] - allocation[i])
            unmet_penalty += unmet * (len(self.classes) - i)
        
        reward -= unmet_penalty / 100
        
        # Generate new demands
        self.demands = np.array([
            max(self.demand_ranges[i][0], 
                min(self.demand_ranges[i][1],
                    self.demands[i] + random.uniform(-5, 5)))
            for i in range(len(self.classes))
        ])
        
        # Apply active events
        for event in self.active_events[:]:
            class_idx = event['class']
            self.demands[class_idx] = min(
                self.demand_ranges[class_idx][1],
                event['original_demand'] * event['severity']
            )
            event['duration'] -= 1
            if event['duration'] <= 0:
                self.active_events.remove(event)
        
        # Random critical event
        if random.random() < 0.1:
            class_idx = random.randint(0, len(self.classes)-2)  # Exclude non-critical
            self.create_event(class_idx, severity=random.uniform(1.5, 3.0))
        
        self.allocations = allocation
        return self._get_state(), reward
    
    def _action_to_allocation(self, action):
        allocation = np.array(action) * self.total_bandwidth
        total_requested = sum(allocation)
        if total_requested > self.total_bandwidth:
            scale_factor = self.total_bandwidth / total_requested
            allocation = allocation * scale_factor
        return allocation
    
    def get_status(self):
        """Return a JSON-serializable status dictionary"""
        status = {
            'demands': self.demands.tolist(),
            'allocations': self.allocations.tolist(),
            'priorities': [float(p) for p in self.device_criticality],
            'events': self.active_events,
            'classes': [
                {'name': name, 'priority': float(priority)} 
                for name, priority in zip(self.class_names, self.device_criticality)
            ]
        }
        return json.loads(json.dumps(status, cls=NumpyEncoder))