# Bandwidth Orchestrator

A reinforcement learning-based network bandwidth management system that dynamically allocates bandwidth resources across different traffic classes using Deep Q-Network (DQN) algorithms.

## 🚀 Overview

The Bandwidth Orchestrator is an intelligent network management system that uses reinforcement learning to optimize bandwidth allocation in real-time. It automatically adjusts bandwidth distribution across different traffic classes (ultra-critical, high-critical, non-critical) based on current demands, priorities, and network conditions.

## ✨ Features

- **Reinforcement Learning**: Uses Deep Q-Network (DQN) for intelligent bandwidth allocation
- **Real-time Dashboard**: Live web-based monitoring interface
- **Multi-class Traffic Management**: Supports different traffic classes with varying priorities
- **Dynamic Adaptation**: Automatically responds to changing network demands
- **Event Simulation**: Simulates critical network events for testing
- **Configurable Parameters**: Easy configuration through YAML files

## 🏗️ Architecture

### Core Components

1. **BandwidthAgent** (`src/agent.py`)
   - Implements DQN with experience replay
   - Uses target network for stable training
   - Epsilon-greedy exploration strategy
   - Neural network: 4-layer fully connected network

2. **NetworkEnvironment** (`src/environment.py`)
   - Simulates network traffic and demands
   - Manages traffic classes and priorities
   - Generates rewards based on allocation efficiency
   - Handles critical events and demand fluctuations

3. **Dashboard** (`src/dashboard.py`)
   - Real-time web interface
   - Visualizes network status and allocations
   - Interactive controls for testing

4. **Main Controller** (`src/main.py`)
   - Orchestrates training and deployment
   - Manages threading for concurrent operations

## 📋 Requirements

- Python 3.8+
- PyTorch 2.1.0
- NumPy 1.26.0
- Flask 3.0.2
- Flask-SocketIO 5.3.6
- Matplotlib 3.8.0
- PyYAML 6.0.1

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bandwidth-orchestrator
   ```

2. **Install dependencies**
   ```bash
   pip install -r models/requirements.txt
   ```

3. **Verify installation**
   ```bash
   python src/main.py --help
   ```

## 🚀 Usage

### Training Mode

Train a new model with custom parameters:

```bash
python src/main.py --mode train --config config/default.yaml --model models/bandwidth_orchestrator.pth
```

### Deployment Mode

Deploy a trained model with live dashboard:

```bash
python src/main.py --mode deploy --config config/default.yaml --model models/bandwidth_orchestrator.pth
```

### Dashboard Only

Run the dashboard interface:

```bash
python src/main.py --mode dashboard --config config/default.yaml
```

### Command Line Options

- `--mode`: Operation mode (`train`, `deploy`, `dashboard`)
- `--config`: Configuration file path
- `--model`: Model file path for loading/saving
- `--steps`: Number of training steps (overrides config)
- `--learning_rate`: Learning rate (overrides config)

## ⚙️ Configuration

The system is configured through YAML files. Key configuration sections:

### Network Configuration
```yaml
network:
  total_bandwidth: 100
  classes:
    - name: ultra-critical
      min_bandwidth: 0
      priority: 3
      demand_range: [5, 40]
    - name: high-critical
      min_bandwidth: 0
      priority: 2
      demand_range: [10, 50]
    - name: non-critical
      min_bandwidth: 15
      priority: 1
      demand_range: [20, 60]
```

### Training Configuration
```yaml
training:
  steps: 3000
  learning_rate: 0.0005
  batch_size: 128
  memory_capacity: 10000
  gamma: 0.99
  eps_start: 1.0
  eps_end: 0.01
  eps_decay: 0.998
  tau: 0.01
```

## 🧠 Algorithm Details

### Deep Q-Network (DQN)

The system uses a DQN implementation with the following features:

- **Neural Network Architecture**: 4-layer fully connected network
  - Input layer: State dimension
  - Hidden layers: 256 → 256 → 128 neurons
  - Output layer: Action dimension
  - Activation: ReLU

- **Experience Replay**: Stores transitions in a circular buffer
- **Target Network**: Separate network for stable Q-value estimation
- **Soft Updates**: Gradual target network updates using tau parameter
- **Exploration Strategy**: Epsilon-greedy with exponential decay

### State Space
- Current bandwidth demands for each traffic class
- Current bandwidth allocations for each traffic class

### Action Space
- Bandwidth allocation ratios for each traffic class
- Actions are continuous values between 0.1 and 0.9

### Reward Function
- Positive reward for meeting demands (weighted by priority)
- Bonus rewards for critical traffic classes
- Penalties for unmet demands
- Penalty scaling based on traffic class importance

## 📊 Dashboard Features

The web dashboard provides:

- **Real-time Monitoring**: Live updates of network status
- **Bandwidth Visualization**: Charts showing demands vs allocations
- **Event Tracking**: Display of active critical events
- **Performance Metrics**: Training loss and reward history
- **Interactive Controls**: Manual event triggering and priority adjustment

## 🔧 Development

### Project Structure
```
bandwidth-orchestrator/
├── config/
│   └── default.yaml          # Configuration file
├── models/
│   ├── bandwidth_orchestrator.pth  # Trained model
│   ├── requirements.txt      # Dependencies
│   └── README.md            # Model documentation
├── src/
│   ├── agent.py             # DQN agent implementation
│   ├── environment.py       # Network environment simulation
│   ├── dashboard.py         # Web dashboard
│   ├── main.py             # Main controller
│   └── utils.py            # Utility functions
├── templates/
│   └── index.html          # Dashboard template
└── README.md               # This file
```

### Adding New Features

1. **New Traffic Classes**: Modify `config/default.yaml`
2. **Custom Reward Functions**: Edit `src/environment.py`
3. **Different RL Algorithms**: Extend `src/agent.py`
4. **Additional Dashboard Features**: Modify `src/dashboard.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

- PyTorch for the deep learning framework
- Flask for the web dashboard
- The reinforcement learning community for DQN algorithms

## 📞 Support

For questions or issues, please open an issue on the repository or contact the development team.
