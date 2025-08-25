import threading
import time
from utils import load_config, parse_args, merge_config
from environment import NetworkEnvironment
from agent import BandwidthAgent
from dashboard import Dashboard

def train_model(config, model_path):
    env = NetworkEnvironment(config)
    agent = BandwidthAgent(env.state_dim, len(config['network']['classes']), config)
    
    total_reward = 0
    state = env.reset()
    
    for step in range(config['training']['steps']):
        action = agent.select_action(state)
        next_state, reward = env.step(action)
        total_reward += reward
        
        agent.store_transition(state, action, reward, next_state)
        agent.train()
        
        state = next_state
        
        if step % 100 == 0:
            print(f"Step {step}/{config['training']['steps']}: "
                  f"Reward={reward:.2f}, Total={total_reward:.2f}")
    
    agent.save_model(model_path)
    print(f"Training completed. Model saved to {model_path}")
    return env, agent

def run_dashboard(env, agent, config):
    dashboard = Dashboard(env, agent, config)
    dashboard.run()

def deploy_model(config, model_path):
    env = NetworkEnvironment(config)
    agent = BandwidthAgent(env.state_dim, len(config['network']['classes']), config)
    agent.load_model(model_path)
    
    print("Running in deployment mode with live dashboard")
    dashboard_thread = threading.Thread(
        target=run_dashboard, 
        args=(env, agent, config),
        daemon=True
    )
    dashboard_thread.start()
    
    # Simulation loop
    state = env.reset()
    while True:
        action = agent.select_action(state)
        next_state, _ = env.step(action)
        state = next_state
        time.sleep(0.1)  # 100ms control loop

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(config, args)
    
    if args.mode == 'train':
        train_model(config, args.model)
    elif args.mode == 'deploy':
        deploy_model(config, args.model)
    elif args.mode == 'dashboard':
        env = NetworkEnvironment(config)
        agent = BandwidthAgent(env.state_dim, len(config['network']['classes']), config)
        if args.model:
            agent.load_model(args.model)
        run_dashboard(env, agent, config)