import yaml
import argparse
import os

def load_config(file_path="config/default.yaml"):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='Bandwidth Orchestrator')
    parser.add_argument('--mode', choices=['train', 'deploy', 'dashboard'], 
                        default='train', help='Operation mode')
    parser.add_argument('--config', default='config/default.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--model', default='models/bandwidth_orchestrator.pth', 
                        help='Path to trained model')
    parser.add_argument('--steps', type=int, help='Override training steps')
    parser.add_argument('--bandwidth', type=int, help='Override total bandwidth')
    return parser.parse_args()

def merge_config(config, args):
    if args.steps:
        config['training']['steps'] = args.steps
    if args.bandwidth:
        config['network']['total_bandwidth'] = args.bandwidth
    return config

def get_project_root():
    """Get absolute path to project root"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)