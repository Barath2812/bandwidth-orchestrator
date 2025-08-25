from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import os
import json
from utils import get_project_root
import numpy as np

class Dashboard:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        
        # Configure template path
        template_path = os.path.join(get_project_root(), 'templates')
        
        self.app = Flask(__name__, template_folder=template_path)
        self.socketio = SocketIO(self.app, json=json)
        self.running = False
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html',
                                  classes=self.env.get_status()['classes'],
                                  config=self.config)
        
        @self.app.route('/status')
        def get_status():
            return jsonify(self.env.get_status())
        
        @self.app.route('/update_priority', methods=['POST'])
        def update_priority():
            data = request.json
            class_idx = data['class_idx']
            priority = float(data['priority'])
            self.env.update_priority(class_idx, priority)
            return jsonify({"status": "success"})
        
        @self.app.route('/create_event', methods=['POST'])
        def create_event():
            data = request.json
            class_idx = data['class_idx']
            severity = float(data.get('severity', 2.0))
            self.env.create_event(class_idx, severity)
            return jsonify({"status": "success"})
        
        @self.app.route('/config')
        def get_config():
            return jsonify({
                'classes': self.env.get_status()['classes'],
                'config': self.config
            })
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            print('Client connected')
            emit('status_update', self.env.get_status())
        
        @self.socketio.on('request_config')
        def handle_request_config():
            emit('config_data', {
                'classes': self.env.get_status()['classes'],
                'config': self.config
            })
        
        @self.socketio.on('manual_allocate')
        def handle_manual_allocate(data):
            print(f"Manual allocation: {data}")
            emit('allocation_update', data)
        
        @self.socketio.on_error_default
        def default_error_handler(e):
            print(f"SocketIO error: {str(e)}")
    
    def update_clients(self):
        while self.running:
            try:
                status = self.env.get_status()
                self.socketio.emit('status_update', status)
            except Exception as e:
                print(f"Error emitting status update: {str(e)}")
            time.sleep(self.config['dashboard']['refresh_interval'] / 1000)
    
    def run(self):
        dashboard_cfg = self.config['dashboard']
        self.running = True
        update_thread = threading.Thread(target=self.update_clients, daemon=True)
        update_thread.start()
        print(f"Dashboard running at http://{dashboard_cfg['host']}:{dashboard_cfg['port']}")
        self.socketio.run(self.app, 
                         host=dashboard_cfg['host'], 
                         port=dashboard_cfg['port'],
                         use_reloader=False,
                         allow_unsafe_werkzeug=True)