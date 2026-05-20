import json
import torch
import torch.nn as nn
import numpy as np
import random

import os

# 1. Load Embeddings
embeddings_path = 'mile_end_embeddings.json'
if not os.path.exists(embeddings_path):
    embeddings_path = '/mnt/mile_end_embeddings.json'

if os.path.exists(embeddings_path):
    with open(embeddings_path, 'r') as f:
        emb_data = json.load(f)
        nodes = emb_data['nodes']
    # Map node_idx to vector
        node_vectors = {n['node_idx']: n['vector'] for n in nodes}
else:
    node_vectors = {}

# 2. Simple Drone Environment
class DroneEnv:
    def __init__(self, node_vectors, num_stops=5):
        self.node_vectors = node_vectors
        self.node_indices = list(node_vectors.keys())
        self.num_stops = num_stops
        self.reset()

    def reset(self):
        self.current_node = random.choice(self.node_indices)
        self.targets = random.sample(self.node_indices, self.num_stops)
        self.battery = 100.0
        self.wind_speed = np.random.uniform(0, 30)
        self.payload = np.random.uniform(1, 10)
        self.delivered = 0
        return self._get_state()

    def _get_state(self):
        # State: [Embedding(64), Wind(1), Battery(1), Payload(1)] = 67 dimensions
        emb = np.array(self.node_vectors[self.current_node])
        meta = np.array([self.wind_speed / 30.0, self.battery / 100.0, self.payload / 10.0])
        return np.concatenate([emb, meta])

    def step(self, action_node_idx):
        # Calculate Reward
        # Reward: +(Delivered) - (Battery Cost) - (Wind Risk)
        
        # Simulating battery cost based on distance and wind
        dist = 0.5 # dummy distance
        cost = dist * (1.0 + self.payload * 0.1) * (1.0 + self.wind_speed * 0.05)
        self.battery -= cost
        
        reward = 0
        done = False
        
        if action_node_idx in self.targets:
            self.delivered += 1
            self.targets.remove(action_node_idx)
            reward += 10.0 # Big reward for delivery
            
        reward -= (cost * 0.1) # Battery penalty
        
        if self.wind_speed > 25:
            reward -= 5.0 # High wind risk penalty
            
        if self.battery <= 0 or not self.targets:
            done = True
            
        self.current_node = action_node_idx
        return self._get_state(), reward, done

# 3. Neural Network (Policy)
class PolicyNet(nn.Module):
    def __init__(self, input_dim=67, output_dim=128):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Scorer for a node
        )

    def forward(self, x):
        return self.fc(x)

# 4. Training Loop (Simplified)
def train():
    env = DroneEnv(node_vectors)
    model = PolicyNet()
    
    print("Starting training...")
    for episode in range(100):
        state = env.reset()
        total_reward = 0
        for _ in range(20):
            # Agent looks at 5 random neighbor candidates
            candidates = random.sample(env.node_indices, 5)
            scores = []
            for c in candidates:
                # Prepare state for candidate
                c_emb = np.array(node_vectors[c])
                c_state = np.concatenate([c_emb, state[64:]])
                score = model(torch.FloatTensor(c_state))
                scores.append(score)
            
            action_idx = candidates[torch.argmax(torch.cat(scores)).item()]
            next_state, reward, done = env.step(action_idx)
            
            total_reward += reward
            state = next_state
            if done:
                break
            
        print(f"Episode {episode}: Reward {total_reward:.2f}")

    # 5. Export to ONNX
    dummy_input = torch.randn(1, 67)
    onnx_path = "drone_agent.onnx"
    if os.path.exists("/data"):
        onnx_path = "/data/drone_agent.onnx"
        
    torch.onnx.export(model, dummy_input, onnx_path)
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    train()
