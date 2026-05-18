#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np

def test_inference(model_path):
    print(f"Testing model: {model_path}")
    try:
        sess = ort.InferenceSession(model_path)
        
        # Inputs:
        # locs: [1, N, 3]
        # demand: [1, N, 1]
        # capacity: [1, 1]
        
        N = 5 # 1 depot + 4 customers
        locs = np.random.rand(1, N, 3).astype(np.float32)
        demand = np.array([[[0.0], [0.1], [0.2], [0.1], [0.3]]], dtype=np.float32)
        capacity = np.array([[1.0]], dtype=np.float32)
        
        outputs = sess.run(None, {
            "locs": locs,
            "demand": demand,
            "capacity": capacity
        })
        
        print("Success! Outputs keys:", [o.name for o in sess.get_outputs()])
        # Usually output is 'actions' [1, seq_len]
        actions = outputs[0]
        print("Actions shape:", actions.shape)
        print("Sample actions:", actions)
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_inference("cvrp20_model.onnx")
    test_inference("cvrp50_model.onnx")
