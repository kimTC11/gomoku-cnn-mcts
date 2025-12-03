"""
Simple Neural Network module for TensorFlow-free prediction
"""
import numpy as np

class SimpleNeuralNetwork:
    """A simple neural network predictor that doesn't require TensorFlow"""
    
    def __init__(self, weights):
        self.weights = weights
        self.layer_names = sorted([k for k in weights.keys() if k != 'top_level_model_weights'])
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def predict(self, X):
        """Make prediction using extracted weights"""
        try:
            x = np.array(X)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Forward pass through layers
            for layer_name in self.layer_names:
                layer_weights = self.weights[layer_name]
                
                if 'kernel:0' in layer_weights and 'bias:0' in layer_weights:
                    W = layer_weights['kernel:0']
                    b = layer_weights['bias:0']
                    
                    # Linear transformation
                    x = np.dot(x, W) + b
                    
                    # Apply activation (assuming ReLU for hidden layers, sigmoid for output)
                    if layer_name == self.layer_names[-1]:  # Output layer
                        x = self.sigmoid(x)
                    else:  # Hidden layers
                        x = np.maximum(0, x)  # ReLU
            
            return x
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return np.array([[0.5]])  # Default prediction