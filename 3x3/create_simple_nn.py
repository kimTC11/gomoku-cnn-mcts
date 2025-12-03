#!/usr/bin/env python3
"""
Convert Keras .h5 model to a simpler format that avoids TensorFlow loading issues
"""
import os
import pickle
import numpy as np
import pandas as pd
from simple_neural_network import SimpleNeuralNetwork

def extract_model_weights():
    """Extract model weights and create a simple predictor"""
    print("🔄 Attempting to extract model weights without loading TensorFlow...")
    
    try:
        # Try to load with h5py directly to extract weights
        import h5py
        
        model_path = "models/neural_network_model.h5"
        print(f"📁 Reading {model_path} with h5py...")
        
        weights = {}
        with h5py.File(model_path, 'r') as f:
            # Print structure
            print("📋 Model structure:")
            def print_structure(name, obj):
                print(f"  {name}: {obj}")
            
            f.visititems(print_structure)
            
            # Try to extract weights
            if 'model_weights' in f:
                model_weights = f['model_weights']
                print("✅ Found model_weights group")
                
                for layer_name in model_weights.keys():
                    print(f"📦 Layer: {layer_name}")
                    layer_group = model_weights[layer_name]
                    
                    # Navigate through the nested structure
                    if 'sequential' in layer_group:
                        seq_group = layer_group['sequential']
                        
                        # Find the actual layer (dense, dense_1, etc.)
                        for actual_layer_name in seq_group.keys():
                            actual_layer = seq_group[actual_layer_name]
                            layer_weights = {}
                            
                            for weight_name in actual_layer.keys():
                                weight_data = actual_layer[weight_name][:]
                                layer_weights[f"{weight_name}:0"] = weight_data
                                print(f"  {weight_name}: {weight_data.shape}")
                            
                            weights[layer_name] = layer_weights
                            break
        
        return weights
        
    except Exception as e:
        print(f"❌ Weight extraction failed: {e}")
        return None

def create_simple_neural_predictor(weights):
    """Create a simple neural network predictor using extracted weights"""
    if not weights:
        return None
    
    print("🧠 Creating simple neural network predictor...")
    predictor = SimpleNeuralNetwork(weights)
    print(f"📋 Available layers: {predictor.layer_names}")
    return predictor

def test_simple_predictor():
    """Test the simple predictor"""
    print("🧪 Testing simple neural network predictor...")
    
    # Extract weights
    weights = extract_model_weights()
    if not weights:
        print("❌ Cannot create simple predictor - weight extraction failed")
        return False
    
    # Create predictor
    predictor = create_simple_neural_predictor(weights)
    if not predictor:
        print("❌ Cannot create simple predictor")
        return False
    
    # Test prediction
    try:
        # Load encoders for proper test
        with open('models/catboost_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        # Test board
        test_board = ['b', 'b', 'b', 'b', 'x', 'b', 'b', 'b', 'b']
        columns = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']
        board_df = pd.DataFrame([test_board], columns=columns)
        board_encoded = encoder.transform(board_df)
        
        # Make prediction
        prediction = predictor.predict(board_encoded)
        print(f"🔮 Simple NN prediction: {prediction[0][0]:.4f}")
        
        # Save the predictor
        with open('models/simple_neural_predictor.pkl', 'wb') as f:
            pickle.dump(predictor, f)
        print("✅ Simple predictor saved to models/simple_neural_predictor.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple predictor test failed: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Creating TensorFlow-free Neural Network predictor")
    print("=" * 50)
    
    success = test_simple_predictor()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Simple NN predictor created!")
        print("💡 You can now use XGBoost + Simple NN without TensorFlow")
    else:
        print("💥 FAILURE: Could not create simple predictor")
        print("💡 Recommendation: Use XGBoost-only mode")
    
    return success

if __name__ == "__main__":
    main()