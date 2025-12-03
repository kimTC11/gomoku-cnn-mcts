from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global variables for models
models = {
    'neural_model': None,
    'xgboost_model': None,
    'catboost_encoder': None,
    'label_encoder': None
}

def load_models():
    """Load pre-trained ML models"""
    global models
    
    # Load encoders
    try:
        if os.path.exists('models/catboost_encoder.pkl'):
            with open('models/catboost_encoder.pkl', 'rb') as f:
                models['catboost_encoder'] = pickle.load(f)
            print("✅ CatBoost encoder loaded")
        
        if os.path.exists('models/xgboost_model.pkl'):
            with open('models/xgboost_model.pkl', 'rb') as f:
                models['xgboost_model'] = pickle.load(f)
            print("✅ XGBoost model loaded")
        
        if os.path.exists('models/simple_neural_predictor.pkl'):
            with open('models/simple_neural_predictor.pkl', 'rb') as f:
                models['neural_model'] = pickle.load(f)
            print("✅ Simple Neural Network loaded")
        elif os.path.exists('models/neural_network_model.h5'):
            print("⚠️  Found .h5 model but not converted")
            print("💡 Run 'python create_simple_nn.py' to convert")
                    
    except Exception as e:
        print(f"⚠️  Error loading models: {e}")

def board_to_string_format(board):
    """Convert 3x3 board to string format for model prediction"""
    # Ensure board is a proper 2D structure
    if not isinstance(board, (list, np.ndarray)):
        raise ValueError(f"Board must be a list or numpy array, got {type(board)}")
    
    # Convert to numpy array for safe indexing
    board_array = np.array(board)
    
    if board_array.shape != (3, 3):
        raise ValueError(f"Board must be 3x3, got shape {board_array.shape}")
    
    string_board = []
    for i in range(3):
        for j in range(3):
            cell_value = board_array[i, j]
            if cell_value == 1:  # X
                string_board.append('x')
            elif cell_value == -1:  # O (changed from 0 to match web format)
                string_board.append('o')
            else:  # Empty
                string_board.append('b')
    return string_board

def get_available_moves(board):
    """Get all available moves on the board"""
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:  # Empty cell
                moves.append((i, j))
    return moves

def check_winner(board):
    """Check for winner or draw"""
    # Check rows
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
    
    # Check columns
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] != 0:
            return board[0][j]
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    
    # Check draw
    if all(board[i][j] != 0 for i in range(3) for j in range(3)):
        return "Draw"
    
    return None

def ai_move_simple(board):
    """Simple rule-based AI"""
    available = get_available_moves(board)
    
    if not available:
        return None
    
    # Current player from API (will be -1 for AI)
    ai_player = -1
    human_player = 1
    
    # 1. Try to win
    for move in available:
        test_board = [row[:] for row in board]
        test_board[move[0]][move[1]] = ai_player
        if check_winner(test_board) == ai_player:
            return move
    
    # 2. Block human from winning
    for move in available:
        test_board = [row[:] for row in board]
        test_board[move[0]][move[1]] = human_player
        if check_winner(test_board) == human_player:
            return move
    
    # 3. Take center
    if (1, 1) in available:
        return (1, 1)
    
    # 4. Take corner
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    corner_moves = [move for move in available if move in corners]
    if corner_moves:
        return random.choice(corner_moves)
    
    # 5. Random choice
    return random.choice(available)

def ai_move_ml(board):
    """AI using XGBoost - fast predictions only"""
    available = get_available_moves(board)
    
    if not available:
        return None
    
    # Ensure board is numpy array
    board_array = np.array(board)
    
    # FAST PATH: Check for immediate win (AI is -1)
    for move in available:
        test_board = board_array.copy()
        test_board[move[0]][move[1]] = -1
        if check_winner(test_board) == -1:
            print(f"⚡ Win move")
            return move
    
    # FAST PATH: Block opponent from winning (Player is 1)
    for move in available:
        test_board = board_array.copy()
        test_board[move[0]][move[1]] = 1
        if check_winner(test_board) == 1:
            print(f"⚡ Block move")
            return move
    
    # FAST PATH: Take center (strong opening)
    if (1, 1) in available:
        print(f"⚡ Center")
        return (1, 1)
    
    # Use XGBoost only (fast)
    xgboost_model = models['xgboost_model'] 
    catboost_encoder = models['catboost_encoder']
    
    if not catboost_encoder or not xgboost_model:
        # Fallback: corner or random
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        corner_moves = [move for move in available if move in corners]
        return random.choice(corner_moves) if corner_moves else random.choice(available)
    
    try:
        best_move = None
        best_score = -1
        
        for move in available:
            # Try the move - AI is -1, but model trained with 1=X winning
            test_board = board_array.copy()
            test_board[move[0]][move[1]] = 1  # Evaluate as X
            
            # Convert to string format
            string_board = board_to_string_format(test_board)
            
            # Create DataFrame
            columns = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']
            board_df = pd.DataFrame([string_board], columns=columns)
            board_encoded = catboost_encoder.transform(board_df)
            
            # XGBoost prediction only
            xgb_prob = xgboost_model.predict_proba(board_encoded)[0][1]
            if xgb_prob > best_score:
                best_score = xgb_prob
                best_move = move
        
        if best_move:
            print(f"🧠 ML: {best_move}")
            return best_move
        
        # Fallback
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        corner_moves = [move for move in available if move in corners]
        return random.choice(corner_moves) if corner_moves else random.choice(available)
        
    except Exception as e:
        print(f"❌ ML error: {e}")
        # Fallback
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        corner_moves = [move for move in available if move in corners]
        return random.choice(corner_moves) if corner_moves else random.choice(available)

@app.route('/api/move', methods=['GET'])
def get_move():
    try:
        board_size = int(request.args.get('boardSize', 3))
        matrix_str = request.args.get('matrix', '[]')
        next_player = int(request.args.get('nextMove', -1))
        
        print(f"\n🎮 3x3 Move Request | Player: {next_player}")
        
        # Validate board size
        if board_size != 3:
            return jsonify({'error': 'This API only supports 3x3 board'}), 400
        
        # Parse the matrix
        import json
        matrix = json.loads(matrix_str)
        
        # Validate matrix structure
        if not isinstance(matrix, list):
            return jsonify({'error': 'Matrix must be a list'}), 400
        
        if len(matrix) != 3:
            return jsonify({'error': f'Matrix must have 3 rows, got {len(matrix)}'}), 400
        
        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != 3:
                return jsonify({'error': f'Row {i} must be a list of length 3'}), 400
        
        # Convert to numpy array for validation
        board = np.array(matrix)
        
        # Get AI move using ML models if available
        if models['xgboost_model'] and models['catboost_encoder']:
            move = ai_move_ml(board)
        else:
            move = ai_move_simple(board)
        
        if move is None:
            return jsonify({'error': 'No valid moves available'}), 400
        
        row, col = move
        print(f"✅ Move: ({row}, {col})\n")
        
        return jsonify({'row': int(row), 'col': int(col)})
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    model_status = {
        'catboost_encoder': models['catboost_encoder'] is not None,
        'xgboost_model': models['xgboost_model'] is not None,
        'neural_model': models['neural_model'] is not None
    }
    
    return jsonify({
        'status': 'ok', 
        'apiServer': {'reachable': True},
        'models_loaded': model_status,
        'board_size': 3,
        'game_type': '3x3 Tic-Tac-Toe with ML'
    })

if __name__ == '__main__':
    PORT = 5003  # Different port from 10x10 API
    
    print("\n" + "="*60)
    print("🎮 3x3 Tic-Tac-Toe ML API Server Starting...")
    print("="*60)
    
    # Load models
    load_models()
    
    model_count = sum([
        models['catboost_encoder'] is not None,
        models['xgboost_model'] is not None,
        models['neural_model'] is not None
    ])
    
    print(f"Models Loaded: {model_count}/3")
    print(f"Board Size: 3x3")
    print(f"Server URL: http://localhost:{PORT}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=True)
