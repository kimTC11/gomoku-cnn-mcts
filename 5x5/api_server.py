from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import os
import game
from alphazero import NNetWrapper, MCTS, load_config

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load configuration
args = load_config('config.yaml')
g = game.GomokuGame(args.board_size)
nnet = NNetWrapper(g, args)

# Try to load checkpoint if it exists and is valid
checkpoint_path = os.path.join(args.checkpoint, 'best.pth.tar')
model_loaded = False

if os.path.exists(checkpoint_path):
    try:
        nnet.load_checkpoint(args.checkpoint, 'best.pth.tar')
        model_loaded = True
        print(f"✓ Successfully loaded model from {checkpoint_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not load checkpoint: {e}")
        print("⚠ Starting with untrained model - AI will make random moves")
        model_loaded = False
else:
    print(f"⚠ No checkpoint found at {checkpoint_path}")
    print("⚠ Starting with untrained model - AI will make random moves")

# Create MCTS with reduced simulations for faster web gameplay
# Override numMCTSSims for web API (training uses config.yaml value)
from alphazero import dotdict
web_args = dotdict(vars(args))
web_args['numMCTSSims'] = 800  # Reduced from 400 for faster response
mcts = MCTS(g, nnet, web_args)

@app.route('/api/move', methods=['GET'])
def get_move():
    try:
        board_size = int(request.args.get('boardSize', 5))
        matrix_str = request.args.get('matrix', '[]')
        next_player = int(request.args.get('nextMove', 1))
        
        # Safely parse the matrix
        import json
        matrix = json.loads(matrix_str)
        board = np.array(matrix)
        
        # Recreate game with correct board size if needed
        if board_size != g.n:
            game_instance = game.GomokuGame(board_size)
        else:
            game_instance = g
        
        canonical_board = game_instance.getCanonicalForm(board, next_player)
        
        # Get valid moves
        valid_moves = game_instance.getValidMoves(canonical_board, 1)
        
        if not model_loaded:
            # If no model, make random valid move
            valid_actions = np.where(valid_moves == 1)[0]
            if len(valid_actions) == 0:
                return jsonify({'error': 'No valid moves available'}), 400
            action = np.random.choice(valid_actions)
        else:
            # Get AI move using MCTS
            pi = mcts.getActionProb(canonical_board, temp=0)
            action = np.argmax(pi)
        
        row = action // board_size
        col = action % board_size
        
        print(f"AI move: row={row}, col={col}, player={next_player}")
        return jsonify({'row': int(row), 'col': int(col)})
    except Exception as e:
        print(f"Error in get_move: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'ok', 
        'apiServer': {'reachable': True},
        'model_loaded': model_loaded,
        'board_size': args.board_size
    })

if __name__ == '__main__':
    PORT = 5001  # Port for 5x5 game
    
    print("\n" + "="*60)
    print("🎮 5x5 Gomoku AI Server Starting...")
    print("="*60)
    print(f"Model Status: {'✓ Loaded' if model_loaded else '⚠ Untrained (Random moves)'}")
    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"Server URL: http://localhost:{PORT}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=True)
