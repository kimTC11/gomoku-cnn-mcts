from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import os
import logging
import game
from aphazero_utilized import NNetWrapper, MCTS, load_config, dotdict

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
        log.info(f"✓ Successfully loaded model from {checkpoint_path}")
    except Exception as e:
        log.warning(f"⚠ Warning: Could not load checkpoint: {e}")
        log.warning("⚠ Starting with untrained model - AI will make random moves")
        model_loaded = False
else:
    log.warning(f"⚠ No checkpoint found at {checkpoint_path}")
    log.warning("⚠ Starting with untrained model - AI will make random moves")

# Create MCTS with hybrid GPU + multithreading for fast web gameplay
web_args = dotdict({
    'numMCTSSims': 800,  # Number of MCTS simulations
    'cpuct': 1.0,
    'use_threads': True,  # Enable multithreading
    'num_threads': 8,  # Number of parallel threads
    'cuda': args.cuda,
    'mps': args.mps
})

mcts = MCTS(g, nnet, web_args)

log.info(f"✓ MCTS initialized with {web_args['num_threads']} threads for parallel simulations")
log.info(f"✓ GPU acceleration: {'CUDA' if args.cuda else 'MPS' if args.mps else 'CPU'}")

@app.route('/api/move', methods=['GET', 'POST'])
def get_move():
    """
    Get AI move for the current board state.
    
    Query Parameters (GET) or JSON Body (POST):
    - boardSize: int (default: 10)
    - matrix: 2D array representing the board
    - nextMove: int (1 or -1, player to move)
    
    Returns:
    - row: int
    - col: int
    - confidence: float (optional, policy value)
    """
    try:
        if request.method == 'POST':
            data = request.get_json()
            board_size = int(data.get('boardSize', 10))
            matrix = data.get('matrix', [])
            next_player = int(data.get('nextMove', 1))
        else:
            board_size = int(request.args.get('boardSize', 10))
            matrix_str = request.args.get('matrix', '[]')
            next_player = int(request.args.get('nextMove', 1))
            
            # Safely parse the matrix
            import json
            matrix = json.loads(matrix_str)
        
        board = np.array(matrix)
        
        # Validate board dimensions
        if board.shape != (board_size, board_size):
            return jsonify({
                'error': f'Board shape {board.shape} does not match boardSize {board_size}x{board_size}'
            }), 400
        
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
            confidence = 1.0 / len(valid_actions)  # Uniform distribution
        else:
            # Get AI move using hybrid GPU + multithreaded MCTS
            pi = mcts.getActionProb(canonical_board, temp=0)
            action = np.argmax(pi)
            confidence = float(pi[action])
        
        row = action // board_size
        col = action % board_size
        
        log.info(f"AI move: row={row}, col={col}, player={next_player}, confidence={confidence:.4f}")
        
        return jsonify({
            'row': int(row), 
            'col': int(col),
            'confidence': float(confidence),
            'model_loaded': model_loaded
        })
        
    except Exception as e:
        log.error(f"Error in get_move: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_position():
    """
    Evaluate a board position and return policy distribution.
    
    JSON Body:
    - boardSize: int
    - matrix: 2D array
    - nextMove: int
    
    Returns:
    - policy: array of move probabilities
    - value: estimated win probability for current player
    - best_moves: top 5 moves with coordinates and probabilities
    """
    try:
        data = request.get_json()
        board_size = int(data.get('boardSize', 10))
        matrix = data.get('matrix', [])
        next_player = int(data.get('nextMove', 1))
        
        board = np.array(matrix)
        
        if board_size != g.n:
            game_instance = game.GomokuGame(board_size)
        else:
            game_instance = g
        
        canonical_board = game_instance.getCanonicalForm(board, next_player)
        
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Get policy and value from MCTS
        pi = mcts.getActionProb(canonical_board, temp=1)  # temp=1 for probability distribution
        
        # Get neural network value estimation
        value = nnet.predict(canonical_board)[1][0]
        
        # Find top 5 moves
        top_indices = np.argsort(pi)[-5:][::-1]
        best_moves = []
        for idx in top_indices:
            if pi[idx] > 0:
                row = idx // board_size
                col = idx % board_size
                best_moves.append({
                    'row': int(row),
                    'col': int(col),
                    'probability': float(pi[idx])
                })
        
        return jsonify({
            'policy': pi.tolist(),
            'value': float(value),
            'best_moves': best_moves,
            'board_size': board_size
        })
        
    except Exception as e:
        log.error(f"Error in evaluate_position: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """
    Get server status and configuration.
    """
    return jsonify({
        'status': 'ok', 
        'apiServer': {'reachable': True},
        'model_loaded': model_loaded,
        'board_size': args.board_size,
        'config': {
            'mcts_simulations': web_args['numMCTSSims'],
            'num_threads': web_args['num_threads'],
            'use_threads': web_args['use_threads'],
            'device': args.device,
            'cuda_enabled': args.cuda,
            'mps_enabled': args.mps
        }
    })

@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """
    Get or update MCTS configuration.
    
    GET: Returns current configuration
    POST: Updates configuration (JSON body with numMCTSSims, num_threads, etc.)
    """
    global web_args, mcts
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            # Update configuration
            if 'numMCTSSims' in data:
                web_args['numMCTSSims'] = int(data['numMCTSSims'])
            if 'num_threads' in data:
                web_args['num_threads'] = int(data['num_threads'])
            if 'use_threads' in data:
                web_args['use_threads'] = bool(data['use_threads'])
            if 'cpuct' in data:
                web_args['cpuct'] = float(data['cpuct'])
            
            # Recreate MCTS with new configuration
            mcts = MCTS(g, nnet, web_args)
            
            log.info(f"✓ Configuration updated: {web_args}")
            
            return jsonify({
                'status': 'updated',
                'config': dict(web_args)
            })
        except Exception as e:
            log.error(f"Error updating config: {e}")
            return jsonify({'error': str(e)}), 400
    
    # GET request
    return jsonify({
        'config': dict(web_args)
    })

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    """
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5002))  # Changed from 5000 to avoid conflict
    
    print("\n" + "="*70)
    print("🎮 Gomoku AI Server (Optimized with GPU + Multithreading)")
    print("="*70)
    print(f"Model Status: {'✓ Loaded' if model_loaded else '⚠ Untrained (Random moves)'}")
    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"Device: {args.device.upper()}")
    print(f"MCTS Simulations: {web_args['numMCTSSims']}")
    print(f"Parallel Threads: {web_args['num_threads']}")
    print(f"Multithreading: {'✓ Enabled' if web_args['use_threads'] else '✗ Disabled'}")
    print(f"\nEndpoints:")
    print(f"  • GET/POST /api/move       - Get AI move")
    print(f"  • POST     /api/evaluate   - Evaluate position")
    print(f"  • GET      /api/status     - Server status")
    print(f"  • GET/POST /api/config     - View/update config")
    print(f"  • GET      /api/health     - Health check")
    print(f"\nServer URL: http://localhost:{PORT}")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
