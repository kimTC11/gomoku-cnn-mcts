# Unified Game Interface

This directory provides a unified web interface for playing both 3x3 Tic-Tac-Toe (ML-based) and Gomoku (AlphaZero-based) games.

## Architecture

```
┌─────────────────┐
│   Web Browser   │
│  (index.html)   │
└────────┬────────┘
         │ http://localhost:8080
         ▼
┌─────────────────┐
│  Proxy Server   │
│   (port 8080)   │
└────────┬────────┘
         │
         ├─ /api/3x3/*  ──► ML API (port 5003)    [3x3 ML models]
         │
         └─ /api/*      ──► AlphaZero API (port 5002) [MCTS + Neural Net]
```

## Game Modes

### 1. 3x3 Tic-Tac-Toe (ML AI)
- **AI Engine**: Ensemble of XGBoost and Neural Network
- **Models**: Pre-trained on 255,168 game positions
- **Strategy**: Probabilistic prediction with rule-based fallback
- **API Port**: 5003
- **Training**: See `../3x3/mlp.ipynb`

### 2. Gomoku (AlphaZero)
- **AI Engine**: Monte Carlo Tree Search (MCTS) + Deep Neural Network
- **Board Sizes**: 3x3, 5x5, 10x10 (configurable)
- **Win Conditions**: 3-in-a-row or 5-in-a-row
- **API Port**: 5002
- **Training**: AlphaZero self-play reinforcement learning

## Quick Start

### Option 1: Start All Servers (Recommended)

```bash
cd /Users/kimcuong/source/python/tictactoe-cnn-mcts/10x10
./start_all_servers.sh
```

This will start:
- 3x3 ML API Server (port 5003)
- AlphaZero API Server (port 5002)
- Proxy Server (port 8080)

Then open: http://localhost:8080

### Option 2: Manual Start (Individual Terminals)

**Terminal 1 - 3x3 ML API:**
```bash
cd /Users/kimcuong/source/python/tictactoe-cnn-mcts/3x3
source ../.venv/bin/activate
python api_server.py
```

**Terminal 2 - AlphaZero API:**
```bash
cd /Users/kimcuong/source/python/tictactoe-cnn-mcts/10x10
source ../.venv/bin/activate
python api_server.py
```

**Terminal 3 - Proxy Server:**
```bash
cd /Users/kimcuong/source/python/tictactoe-cnn-mcts/10x10
source ../.venv/bin/activate
python proxy_server.py
```

Then open: http://localhost:8080

## Remote Access with Ngrok

To share the game over the internet:

```bash
ngrok http 8080
```

Share the generated URL (e.g., `https://xxxx-xxx-xxx-xxx.ngrok-free.app`)

## Using the Interface

1. **Select Game Mode**:
   - "3x3 Tic-Tac-Toe (ML AI)" - Classic tic-tac-toe with ML predictions
   - "Gomoku (AlphaZero)" - Customizable Gomoku with advanced AI

2. **Configure Settings**:
   - **Who Goes First**: Choose if you or AI starts
   - **AI Plays As**: Select X or O for the AI
   - For Gomoku mode:
     - **Board Size**: 3x3, 5x5, or 10x10
     - **Win Condition**: 3 or 5 in a row

3. **Play**:
   - Click cells to make your move
   - AI responds automatically
   - Use "New Game" to restart with current settings
   - Use "Reset Board" to clear the board

## API Endpoints

### 3x3 ML API (port 5003)

**Get AI Move:**
```
GET /api/move?boardSize=3&nextMove=-1&matrix=[[1,0,-1],[0,0,0],[0,0,0]]
```

Response:
```json
{
  "row": 1,
  "col": 1,
  "model_used": "ensemble",
  "confidence": 0.85
}
```

**Check Status:**
```
GET /api/status
```

### AlphaZero API (port 5002)

**Get AI Move:**
```
GET /api/move?boardSize=10&winLength=5&nextMove=-1&matrix=[[...]]&last_move_row=4&last_move_col=5
```

Response:
```json
{
  "row": 5,
  "col": 6
}
```

**Check Status:**
```
GET /api/status
```

## Files

- `index.html` - Unified web interface with game mode selector
- `proxy_server.py` - Routes requests to appropriate API server
- `api_server.py` (10x10/) - AlphaZero API server
- `api_server.py` (3x3/) - ML models API server
- `start_all_servers.sh` - Convenient startup script

## Dependencies

All dependencies are managed through `uv` and installed in `../.venv`:

- `flask` - Web framework for APIs
- `flask-cors` - CORS support
- `keras` - Neural network models (3x3 ML)
- `xgboost` - Gradient boosting models (3x3 ML)
- `category_encoders` - Feature encoding
- `pytorch` - Deep learning (AlphaZero)
- `numpy`, `pandas` - Data processing
- `requests` - HTTP client (proxy)

## Troubleshooting

### "Cannot connect to API server"
- Check that all 3 servers are running
- Verify ports 5002, 5003, and 8080 are not in use
- Check logs: `tail -f ../3x3/ml3x3.log` or `tail -f alphazero.log`

### "Models not loaded" (3x3 ML)
- Ensure model files exist in `../3x3/models/`
- Required files:
  - `xgboost_model.pkl`
  - `neural_network_model.h5`
  - `catboost_encoder.pkl`
  - `label_encoder.pkl`

### "AI makes invalid moves"
- Restart the API servers
- Check that board state is being sent correctly (see browser console)

### AlphaZero is slow
- First move takes ~10-15 seconds (model loading + MCTS)
- Subsequent moves are faster (~2-5 seconds)
- Reduce MCTS simulations in `api_server.py` for faster responses

## Development

### Retrain 3x3 ML Models
See `../3x3/mlp.ipynb` for training pipeline:
1. Download dataset from Google Drive
2. Train XGBoost and Neural Network models
3. Evaluate performance
4. Save models to `models/` directory

### Modify AlphaZero
Edit `../alphazero.py` and `../game.py` for game logic changes.
Retrain with: `python alphazero.py`

### Update UI
Edit `index.html` for visual changes or new features.
Changes take effect immediately (just refresh browser).

## Performance

### 3x3 ML API
- Response time: ~50-200ms
- Memory usage: ~500MB (models loaded)
- Accuracy: ~85% match rate with optimal play

### AlphaZero API
- Response time: 2-15 seconds (depends on MCTS depth)
- Memory usage: ~800MB (neural network + MCTS tree)
- Strength: Near-optimal play after sufficient training

## License

See project root for license information.
