# Gomoku CNN MCTS

A deep reinforcement learning implementation of Gomoku (Five-in-a-Row) using AlphaZero algorithm with Convolutional Neural Networks and Monte Carlo Tree Search.

## 🎯 Features

- **AlphaZero Algorithm**: State-of-the-art reinforcement learning approach
- **CNN Architecture**: Deep convolutional neural network for position evaluation
- **MCTS Integration**: Monte Carlo Tree Search for strategic planning
- **GPU Acceleration**: Support for CUDA and Apple Silicon (MPS)
- **Interactive Play**: Human vs AI gameplay with GUI
- **Training Visualization**: Weights & Biases integration for monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- GPU support (optional but recommended)

### Installation

# Terminal 1: API Server
cd /Users/kimcuong/source/python/tictactoe-cnn-mcts/10x10
uv run api_server.py

# Terminal 2: Proxy Server  
cd /Users/kimcuong/source/python/tictactoe-cnn-mcts/10x10
uv run proxy_server.py

# Terminal 3: Ngrok
ngrok http 8080

```bash
# Clone the repository
git clone https://github.com/kimTC11/gomoku-cnn-mcts.git
cd gomoku-cnn-mcts

# Install dependencies
pip install torch numpy tqdm wandb pygame
```

### Usage

#### Train a New Model

```bash
python alphazero.py --train --wandb
```

#### Play Against Trained Model

```bash
python alphazero.py --play \
    --round=2 \
    --player1=human \
    --player2=alphazero \
    --ckpt_file=temp.pth.tar \
    --verbose
```

## 🎮 Game Options

- `--player1`: Choose player 1 type (`human`, `random`, `greedy`, `alphazero`)
- `--player2`: Choose player 2 type (`human`, `random`, `greedy`, `alphazero`)
- `--round`: Number of games to play
- `--ckpt_file`: Checkpoint file to load (default: `best.pth.tar`)

## ⚙️ Configuration

Edit `config.yaml` to customize training parameters:

```yaml
# Training parameters
training:
  epochs: 10
  batch_size: 256
  num_iterations: 200
  
# Neural Network parameters
network:
  num_channels: 512
  dropout: 0.1
  
# MCTS parameters
mcts:
  num_sims: 800
  cpuct: 4.0
```

## 🏗️ Architecture

- **Neural Network**: CNN with residual connections
- **MCTS**: Upper Confidence Bound for Trees (UCT)
- **Self-Play**: Generates training data through game simulation
- **Model Evaluation**: Tournament-style model comparison

## 📊 Training Process

1. **Self-Play**: Generate games using current model
2. **Training**: Update neural network on collected data  
3. **Evaluation**: Compare new model vs previous version
4. **Selection**: Keep better performing model

## 🖥️ GPU Support

The project automatically detects and uses available GPU acceleration:
- **NVIDIA GPU**: CUDA support
- **Apple Silicon**: Metal Performance Shaders (MPS)
- **CPU Fallback**: Automatic fallback if no GPU available

## 📁 Project Structure

```
├── alphazero.py      # Main training and inference script
├── game.py           # Game logic and GUI implementation
├── config.yaml       # Configuration parameters
├── temp/             # Model checkpoints
└── wandb/            # Training logs and metrics
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the AlphaZero algorithm by DeepMind
- Inspired by the original AlphaGo and AlphaZero papers
- Built with PyTorch and modern deep learning practices