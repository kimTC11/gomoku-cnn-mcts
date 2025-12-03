"""
Tic-Tac-Toe Game with ML Models
Load models đã train từ notebook
"""

import streamlit as st
import numpy as np
import pandas as pd
import random
import pickle
import os

st.set_page_config(page_title="🎮 Tic-Tac-Toe AI", page_icon="🎯", layout="centered")

# Functions for loading models
def load_models():
    """Load pre-trained models"""
    models = {
        'neural_model': None,
        'xgboost_model': None,
        'catboost_encoder': None,
        'label_encoder': None
    }
    
    # Load encoders first (these work fine)
    try:
        if os.path.exists('models/catboost_encoder.pkl'):
            with open('models/catboost_encoder.pkl', 'rb') as f:
                models['catboost_encoder'] = pickle.load(f)
            st.success("✅ CatBoost encoder loaded successfully!")
        
        if os.path.exists('models/xgboost_model.pkl'):
            with open('models/xgboost_model.pkl', 'rb') as f:
                models['xgboost_model'] = pickle.load(f)
            st.success("✅ XGBoost model loaded successfully!")
                
    except Exception as e:
        st.warning(f"⚠️ Unable to load XGBoost/CatBoost: {e}")
    
    # Try to load Simple Neural Network (TensorFlow-free)
    try:
        if os.path.exists('models/simple_neural_predictor.pkl'):
            with open('models/simple_neural_predictor.pkl', 'rb') as f:
                models['neural_model'] = pickle.load(f)
            st.success("✅ Simple Neural Network loaded! (TensorFlow-free)")
        elif os.path.exists('models/neural_network_model.h5'):
            st.warning("⚠️ Found .h5 model but not yet converted")
            st.info("💡 Run 'python create_simple_nn.py' to convert model")
                    
    except Exception as e:
        st.warning(f"⚠️ Unable to load Neural Network: {e}")
        st.info("🎲 Using XGBoost as Neural Network replacement")
    
    return models

def board_to_string_format(board):
    """Convert board to string format for model prediction"""
    string_board = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 1:  # X
                string_board.append('x')
            elif board[i][j] == 0:  # O
                string_board.append('o')
            else:  # Empty
                string_board.append('b')
    return string_board

def check_winner(board):
    """Kiểm tra người thắng"""
    # Kiểm tra hàng ngang
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != -1:
            return board[i][0]
    
    # Kiểm tra hàng dọc
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] != -1:
            return board[0][j]
    
    # Kiểm tra đường chéo
    if board[0][0] == board[1][1] == board[2][2] != -1:
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] != -1:
        return board[0][2]
    
    # Kiểm tra hòa
    flat = [cell for row in board for cell in row]
    if -1 not in flat:
        return "Draw"
    
    return None

def get_available_moves(board):
    """Lấy các nước đi có thể"""
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == -1:
                moves.append((i, j))
    return moves

def ai_move_simple(board):
    """Simple rule-based AI - AI plays X"""
    available = get_available_moves(board)
    
    if not available:
        return None
    
    # 1. Try to win (AI plays X = 1)
    for move in available:
        test_board = [row[:] for row in board]
        test_board[move[0]][move[1]] = 1
        if check_winner(test_board) == 1:
            return move
    
    # 2. Block player from winning (Player plays O = 0)
    for move in available:
        test_board = [row[:] for row in board]
        test_board[move[0]][move[1]] = 0
        if check_winner(test_board) == 0:
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

def ai_move_ml(board, models):
    """AI using ML models - AI plays X (goes first) as trained in dataset"""
    available = get_available_moves(board)
    
    if not available:
        return None
    
    neural_model = models['neural_model']
    xgboost_model = models['xgboost_model'] 
    catboost_encoder = models['catboost_encoder']
    
    # Need at least catboost encoder and one model
    if not catboost_encoder or (not neural_model and not xgboost_model):
        return ai_move_simple(board)
    
    try:
        best_move = None
        best_score = -1
        
        for move in available:
            # Try the move
            test_board = [row[:] for row in board]
            test_board[move[0]][move[1]] = 1  # AI plays as X (như dataset training)
            
            # Convert to string format for prediction
            string_board = board_to_string_format(test_board)
            
            # Create DataFrame for prediction
            columns = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']
            board_df = pd.DataFrame([string_board], columns=columns)
            board_encoded = catboost_encoder.transform(board_df)
            
            # Get predictions from available models
            predictions = []
            
            if neural_model:
                try:
                    nn_prediction = neural_model.predict(board_encoded)
                    nn_prob = nn_prediction[0][0]
                    predictions.append(nn_prob)
                except:
                    pass  # Skip if Neural Network fails
            
            if xgboost_model:
                try:
                    xgb_prob = xgboost_model.predict_proba(board_encoded)[0][1]
                    predictions.append(xgb_prob)
                except:
                    pass  # Skip if XGBoost fails
            
            # Average predictions if available
            if predictions:
                avg_prob = sum(predictions) / len(predictions)
                # AI chơi X, muốn MAXIMIZE X winning probability
                score = avg_prob
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move if best_move else ai_move_simple(board)
        
    except Exception as e:
        st.error(f"ML prediction error: {e}")
        return ai_move_simple(board)

def play_move(row, col):
    """Handle move - User plays O (0), AI plays X (1)"""
    if st.session_state.game_over:
        return
    
    if st.session_state.board[row][col] != -1:
        return
    
    # User plays O (0) 
    st.session_state.board[row][col] = 0
    
    # Check if User wins
    winner = check_winner(st.session_state.board)
    if winner is not None:
        st.session_state.winner = winner
        st.session_state.game_over = True
        return
    
    # Check draw
    if all(cell != -1 for row in st.session_state.board for cell in row):
        st.session_state.game_over = True
        st.session_state.winner = -1  # Draw
        return
    
    # AI plays X (1)
    if st.session_state.ai_mode == "ml" and st.session_state.models:
        ai_move_pos = ai_move_ml(st.session_state.board, st.session_state.models)
    else:
        ai_move_pos = ai_move_simple(st.session_state.board)
    
    if ai_move_pos:
        st.session_state.board[ai_move_pos[0]][ai_move_pos[1]] = 1
        
        # Check if AI wins
        winner = check_winner(st.session_state.board)
        if winner is not None:
            st.session_state.winner = winner
            st.session_state.game_over = True

def reset_game():
    """Reset game - AI goes first (X), User goes second (O)"""
    st.session_state.board = [[-1, -1, -1] for _ in range(3)]
    st.session_state.winner = None
    st.session_state.game_over = False
    
    # AI đi trước ngay lập tức (X)
    if 'ai_mode' in st.session_state and st.session_state.ai_mode == "ml" and 'models' in st.session_state:
        ai_move = ai_move_ml(st.session_state.board, st.session_state.models)
    else:
        ai_move = ai_move_simple(st.session_state.board)
    
    if ai_move:
        st.session_state.board[ai_move[0]][ai_move[1]] = 1  # AI plays X

def main():
    st.title("🎮 Intelligent Tic-Tac-Toe AI")
    
    # Load models
    if 'models' not in st.session_state:
        with st.spinner("🤖 Loading AI models..."):
            st.session_state.models = load_models()
    
    # AI Mode selector
    models = st.session_state.models
    has_ml_models = models['neural_model'] is not None
    
    if has_ml_models:
        ai_options = ["🧠 Neural Network AI", "🎲 Simple AI"]
        selected = st.radio("Chọn loại AI:", ai_options, horizontal=True)
        st.session_state.ai_mode = "ML" if "Neural" in selected else "Simple"
    else:
        st.info("🎲 Chỉ có Simple AI (không tìm thấy ML models)")
        st.session_state.ai_mode = "Simple"
    
    # Display current AI type
    if st.session_state.ai_mode == "ML":
        st.success("🧠 Đang sử dụng: Neural Network + XGBoost AI")
    else:
        st.info("🎲 Đang sử dụng: Rule-based AI")
    
    st.markdown("### Bạn là ❌, AI là ⭕")
    
    # Khởi tạo game
    if 'board' not in st.session_state:
        reset_game()
    
    # CSS
    st.markdown("""
    <style>
    div[data-testid="column"] > div > div > button {
        height: 80px !important;
        width: 80px !important;
        font-size: 2rem !important;
        border: 2px solid #1976d2 !important;
        border-radius: 8px !important;
        margin: 2px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Bảng game
    st.markdown("---")
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            with cols[j]:
                cell = st.session_state.board[i][j]
                
                if cell == 1:  # X
                    label = "❌"
                    disabled = True
                elif cell == 0:  # O
                    label = "⭕"
                    disabled = True
                else:  # Trống
                    label = " "
                    disabled = st.session_state.game_over
                
                if st.button(label, key=f"btn_{i}_{j}", disabled=disabled):
                    play_move(i, j)
                    st.rerun()
    
    st.markdown("---")
    
    # Trạng thái game
    if st.session_state.game_over:
        if st.session_state.winner == 0:
            st.success("🎉 Bạn thắng! (O)")
        elif st.session_state.winner == 1:
            st.error("🤖 AI thắng! (X) - Trained model!")
        else:
            st.info("🤝 Hòa!")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Chơi lại"):
            reset_game()
            st.rerun()
    
    with col2:
        if st.button("📊 Models Info"):
            st.balloons()
            st.write("**Models hiện có:**")
            for name, model in st.session_state.models.items():
                status = "✅" if model is not None else "❌"
                st.write(f"{status} {name}")
    
    # Thống kê
    st.markdown("### 📊 Thống kê")
    flat_board = [cell for row in st.session_state.board for cell in row]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("❌ X", flat_board.count(1))
    with col2:
        st.metric("⭕ O", flat_board.count(0))
    with col3:
        st.metric("⬜ Trống", flat_board.count(-1))

if __name__ == "__main__":
    main()