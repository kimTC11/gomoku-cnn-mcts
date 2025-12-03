#!/usr/bin/env python3
"""
Simple proxy server that serves index.html and forwards API requests to local Flask servers
Supports 3x3 ML API (port 5003), 5x5 AlphaZero API (port 5001), and 10x10 AlphaZero API (port 5002)
"""
from flask import Flask, send_from_directory, request, jsonify
import requests

app = Flask(__name__)

# Local API server URLs
ALPHAZERO_10X10_API_URL = 'http://localhost:5002'  # 10x10 AlphaZero
ALPHAZERO_5X5_API_URL = 'http://localhost:5001'    # 5x5 AlphaZero
ML3X3_API_URL = 'http://localhost:5003'            # 3x3 ML

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/3x3/<path:path>')
def proxy_ml3x3_api(path):
    """Forward 3x3 ML API requests to port 5003"""
    url = f'{ML3X3_API_URL}/api/{path}'
    
    # Forward query parameters
    if request.query_string:
        url += f'?{request.query_string.decode()}'
    
    try:
        # Forward the request to 3x3 ML API server
        response = requests.get(url, timeout=90)
        return response.json(), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to 3x3 ML API server. Make sure it is running on port 5003.',
            'hint': 'Run: cd ../3x3 && python api_server.py'
        }), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/5x5/<path:path>')
def proxy_5x5_api(path):
    """Forward 5x5 AlphaZero API requests to port 5001"""
    url = f'{ALPHAZERO_5X5_API_URL}/api/{path}'
    
    # Forward query parameters
    if request.query_string:
        url += f'?{request.query_string.decode()}'
    
    try:
        # Forward the request to 5x5 AlphaZero API server
        response = requests.get(url, timeout=90)
        return response.json(), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to 5x5 AlphaZero API server. Make sure it is running on port 5001.',
            'hint': 'Run: cd ../5x5 && python api_server.py'
        }), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<path:path>')
def proxy_api(path):
    """Forward 10x10 AlphaZero API requests to port 5002"""
    url = f'{ALPHAZERO_10X10_API_URL}/api/{path}'
    
    # Forward query parameters
    if request.query_string:
        url += f'?{request.query_string.decode()}'
    
    try:
        # Forward the request to AlphaZero API server
        response = requests.get(url, timeout=90)
        return response.json(), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to 10x10 AlphaZero API server. Make sure it is running on port 5002.',
            'hint': 'Run: cd ../10x10 && python api_server.py'
        }), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Unified Game Proxy Server Starting...")
    print("="*60)
    print("This server serves the HTML and forwards API calls")
    print("Supported game modes:")
    print("  - 3x3 Tic-Tac-Toe (ML) -> localhost:5003")
    print("  - 5x5 Gomoku (AlphaZero) -> localhost:5001")
    print("  - 10x10 Gomoku (AlphaZero) -> localhost:5002")
    print("\nLocal URL: http://localhost:8080")
    print("Use ngrok to share: ngrok http 8080")
    print("\nMake sure all API servers are running:")
    print("  Terminal 1: cd ../3x3 && python api_server.py")
    print("  Terminal 2: cd ../5x5 && python api_server.py")
    print("  Terminal 3: cd ../10x10 && python api_server.py")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
