#!/usr/bin/env python3
"""
Test script for 3x3 API server
"""
import requests
import json
import time

# Wait for server to be ready
time.sleep(2)

# Test 1: Check status
print("Test 1: Checking API status...")
try:
    response = requests.get('http://localhost:5003/api/status')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Status check passed\n")
except Exception as e:
    print(f"❌ Status check failed: {e}\n")

# Test 2: Get AI move with empty board (AI goes first)
print("Test 2: Empty board - AI moves first...")
try:
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    params = {
        'boardSize': 3,
        'nextMove': -1,
        'matrix': json.dumps(board)
    }
    response = requests.get('http://localhost:5003/api/move', params=params)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"AI chose: row={result.get('row')}, col={result.get('col')}")
    print("✅ Empty board test passed\n")
except Exception as e:
    print(f"❌ Empty board test failed: {e}\n")

# Test 3: Get AI move with partially filled board
print("Test 3: Partially filled board...")
try:
    # X at (0,0), O at (0,2)
    board = [[1, 0, -1], [0, 0, 0], [0, 0, 0]]
    params = {
        'boardSize': 3,
        'nextMove': -1,
        'matrix': json.dumps(board)
    }
    response = requests.get('http://localhost:5003/api/move', params=params)
    print(f"Status: {response.status_code}")
    result = response.json()
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"AI chose: row={result.get('row')}, col={result.get('col')}")
        print("✅ Partially filled board test passed\n")
except Exception as e:
    print(f"❌ Partially filled board test failed: {e}\n")

# Test 4: Board with winning opportunity
print("Test 4: Board with winning opportunity...")
try:
    # AI (O) has two in a row: [O, O, _]
    # Should block or win
    board = [[-1, -1, 0], [1, 1, 0], [0, 0, 0]]
    params = {
        'boardSize': 3,
        'nextMove': -1,
        'matrix': json.dumps(board)
    }
    response = requests.get('http://localhost:5003/api/move', params=params)
    print(f"Status: {response.status_code}")
    result = response.json()
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"AI chose: row={result.get('row')}, col={result.get('col')}")
        print("✅ Winning opportunity test passed\n")
except Exception as e:
    print(f"❌ Winning opportunity test failed: {e}\n")

print("All tests completed!")
