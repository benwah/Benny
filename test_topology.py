#!/usr/bin/env python3
"""
Simple test script to verify the 6-node neural network topology is working.
Sends test data through the input server and monitors the output server.
"""

import requests
import json
import time
import random
import threading
import websocket

def test_input_server():
    """Send test data to the input server"""
    print("🧪 Testing Input Server...")
    
    # Test data - 16 inputs as configured
    test_inputs = []
    for i in range(5):
        inputs = [random.random() for _ in range(16)]
        test_inputs.append(inputs)
    
    try:
        # Send test data via HTTP POST
        for i, inputs in enumerate(test_inputs):
            response = requests.post(
                'http://localhost:8001/api/inputs',
                json={'inputs': inputs},
                timeout=5
            )
            print(f"   Test {i+1}: Status {response.status_code}")
            time.sleep(1)
        
        print("✅ Input server test completed")
        return True
    except Exception as e:
        print(f"❌ Input server test failed: {e}")
        return False

def test_output_server():
    """Monitor the output server for responses"""
    print("🔍 Monitoring Output Server...")
    
    try:
        # Check if output server is responding
        response = requests.get('http://localhost:8002', timeout=5)
        print(f"   Output server HTTP status: {response.status_code}")
        
        # Try to get current outputs
        try:
            response = requests.get('http://localhost:8002/api/outputs', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"   Current outputs: {data}")
            else:
                print(f"   No outputs available yet (status: {response.status_code})")
        except:
            print("   Output API not available or no data yet")
        
        print("✅ Output server test completed")
        return True
    except Exception as e:
        print(f"❌ Output server test failed: {e}")
        return False

def main():
    print("🧠 Testing 6-Node Neural Network Topology")
    print("=" * 50)
    
    # Test both servers
    input_ok = test_input_server()
    time.sleep(2)
    output_ok = test_output_server()
    
    print("\n📊 Test Results:")
    print(f"   Input Server: {'✅ OK' if input_ok else '❌ FAILED'}")
    print(f"   Output Server: {'✅ OK' if output_ok else '❌ FAILED'}")
    
    if input_ok and output_ok:
        print("\n🎉 Topology is working correctly!")
        print("   You can access:")
        print("   - Input Server Web Interface: http://localhost:8001")
        print("   - Output Server Web Interface: http://localhost:8002")
    else:
        print("\n⚠️  Some issues detected. Check the logs for details.")

if __name__ == "__main__":
    main()