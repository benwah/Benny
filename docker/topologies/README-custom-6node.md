# Custom 6-Node Neural Network Topology

## Overview
This topology creates a linear chain of 6 neural networks with input and output servers for complete end-to-end testing.

## Architecture
```
Input Server -> nn0 -> nn1 -> nn2 -> nn3 -> nn4 -> nn5 -> Output Server
```

## Network Specifications

### Neural Networks
- **nn0**: 16 inputs, 12->8->4 hidden layers (receives from Input Server)
- **nn1**: 4 inputs, 8->6->4 hidden layers
- **nn2**: 4 inputs, 6->8->6 hidden layers  
- **nn3**: 6 inputs, 8->10->8 hidden layers
- **nn4**: 8 inputs, 10->12->10 hidden layers
- **nn5**: 10 inputs, 12->10->8 outputs (sends to Output Server)

### Servers
- **Input Server**: Web interface for manual input control (16 inputs)
- **Output Server**: Real-time visualization dashboard (8 outputs)

## Port Mappings
- **Input Server Web Interface**: 8001
- **Output Server Web Interface**: 8002
- **Neural Networks**: 8080-8085 (nn0-nn5)
- **WebSocket Ports**: 3001 (input), 12001 (output)

## Usage

### Start the topology:
```bash
cd docker
./neural-topology.sh start custom-6node
```

### Access interfaces:
- Input Server: http://localhost:8001
- Output Server: http://localhost:8002

### Stop the topology:
```bash
./neural-topology.sh stop custom-6node
```

### View logs:
```bash
./neural-topology.sh logs custom-6node
```

## Learning Modes
Each neural network uses a different Hebbian learning mode:
- nn0: Classic Hebbian
- nn1: Competitive Hebbian
- nn2: Oja's Rule
- nn3: BCM (Bienenstock-Cooper-Munro)
- nn4: Anti-Hebbian
- nn5: Hybrid mode

## Data Flow
1. User inputs data via web interface (Input Server)
2. Data flows through the neural network chain (nn0 -> nn1 -> nn2 -> nn3 -> nn4 -> nn5)
3. Final outputs are visualized in real-time (Output Server)
4. Each network applies its own learning algorithm and forwards processed data

This topology is ideal for testing complex distributed neural network behaviors and observing how different learning algorithms affect data transformation through the network chain.