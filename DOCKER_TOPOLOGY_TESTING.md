# Neural Network Docker Topology Testing System

## Overview

This system provides comprehensive Docker containerization for testing Neural Network Servers in various distributed topologies. It enables researchers and developers to experiment with different network architectures and communication patterns in a controlled, scalable environment.

## Features

### 🐳 Docker Containerization
- **Multi-stage builds** for optimized container sizes
- **Specialized containers** for different neural network roles
- **Configurable environments** with environment variables
- **Persistent data** and configuration management

### 🌐 Topology Support
- **Linear Topology**: Sequential chain of neural networks
- **Star Topology**: Central hub with connected nodes
- **Mesh Topology**: Fully connected network of nodes
- **Ring Topology**: Circular connection pattern

### 🧪 Testing Framework
- **Automated data injection** with configurable patterns
- **Real-time monitoring** with web dashboard
- **Performance metrics** and statistics
- **Topology validation** and health checks

### 📊 Monitoring & Analytics
- **Web-based dashboard** for real-time visualization
- **Network topology mapping** and status tracking
- **Performance metrics** collection and analysis
- **Log aggregation** and debugging tools

## Quick Start

### 1. Build Docker Images
```bash
cd docker
./neural-topology.sh build
```

### 2. Start a Topology
```bash
# Start linear topology (3 nodes in sequence)
./neural-topology.sh start linear

# Start star topology (1 hub + 4 nodes)
./neural-topology.sh start star

# Start mesh topology (4 fully connected nodes)
./neural-topology.sh start mesh

# Start ring topology (4 nodes in circular pattern)
./neural-topology.sh start ring
```

### 3. Test the Topology
```bash
# Run default test (60 seconds, 1 Hz, random data)
./neural-topology.sh test linear

# Custom test parameters
./neural-topology.sh test star -d 120 -r 2.0 -p sine
```

### 4. Monitor in Real-time
```bash
# Start monitoring dashboard
./neural-topology.sh monitor

# Access at http://localhost:3000
```

### 5. View Logs and Status
```bash
# Check container status
./neural-topology.sh status

# View topology logs
./neural-topology.sh logs linear
```

## Architecture

### Container Types

#### 1. Neural Network Server (`neural-network-server`)
- **Purpose**: Dedicated neural network daemon
- **Features**: Configurable architecture, Hebbian learning, NNP protocol
- **Ports**: 8080 (configurable)
- **Environment Variables**:
  - `NEURAL_NODE_NAME`: Node identifier
  - `NEURAL_HOST`: Bind address (default: 0.0.0.0)
  - `NEURAL_PORT`: Service port (default: 8080)
  - `NEURAL_LAYERS`: Network architecture (e.g., "4,8,4,1")
  - `NEURAL_LEARNING_RATE`: Learning rate (default: 0.01)
  - `NEURAL_HEBBIAN_MODE`: Hebbian learning mode (Classic/AntiHebbian/Hybrid)

#### 2. Multi-purpose Container (`neural-network`)
- **Purpose**: General-purpose neural network tools
- **Includes**: All binaries (neural_network, input_server, output_server, topology_tester, topology_monitor)
- **Use Cases**: Testing, monitoring, data injection

### Network Protocol

The system uses a custom **Neural Network Protocol (NNP)** for inter-node communication:
- **Binary protocol** for efficient data transfer
- **Message types**: ForwardData, BackwardData, WeightUpdate, StatusQuery
- **Capabilities negotiation** for feature discovery
- **Connection pooling** and automatic reconnection

### Configuration Management

#### Network Configuration (`docker/config/default.toml`)
```toml
# Network architecture
architecture = [4, 8, 4, 1]
learning_rate = 0.01
hebbian_mode = "Classic"
hebbian_rate = 0.05
anti_hebbian_rate = 0.01
decay_rate = 0.001
homeostatic_rate = 0.001
target_activity = 0.5
history_size = 100
use_backprop = true
backprop_rate = 0.01
online_learning = true

[training]
batch_size = 32
print_interval = 100
early_stop_threshold = 0.001
early_stop_patience = 50
validation_split = 0.2

[distributed]
enable = true
port = 8080
security = "none"
timeout_ms = 5000
buffer_size = 1024

[server]
name = "neural-server"
address = "0.0.0.0"
hebbian_learning = true
daemon_mode = true
```

## Topology Configurations

### Linear Topology
```yaml
# 3 nodes in sequence: node-1 → node-2 → node-3
services:
  neural-node-1:
    environment:
      NEURAL_NODE_NAME: node-1
      NEURAL_OUTPUTS: neural-node-2:8080
    ports: ["8081:8080"]
  
  neural-node-2:
    environment:
      NEURAL_NODE_NAME: node-2
      NEURAL_OUTPUTS: neural-node-3:8080
    ports: ["8082:8080"]
  
  neural-node-3:
    environment:
      NEURAL_NODE_NAME: node-3
    ports: ["8083:8080"]
```

### Star Topology
```yaml
# Central hub with 4 connected nodes
services:
  neural-hub:
    environment:
      NEURAL_NODE_NAME: hub
      NEURAL_OUTPUTS: neural-node-1:8080,neural-node-2:8080,neural-node-3:8080,neural-node-4:8080
    ports: ["8080:8080"]
  
  neural-node-1:
    environment:
      NEURAL_NODE_NAME: node-1
      NEURAL_OUTPUTS: neural-hub:8080
    ports: ["8081:8080"]
  # ... additional nodes
```

### Mesh Topology
```yaml
# Fully connected 4-node mesh
services:
  neural-node-1:
    environment:
      NEURAL_NODE_NAME: node-1
      NEURAL_OUTPUTS: neural-node-2:8080,neural-node-3:8080,neural-node-4:8080
    ports: ["8081:8080"]
  # ... each node connects to all others
```

### Ring Topology
```yaml
# Circular connection pattern
services:
  neural-node-1:
    environment:
      NEURAL_NODE_NAME: node-1
      NEURAL_OUTPUTS: neural-node-2:8080
    ports: ["8081:8080"]
  
  neural-node-2:
    environment:
      NEURAL_NODE_NAME: node-2
      NEURAL_OUTPUTS: neural-node-3:8080
    ports: ["8082:8080"]
  
  neural-node-3:
    environment:
      NEURAL_NODE_NAME: node-3
      NEURAL_OUTPUTS: neural-node-4:8080
    ports: ["8083:8080"]
  
  neural-node-4:
    environment:
      NEURAL_NODE_NAME: node-4
      NEURAL_OUTPUTS: neural-node-1:8080
    ports: ["8084:8080"]
```

## Testing Tools

### Topology Tester (`topology_tester`)
- **Data Injection**: Sends test data to network entry points
- **Pattern Generation**: Random, sine wave, step function, pulse, XOR patterns
- **Rate Control**: Configurable injection rate (Hz)
- **Multi-node Support**: Distributes data across multiple entry points
- **Statistics**: Tracks message count, timing, success rate

### Topology Monitor (`topology_monitor`)
- **Web Dashboard**: Real-time visualization at http://localhost:3000
- **Node Discovery**: Automatically detects running nodes
- **Health Monitoring**: Tracks node status and connectivity
- **Performance Metrics**: Latency, throughput, error rates
- **Topology Visualization**: Interactive network diagrams

## Management Script

The `neural-topology.sh` script provides a unified interface for all operations:

```bash
# Build and deployment
./neural-topology.sh build                    # Build Docker images
./neural-topology.sh start TOPOLOGY           # Start topology
./neural-topology.sh stop TOPOLOGY            # Stop topology
./neural-topology.sh clean                    # Clean up all containers

# Testing and monitoring
./neural-topology.sh test TOPOLOGY [OPTIONS]  # Run topology tests
./neural-topology.sh monitor [OPTIONS]        # Start monitoring dashboard
./neural-topology.sh status                   # Show container status
./neural-topology.sh logs TOPOLOGY            # View topology logs

# Options
-d, --duration SECONDS     # Test duration (default: 60)
-r, --rate HZ             # Data injection rate (default: 1.0)
-p, --pattern PATTERN     # Data pattern (random, sine, step, pulse, xor)
-m, --monitor-port PORT   # Monitor web interface port (default: 3000)
-v, --verbose             # Verbose output
```

## Use Cases

### 1. Research and Development
- **Algorithm Testing**: Compare different learning algorithms across topologies
- **Scalability Studies**: Test performance with varying network sizes
- **Communication Patterns**: Analyze data flow in different topologies
- **Fault Tolerance**: Test network behavior with node failures

### 2. Education and Training
- **Distributed Systems**: Demonstrate neural network communication
- **Network Topologies**: Visualize different connection patterns
- **Performance Analysis**: Study latency and throughput characteristics
- **Protocol Design**: Understand inter-node communication protocols

### 3. Production Validation
- **Load Testing**: Validate system performance under load
- **Integration Testing**: Test neural network integration patterns
- **Deployment Validation**: Verify containerized deployments
- **Monitoring Setup**: Establish monitoring and alerting systems

## Performance Characteristics

### Container Sizes
- **neural-network-server**: 88.6MB (optimized for neural network daemon)
- **neural-network**: 110MB (includes all tools and utilities)

### Network Performance
- **Protocol Overhead**: Minimal binary protocol with efficient serialization
- **Connection Management**: Persistent connections with automatic reconnection
- **Throughput**: Tested at 1-10 Hz data injection rates
- **Latency**: Sub-millisecond inter-container communication

### Resource Usage
- **CPU**: Low baseline usage, scales with network activity
- **Memory**: ~50-100MB per container depending on network size
- **Network**: Efficient binary protocol minimizes bandwidth usage
- **Storage**: Persistent volumes for configuration and data

## Troubleshooting

### Common Issues

#### 1. Container Startup Failures
```bash
# Check logs for configuration errors
./neural-topology.sh logs TOPOLOGY

# Verify configuration file syntax
docker exec CONTAINER cat /app/config/network.toml
```

#### 2. Network Connectivity Issues
```bash
# Check container network
docker network ls
docker network inspect topologies_neural-net

# Test inter-container connectivity
docker exec neural-node-1 ping neural-node-2
```

#### 3. Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :808

# Modify port mappings in topology YAML files
```

#### 4. Performance Issues
```bash
# Monitor resource usage
docker stats

# Check network latency
./neural-topology.sh test TOPOLOGY -r 0.1 -d 10
```

### Debug Mode
Enable verbose logging for detailed troubleshooting:
```bash
./neural-topology.sh -v start linear
./neural-topology.sh -v test linear
```

## Future Enhancements

### Planned Features
- **SSL/TLS Support**: Encrypted inter-node communication
- **Authentication**: Node authentication and authorization
- **Dynamic Scaling**: Auto-scaling based on load
- **Advanced Monitoring**: Prometheus/Grafana integration
- **Cloud Deployment**: Kubernetes manifests and Helm charts
- **Performance Optimization**: GPU support and CUDA acceleration

### Extension Points
- **Custom Topologies**: Add new topology configurations
- **Protocol Extensions**: Extend NNP with new message types
- **Monitoring Plugins**: Custom metrics and visualization
- **Testing Patterns**: Additional data generation patterns
- **Integration APIs**: REST/GraphQL APIs for external integration

## Conclusion

This Docker topology testing system provides a comprehensive platform for experimenting with distributed neural networks. It combines the flexibility of containerization with powerful testing and monitoring tools, making it ideal for research, education, and production validation scenarios.

The system's modular design allows for easy extension and customization, while the unified management interface simplifies complex multi-container deployments. Whether you're studying distributed learning algorithms, validating production deployments, or teaching neural network concepts, this system provides the tools and infrastructure needed for success.