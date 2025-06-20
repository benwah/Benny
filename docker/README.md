# Neural Network Topology Testing with Docker

This directory contains a comprehensive Docker-based system for testing Neural Network Servers in various topologies. The system allows you to deploy, test, and monitor distributed neural networks running in containerized environments.

## 🏗️ Architecture Overview

The system consists of several components:

1. **Neural Network Server Containers**: Dockerized neural network servers using the existing distributed network infrastructure
2. **Topology Configurations**: Docker Compose files defining different network topologies
3. **Testing Tools**: Automated tools for injecting data and measuring performance
4. **Monitoring Dashboard**: Real-time web interface for monitoring network status
5. **Management Scripts**: Command-line tools for easy deployment and testing

## 📁 Directory Structure

```
docker/
├── README.md                    # This file
├── neural-topology.sh          # Main management script
├── Dockerfile                   # Multi-purpose neural network image
├── Dockerfile.server           # Specialized server image
├── config/
│   └── default.toml            # Default neural network configuration
└── topologies/
    ├── linear.yml              # Linear topology (A → B → C)
    ├── star.yml                # Star topology (hub with spokes)
    ├── mesh.yml                # Mesh topology (full connectivity)
    └── ring.yml                # Ring topology (circular)
```

## 🚀 Quick Start

### 1. Build Docker Images

```bash
./neural-topology.sh build
```

### 2. Start a Topology

```bash
# Start linear topology
./neural-topology.sh start linear

# Start star topology
./neural-topology.sh start star

# Start mesh topology
./neural-topology.sh start mesh

# Start ring topology
./neural-topology.sh start ring
```

### 3. Test the Topology

```bash
# Run basic test
./neural-topology.sh test linear

# Run extended test with custom parameters
./neural-topology.sh test star -d 120 -r 2.0 -p sine
```

### 4. Monitor the Network

```bash
# Start monitoring dashboard
./neural-topology.sh monitor

# Open http://localhost:3000 in your browser
```

### 5. View Status and Logs

```bash
# Show status of all containers
./neural-topology.sh status

# View logs for a specific topology
./neural-topology.sh logs linear
```

### 6. Clean Up

```bash
# Stop a specific topology
./neural-topology.sh stop linear

# Clean up everything
./neural-topology.sh clean
```

## 🌐 Available Topologies

### Linear Topology
```
node-1 → node-2 → node-3
```
- **Use Case**: Sequential processing, pipeline architectures
- **Ports**: 8081, 8082, 8083
- **Features**: Different Hebbian learning modes per node

### Star Topology
```
    node-1
       ↓
node-2 → hub ← node-4
       ↑
    node-3
```
- **Use Case**: Centralized processing, hub-and-spoke architectures
- **Ports**: 8080 (hub), 8081-8084 (nodes)
- **Features**: Central aggregation with diverse peripheral processing

### Mesh Topology
```
node-1 ↔ node-2
  ↕       ↕
node-4 ↔ node-3
```
- **Use Case**: Fault-tolerant, highly connected networks
- **Ports**: 8081-8084
- **Features**: Full connectivity between all nodes

### Ring Topology
```
node-1 → node-2
  ↑         ↓
node-4 ← node-3
```
- **Use Case**: Circular processing, token-passing architectures
- **Ports**: 8081-8084
- **Features**: Circular data flow with feedback loops

## 🧪 Testing Features

### Data Patterns

The testing system supports various data injection patterns:

- **Random**: Random values between -1 and 1
- **Sine**: Sinusoidal waves with different frequencies
- **Step**: Sequential activation of inputs
- **Pulse**: Periodic bursts of activity
- **XOR**: Classic XOR problem patterns

### Test Parameters

- **Duration**: How long to run the test (seconds)
- **Rate**: Data injection frequency (Hz)
- **Pattern**: Type of data to inject
- **Entry Points**: Which nodes to inject data into

### Example Test Commands

```bash
# Quick test with random data
./neural-topology.sh test linear -d 30

# Extended test with sine wave pattern
./neural-topology.sh test star -d 300 -r 0.5 -p sine

# High-frequency test with XOR patterns
./neural-topology.sh test mesh -d 60 -r 10.0 -p xor -v
```

## 📊 Monitoring Dashboard

The monitoring dashboard provides real-time visualization of:

- **Network Status**: Online/offline status of each node
- **Message Counts**: Number of messages processed by each node
- **Error Rates**: Network communication errors
- **Topology Visualization**: Visual representation of the network structure
- **Performance Metrics**: Throughput and latency statistics

### Dashboard Features

- **Real-time Updates**: Automatic refresh every 2 seconds
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Elements**: Click nodes for detailed information
- **Export Capabilities**: Download performance data
- **Alert System**: Notifications for node failures

## 🔧 Configuration

### Neural Network Configuration

Each container uses a TOML configuration file that defines:

```toml
[network]
architecture = [4, 8, 4, 1]      # Layer sizes
learning_rate = 0.01             # Backpropagation learning rate
hebbian_mode = "Classic"         # Hebbian learning mode
hebbian_rate = 0.05             # Hebbian learning rate

[distributed]
enable = true                    # Enable distributed networking
port = 8080                     # Default port
security = "none"               # Security mode (none, tls)
timeout_ms = 5000               # Connection timeout

[server]
name = "neural-server"          # Server name
address = "0.0.0.0"            # Bind address
hebbian_learning = true         # Enable Hebbian learning
daemon_mode = true              # Run as daemon
```

### Environment Variables

Each container can be configured using environment variables:

- `NEURAL_NODE_NAME`: Unique identifier for the node
- `NEURAL_HOST`: Host address to bind to
- `NEURAL_PORT`: Port to listen on
- `NEURAL_LAYERS`: Network architecture (comma-separated)
- `NEURAL_LEARNING_RATE`: Learning rate
- `NEURAL_HEBBIAN_MODE`: Hebbian learning mode

### Custom Topologies

To create a custom topology:

1. Create a new Docker Compose file in `topologies/`
2. Define services with appropriate networking
3. Configure environment variables for each node
4. Specify output endpoints for data flow

Example service definition:
```yaml
neural-node-1:
  build:
    context: ../..
    dockerfile: Dockerfile.server
  environment:
    - NEURAL_NODE_NAME=custom-node-1
    - NEURAL_LAYERS=4,8,2
    - NEURAL_HEBBIAN_MODE=Classic
  command: ["neural_network", "server", "-c", "/app/config/network.toml", "-p", "8080", "--outputs", "neural-node-2:8080"]
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check for port usage
   netstat -tulpn | grep :808
   
   # Stop conflicting services
   sudo systemctl stop <service>
   ```

2. **Docker Permission Issues**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   
   # Restart session or run
   newgrp docker
   ```

3. **Container Startup Failures**
   ```bash
   # Check container logs
   ./neural-topology.sh logs <topology>
   
   # Check Docker daemon logs
   sudo journalctl -u docker.service
   ```

4. **Network Connectivity Issues**
   ```bash
   # Test container networking
   docker network ls
   docker network inspect <network_name>
   
   # Test inter-container communication
   docker exec -it <container> ping <other_container>
   ```

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
# Verbose testing
./neural-topology.sh test linear -v

# Check container logs
docker logs <container_name>

# Monitor network traffic
docker exec -it <container> netstat -tulpn
```

## 📈 Performance Optimization

### Resource Allocation

Adjust Docker resource limits in compose files:

```yaml
services:
  neural-node-1:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

### Network Optimization

- Use host networking for maximum performance
- Adjust buffer sizes in configuration
- Enable connection pooling
- Use binary protocol for efficiency

### Scaling

Scale individual services:

```bash
# Scale a service to multiple instances
docker-compose -f topologies/star.yml up --scale neural-node-1=3
```

## 🔒 Security Considerations

### TLS/SSL Support

Enable TLS encryption by:

1. Generating certificates
2. Mounting certificate files
3. Configuring TLS in the neural network configuration

```yaml
volumes:
  - ./certs/server.crt:/app/certs/server.crt:ro
  - ./certs/server.key:/app/certs/server.key:ro
command: ["neural_network", "server", "--cert", "/app/certs/server.crt", "--key", "/app/certs/server.key"]
```

### Network Isolation

- Use custom Docker networks
- Implement firewall rules
- Restrict container capabilities
- Use non-root users in containers

## 🤝 Contributing

To add new features:

1. **New Topologies**: Add Docker Compose files in `topologies/`
2. **Test Patterns**: Extend the `DataPattern` enum in `topology_tester.rs`
3. **Monitoring Features**: Enhance the web dashboard
4. **Performance Tools**: Add new testing and benchmarking tools

## 📚 Additional Resources

- [Neural Network Protocol Documentation](../README.md#distributed-networks)
- [Hebbian Learning Modes](../README.md#hebbian-learning-modes)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Container Networking](https://docs.docker.com/network/)

## 🎯 Use Cases

This Docker topology testing system is ideal for:

- **Research**: Testing distributed learning algorithms
- **Development**: Validating neural network protocols
- **Education**: Demonstrating distributed AI concepts
- **Production**: Load testing before deployment
- **Experimentation**: Exploring network topologies and their effects

The system provides a complete environment for understanding how neural networks behave in distributed, containerized environments, making it an invaluable tool for both research and practical applications.