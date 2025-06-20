# Repository Context for AI Assistants

## Overview
**Benny** is a high-performance neural network library written in Rust, featuring Hebbian learning, distributed computing, and comprehensive I/O interfaces. This repository implements a complete neural network ecosystem with real-time visualization capabilities.

## Key Architecture Components

### 1. Core Neural Network (`src/neural_network.rs`)
- **Purpose**: Main neural network implementation with 6 Hebbian learning modes
- **Key Features**: Multi-core processing, flexible architecture, serialization
- **Learning Modes**: Classic, Competitive, Oja, BCM, Anti-Hebbian, Hybrid
- **Usage**: Foundation for all neural network operations

### 2. Distributed Network (`src/distributed_network.rs`)
- **Purpose**: Network communication using custom NNP (Neural Network Protocol)
- **Protocol**: Binary protocol with 22-byte headers, CRC32 checksums
- **Features**: Peer-to-peer communication, forward/backward data propagation
- **Security**: TLS encryption support via `src/secure_network.rs`

### 3. I/O Interface System (`src/io_interface.rs`)
- **Purpose**: Connect neural networks to external systems via TCP/SSL
- **Components**: InputNode, OutputNode, IoManager
- **Protocols**: TCP, SSL/TLS with certificate authentication
- **Use Cases**: IoT integration, API processing, enterprise systems

### 4. Neural Network Server (`src/server.rs`)
- **Purpose**: Distributed neural network daemon that binds inputs to the network and outputs to other Neural Network Server instances
- **Architecture**: Uses DistributedNetwork infrastructure with NNP protocol
- **Features**: Forward data processing, Hebbian learning, peer-to-peer communication
- **Protocol**: Binary NNP protocol with 22-byte headers
- **Use Cases**: Distributed neural network topologies, multi-node processing

### 5. Input Server (`src/input_server.rs` + `src/bin/input_server.rs`)
- **Purpose**: Web-based interface for manually activating neural network inputs
- **Architecture**: HTTP server + WebSocket for real-time communication
- **Features**: Browser-based input controls, real-time feedback
- **Ports**: HTTP (default 8000), WebSocket (default 8001)

### 6. Output Server (`src/output_server.rs` + `src/bin/output_server.rs`) ⭐ **RECENTLY ADDED**
- **Purpose**: Real-time visualization of neural network outputs via web dashboard
- **Architecture**: TCP server (neural networks) + HTTP/WebSocket (web interface)
- **Features**: Live charts, activity logging, multi-network monitoring
- **Ports**: TCP (default 8002), HTTP (default 12000), WebSocket (default 12001)
- **Protocol**: JSON-based output data transmission

### 7. Docker Topology Testing System (`docker/` + topology tools) 🐳 **NEW**
- **Purpose**: Comprehensive Docker containerization for testing distributed neural network topologies
- **Architecture**: Multi-container deployments with Docker Compose orchestration
- **Topologies**: Linear (3 nodes), Star (1 hub + 4 nodes), Mesh (4 nodes), Ring (4 nodes)
- **Tools**: Automated testing, real-time monitoring, performance validation
- **Management**: Unified `neural-topology.sh` script for deployment and testing

## Project Structure

```
src/
├── lib.rs                    # Main library exports
├── main.rs                   # CLI entry point
├── cli.rs                    # Command-line interface
├── neural_network.rs         # Core neural network implementation
├── distributed_network.rs    # Distributed computing with NNP protocol
├── secure_network.rs         # TLS/SSL security layer
├── server.rs                 # Neural network server daemon
├── io_interface.rs           # External system I/O interfaces
├── input_server.rs           # Web-based input interface server
├── output_server.rs          # Web-based output visualization server ⭐ NEW
├── network_composer.rs       # Network composition utilities
├── runner.rs                 # Training and execution runner
└── bin/
    ├── input_server.rs       # Input server CLI binary
    ├── output_server.rs      # Output server CLI binary ⭐ NEW
    ├── topology_tester.rs    # Topology testing tool 🐳 NEW
    └── topology_monitor.rs   # Topology monitoring dashboard 🐳 NEW

docker/                       # Docker containerization system 🐳 NEW
├── README.md                 # Docker setup and usage guide
├── neural-topology.sh        # Unified management script
├── config/                   # Default configurations
│   └── default.toml          # Neural network configuration
└── topologies/               # Docker Compose configurations
    ├── linear.yml            # Linear topology (3 nodes)
    ├── star.yml              # Star topology (1 hub + 4 nodes)
    ├── mesh.yml              # Mesh topology (4 nodes)
    └── ring.yml              # Ring topology (4 nodes)

examples/                     # Comprehensive example collection
static/                       # Web interface assets
├── index.html                # Input server web interface
└── topology_monitor.html     # Topology monitoring dashboard 🐳 NEW

Dockerfile                    # Multi-purpose neural network container 🐳 NEW
Dockerfile.server             # Specialized neural network server container 🐳 NEW
DOCKER_TOPOLOGY_TESTING.md    # Comprehensive Docker documentation 🐳 NEW
```

## Key Binaries

### 1. Main CLI (`cargo run` or `neural_network`)
```bash
# Training
cargo run -- train -c config.toml -d data.json -o model.bin

# Neural Network Server mode (distributed daemon)
cargo run -- server -c config.toml -p 8080

# Interactive mode
cargo run -- interactive -c config.toml
```

### 2. Neural Network Server (via CLI server mode)
```bash
# Start distributed neural network server daemon
cargo run -- server -c config.toml -p 8080 --daemon

# Example: Start server with specific model
cargo run -- server -c config.toml -m model.bin -p 8080
```

### 3. Input Server (`cargo run --bin input_server`)
```bash
# Start input server for manual neural network input control
cargo run --bin input_server --listen-port 8000 --web-port 8001
```

### 4. Output Server (`cargo run --bin output_server`) ⭐ **NEW**
```bash
# Start output server for real-time neural network output visualization
cargo run --bin output_server --listen-port 8002 --web-port 12000 --websocket-port 12001
```

### 5. Topology Tester (`cargo run --bin topology_tester`) 🐳 **NEW**
```bash
# Test distributed neural network topologies with automated data injection
cargo run --bin topology_tester --topology linear --duration 60 --rate 1.0
```

### 6. Topology Monitor (`cargo run --bin topology_monitor`) 🐳 **NEW**
```bash
# Start web-based monitoring dashboard for topology visualization
cargo run --bin topology_monitor --port 3000
```

### 7. Docker Topology Management 🐳 **NEW**
```bash
# Build Docker images
cd docker && ./neural-topology.sh build

# Start topology
./neural-topology.sh start linear

# Test topology
./neural-topology.sh test star -d 120 -r 2.0

# Monitor topology
./neural-topology.sh monitor
```

## Common Development Tasks

### Building and Testing
```bash
# Build everything
cargo build --release

# Run tests
cargo test

# Run specific example
cargo run --example simple_example

# Test neural network server
cargo run --example neural_network_server

# Test output server with sample data
cargo run --example simple_output_test

# Test Docker topology system
cd docker && ./neural-topology.sh build && ./neural-topology.sh start linear
```

### Working with Neural Networks
```rust
use neural_network::{NeuralNetwork, HebbianLearningMode};

// Create network
let mut net = NeuralNetwork::with_layers_and_mode(
    &[2, 4, 1], 
    0.1, 
    HebbianLearningMode::Classic
);

// Train
net.train(&[1.0, 0.0], &[1.0]);

// Predict
let output = net.predict(&[1.0, 0.0]);
```

### Working with Distributed Networks
```rust
use neural_network::DistributedNetwork;

// Create distributed node
let (dist_net, receiver) = DistributedNetwork::new(
    "node1".to_string(),
    "0.0.0.0".to_string(),
    8080,
    neural_network,
);

// Start server and connect to peers
dist_net.start_server().await?;
let peer_id = dist_net.connect_to("192.168.1.100", 8080).await?;
```

### Working with I/O Interfaces
```rust
use neural_network::{IoManager, TcpInputInterface, TcpOutputInterface};

// Create I/O manager
let mut manager = IoManager::new(neural_network);

// Add interfaces
manager.add_input_interface(input_id, Box::new(input_interface));
manager.add_output_interface(output_id, Box::new(output_interface));

// Start processing
manager.start_processing().await?;
```

### Working with Neural Network Server
```rust
use neural_network::{DistributedNetwork, NeuralNetwork, ServerConfig};

// Create neural network
let network = NeuralNetwork::with_layers(&[4, 2, 1], 0.01);

// Create server configuration
let config = ServerConfig {
    name: "NeuralNetworkServer".to_string(),
    address: "127.0.0.1".to_string(),
    port: 8080,
    cert_path: None,
    key_path: None,
    output_endpoints: vec!["127.0.0.1:8081".to_string()],
    hebbian_learning: true,
    daemon_mode: false,
};

// Create and start server
let server = NetworkServer::new(network, config)?;
server.start().await?;
```

### Working with Output Server ⭐ **NEW**
```rust
// Neural network side - send outputs to OutputServer
use std::net::TcpStream;
use std::io::Write;

let mut stream = TcpStream::connect("127.0.0.1:8002")?;
let outputs = vec![0.5, 0.8]; // Your neural network outputs
let json_data = serde_json::to_string(&outputs)?;
stream.write_all(json_data.as_bytes())?;
```

### Working with Docker Topology Testing 🐳 **NEW**
```bash
# Quick start with linear topology
cd docker
./neural-topology.sh build
./neural-topology.sh start linear
./neural-topology.sh test linear

# Advanced testing with custom parameters
./neural-topology.sh test star -d 120 -r 2.0 -p sine

# Real-time monitoring
./neural-topology.sh monitor

# View logs and status
./neural-topology.sh status
./neural-topology.sh logs linear

# Clean up
./neural-topology.sh stop linear
./neural-topology.sh clean
```

## Configuration Files

### Network Configuration (`config.toml`)
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

### Training Data (`data.json`)
```json
{
  "inputs": [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ],
  "targets": [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
  ]
}
```

### Docker Compose Configuration (`docker/topologies/linear.yml`) 🐳 **NEW**
```yaml
services:
  neural-node-1:
    build:
      context: ../..
      dockerfile: Dockerfile.server
    environment:
      NEURAL_NODE_NAME: node-1
      NEURAL_OUTPUTS: neural-node-2:8080
    ports: ["8081:8080"]
    volumes: ["../config:/app/config:ro"]
    networks: [neural-net]

  neural-node-2:
    build:
      context: ../..
      dockerfile: Dockerfile.server
    environment:
      NEURAL_NODE_NAME: node-2
      NEURAL_OUTPUTS: neural-node-3:8080
    ports: ["8082:8080"]
    volumes: ["../config:/app/config:ro"]
    networks: [neural-net]

  neural-node-3:
    build:
      context: ../..
      dockerfile: Dockerfile.server
    environment:
      NEURAL_NODE_NAME: node-3
    ports: ["8083:8080"]
    volumes: ["../config:/app/config:ro"]
    networks: [neural-net]

networks:
  neural-net:
    driver: bridge
```

## Important Dependencies

- **tokio**: Async runtime for network operations
- **rayon**: Parallel processing for neural network computations
- **serde/serde_json**: Serialization for data exchange
- **clap**: Command-line interface
- **hyper/warp**: HTTP servers for web interfaces
- **tokio-tungstenite**: WebSocket support
- **tokio-rustls**: TLS/SSL security
- **uuid**: Unique identifiers for distributed nodes

## Development Guidelines

### When Adding New Features
1. **Core Neural Network**: Modify `src/neural_network.rs`
2. **Distributed Features**: Work with `src/distributed_network.rs`
3. **I/O Integration**: Extend `src/io_interface.rs`
4. **Web Interfaces**: Update server modules and `static/` assets
5. **CLI Commands**: Add to `src/cli.rs`

### Testing Strategy
- **Unit Tests**: Test individual components
- **Integration Tests**: Test server communication
- **Examples**: Create comprehensive examples for new features
- **Performance**: Use `examples/benchmark_parallel.rs` as template

### Common Patterns
- **Async/Await**: All network operations are async
- **Error Handling**: Use `Result<T, Box<dyn std::error::Error>>`
- **Configuration**: Use TOML for complex configs, CLI args for simple options
- **Serialization**: JSON for external APIs, bincode for internal protocols

## Recent Major Additions

### Output Server ⭐
The Output Server is a recently implemented system that provides real-time visualization of neural network outputs through a web dashboard.

### Docker Topology Testing System 🐳 **NEW**
A comprehensive Docker containerization system for testing distributed neural network topologies with automated deployment, testing, and monitoring capabilities.

#### Docker System Architecture
```
Management Script → Docker Compose → Neural Network Containers → NNP Protocol
       ↓                ↓                      ↓                    ↓
neural-topology.sh → topology.yml → neural-network-server → TCP:8080-8084
       ↓                ↓                      ↓                    ↓
   Build/Test → Container Orchestration → Distributed Topology → Inter-node Comm
       ↓                ↓                      ↓                    ↓
   Monitor → Web Dashboard → Real-time Metrics → Performance Data
```

#### Output Server Architecture
```
Neural Network → TCP:8002 → OutputServer → WebSocket:12001 → Browser
                                ↓
                           HTTP:12000 → Web Dashboard
```

#### Key Files - Output Server
- `src/output_server.rs`: Main server implementation
- `src/bin/output_server.rs`: CLI binary
- `examples/simple_output_test.rs`: Test client
- `examples/neural_network_with_output.rs`: Integration example

#### Key Files - Docker Topology System 🐳 **NEW**
- `docker/neural-topology.sh`: Unified management script
- `docker/topologies/*.yml`: Docker Compose configurations for each topology
- `docker/config/default.toml`: Default neural network configuration
- `src/bin/topology_tester.rs`: Automated testing tool
- `src/bin/topology_monitor.rs`: Real-time monitoring dashboard
- `static/topology_monitor.html`: Web interface for monitoring
- `Dockerfile`: Multi-purpose neural network container
- `Dockerfile.server`: Specialized neural network server container

#### Usage - Output Server
```bash
# Start server
cargo run --bin output_server

# Send test data
cargo run --example simple_output_test

# View dashboard at http://localhost:12000
```

#### Usage - Docker Topology System 🐳 **NEW**
```bash
# Quick start
cd docker && ./neural-topology.sh build
./neural-topology.sh start linear
./neural-topology.sh test linear

# Advanced usage
./neural-topology.sh test star -d 120 -r 2.0 -p sine
./neural-topology.sh monitor
./neural-topology.sh status
./neural-topology.sh logs linear
./neural-topology.sh clean
```

#### Integration - Output Server
Neural networks send JSON output arrays to the TCP port:
```rust
let outputs = vec![0.5, 0.8];
let json_data = serde_json::to_string(&outputs)?;
stream.write_all(json_data.as_bytes()).await?;
```

#### Integration - Docker Topology System 🐳 **NEW**
Neural networks communicate via NNP protocol in containerized environments:
```rust
// Containers automatically connect based on NEURAL_OUTPUTS environment variable
// Example: NEURAL_OUTPUTS=neural-node-2:8080,neural-node-3:8080
```

#### Web Interface Features
**Output Server:**
- **Real-time Charts**: Live visualization of neural network outputs
- **Network Status**: Connection monitoring and health indicators
- **Activity Log**: Timestamped log of all received data
- **Multi-network Support**: Monitor multiple neural networks simultaneously
- **Responsive Design**: Clean, modern web interface

**Docker Topology Monitor:** 🐳 **NEW**
- **Topology Visualization**: Interactive network diagrams
- **Node Status**: Real-time health monitoring of all containers
- **Performance Metrics**: Latency, throughput, and error rates
- **Log Aggregation**: Centralized logging from all nodes
- **Test Results**: Automated testing results and statistics

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Check if ports 8000-8002, 8080-8084, 12000-12001 are available
2. **TLS Certificates**: Ensure valid certificates for secure connections
3. **Network Connectivity**: Verify firewall settings for distributed networks
4. **Dependencies**: Run `cargo update` if build fails
5. **Docker Issues** 🐳: Ensure Docker daemon is running and user has permissions
6. **Container Startup** 🐳: Check configuration files and environment variables

### Debug Mode
```bash
# Enable logging
RUST_LOG=debug cargo run --bin output_server

# Verbose mode
cargo run -- --verbose train -c config.toml

# Docker debugging 🐳
./neural-topology.sh -v start linear
./neural-topology.sh logs linear
docker logs neural-node-1
```

### Performance Tuning
- **Parallel Processing**: Adjust `RAYON_NUM_THREADS` environment variable
- **Batch Size**: Tune batch_size in configuration
- **Network Topology**: Optimize layer sizes for your use case
- **Docker Resources** 🐳: Allocate sufficient CPU/memory to containers
- **Container Networking** 🐳: Use bridge networks for optimal performance

## AI Assistant Guidelines

### When Working with This Repository
1. **Understand the Context**: This is a neural network library with distributed computing and web interfaces
2. **Check Existing Examples**: Look in `examples/` for similar functionality before implementing
3. **Follow Rust Conventions**: Use proper error handling, async/await patterns
4. **Test Thoroughly**: Always test network communication and web interfaces
5. **Update Documentation**: Keep README.md and this repo.md current

### Common Tasks You Might Be Asked
- **Add New Learning Algorithms**: Extend `HebbianLearningMode` enum
- **Implement New I/O Interfaces**: Create new interface types in `io_interface.rs`
- **Add Web Features**: Extend input/output servers with new capabilities
- **Optimize Performance**: Improve parallel processing or network protocols
- **Add Security Features**: Enhance TLS/SSL implementation
- **Create Examples**: Demonstrate new functionality with comprehensive examples
- **Docker Enhancements** 🐳: Add new topologies, improve containerization, enhance monitoring
- **Topology Testing** 🐳: Create new test patterns, improve validation, add performance benchmarks

### Key Concepts to Remember
- **Hebbian Learning**: Unsupervised learning based on correlation ("neurons that fire together, wire together")
- **NNP Protocol**: Custom binary protocol for neural network communication
- **Distributed Computing**: Networks can span multiple machines
- **Real-time Visualization**: Web interfaces for monitoring and control
- **I/O Integration**: Connect to external systems via TCP/SSL
- **Docker Containerization** 🐳: Scalable deployment with container orchestration
- **Topology Testing** 🐳: Automated validation of distributed network architectures

### Working with the Server Systems
- **Neural Network Server**: Distributed daemon for peer-to-peer neural network communication
  - **Protocol**: Binary NNP protocol with forward/backward data propagation
  - **Use Case**: Building distributed neural network topologies
  - **Testing**: Use `examples/neural_network_server.rs` for testing
  - **Docker** 🐳: Containerized deployment with `Dockerfile.server`
  
- **Input Server**: Web-based manual input control
  - **Protocol**: HTTP/WebSocket for browser communication
  - **Use Case**: Manual testing and interactive neural network control
  - **Testing**: Use `cargo run --bin input_server` and visit web interface
  
- **Output Server**: Real-time visualization of neural network outputs
  - **Protocol**: Simple JSON arrays sent via TCP
  - **Web Interface**: Modern dashboard with live charts and logging
  - **Testing**: Use `examples/simple_output_test.rs` for testing
  - **Integration**: Easy to integrate with existing neural networks

- **Docker Topology System** 🐳: Comprehensive containerization for distributed testing
  - **Management**: Unified `neural-topology.sh` script for all operations
  - **Topologies**: Linear, Star, Mesh, Ring configurations
  - **Testing**: Automated data injection with `topology_tester`
  - **Monitoring**: Real-time dashboard with `topology_monitor`
  - **Use Case**: Research, validation, and production testing of distributed neural networks

This repository represents a complete neural network ecosystem with advanced features for research, development, and production deployment. Recent additions include the Output Server for real-time visualization and a comprehensive Docker containerization system for testing distributed neural network topologies at scale.