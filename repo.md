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
- **Purpose**: Daemon process that binds neural network inputs/outputs to other Neural Network Server instances
- **Architecture**: Uses DistributedNetwork infrastructure with NNP protocol
- **Features**: Network-to-network communication, distributed neural processing
- **Protocol**: Binary NNP protocol for efficient neural data exchange
- **Use Cases**: Multi-node neural networks, distributed AI processing

### 5. Input Server (`src/input_server.rs` + `src/bin/input_server.rs`)
- **Purpose**: Web-based interface for manually activating neural network inputs
- **Architecture**: HTTP server + WebSocket for real-time communication
- **Features**: Browser-based input controls, real-time feedback
- **Ports**: HTTP (default 8000), WebSocket (default 8001)

### 6. Output Server (`src/output_server.rs` + `src/bin/output_server.rs`) ⭐ **NEWLY ADDED**
- **Purpose**: Real-time visualization of neural network outputs via web dashboard
- **Architecture**: TCP server (neural networks) + HTTP/WebSocket (web interface)
- **Features**: Live charts, activity logging, multi-network monitoring
- **Ports**: TCP (default 8002), HTTP (default 12000), WebSocket (default 12001)
- **Protocol**: JSON-based output data transmission

## Project Structure

```
src/
├── lib.rs                    # Main library exports
├── main.rs                   # CLI entry point
├── cli.rs                    # Command-line interface
├── neural_network.rs         # Core neural network implementation
├── distributed_network.rs    # Distributed computing with NNP protocol
├── secure_network.rs         # TLS/SSL security layer
├── io_interface.rs           # External system I/O interfaces
├── input_server.rs           # Web-based input interface server
├── output_server.rs          # Web-based output visualization server ⭐ NEW
├── network_composer.rs       # Network composition utilities
├── runner.rs                 # Training and execution runner
└── bin/
    ├── input_server.rs       # Input server CLI binary
    └── output_server.rs      # Output server CLI binary ⭐ NEW

examples/                     # Comprehensive example collection
static/                       # Web interface assets
```

## Key Binaries

### 1. Main CLI (`cargo run` or `neural_network`)
```bash
# Training
cargo run -- train -c config.toml -d data.json -o model.bin

# Server mode (Neural Network Server daemon)
cargo run -- server -m model.bin -p 8080

# Interactive mode
cargo run -- interactive -c config.toml
```

### 2. Neural Network Server (via main CLI or examples)
```bash
# Start neural network server daemon (distributed neural processing)
cargo run -- server -m model.bin -p 8001 --daemon

# Or run example neural network server
cargo run --example neural_network_server
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

## Three-Server Architecture

This repository implements a comprehensive three-server architecture for neural network operations:

### 1. Neural Network Server (Core Processing)
- **Purpose**: Distributed neural network processing daemon
- **Communication**: Neural Network Protocol (NNP) - binary protocol
- **Connections**: Network-to-network (neural data exchange)
- **Example**: `cargo run --example neural_network_server`
- **Use Case**: Multi-node neural networks, distributed AI processing

### 2. Input Server (Manual Control)
- **Purpose**: Web-based manual input control for neural networks
- **Communication**: HTTP + WebSocket for browser interface
- **Connections**: Browser → Neural Network inputs
- **Binary**: `cargo run --bin input_server`
- **Use Case**: Manual testing, debugging, interactive control

### 3. Output Server (Visualization)
- **Purpose**: Real-time visualization of neural network outputs
- **Communication**: TCP (JSON) + HTTP/WebSocket for browser
- **Connections**: Neural Network outputs → Browser dashboard
- **Binary**: `cargo run --bin output_server`
- **Use Case**: Monitoring, debugging, real-time visualization

### Complete Workflow Example
```bash
# Terminal 1: Start neural network server (processing)
cargo run --example neural_network_server

# Terminal 2: Start input server (manual control)
cargo run --bin input_server

# Terminal 3: Start output server (visualization)
cargo run --bin output_server

# Browser 1: http://localhost:8000 (input control)
# Browser 2: http://localhost:12000 (output visualization)
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

# Test output server with sample data
cargo run --example simple_output_test
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

## Configuration Files

### Network Configuration (`config.toml`)
```toml
[network]
type = "feedforward"
layers = [2, 4, 1]
learning_rate = 0.1
hebbian_rate = 0.05
mode = "Classic"

[training]
epochs = 1000
batch_size = 32

[distributed]
enable = true
port = 8080
security = "tls"
```

### Training Data (`data.json`)
```json
{
  "samples": [
    {"inputs": [0.0, 0.0], "targets": [0.0]},
    {"inputs": [0.0, 1.0], "targets": [1.0]}
  ]
}
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

## Recent Major Addition: Output Server ⭐

The Output Server is a newly implemented system that provides real-time visualization of neural network outputs through a web dashboard.

### Architecture
```
Neural Network → TCP:8002 → OutputServer → WebSocket:12001 → Browser
                                ↓
                           HTTP:12000 → Web Dashboard
```

### Key Files
- `src/output_server.rs`: Main server implementation
- `src/bin/output_server.rs`: CLI binary
- `examples/simple_output_test.rs`: Test client

### Usage
```bash
# Start server
cargo run --bin output_server

# Send test data
cargo run --example simple_output_test

# View dashboard at http://localhost:12000
```

### Integration
Neural networks send JSON output arrays to the TCP port:
```rust
let outputs = vec![0.5, 0.8];
let json_data = serde_json::to_string(&outputs)?;
stream.write_all(json_data.as_bytes()).await?;
```

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Check if ports 8000-8002, 12000-12001 are available
2. **TLS Certificates**: Ensure valid certificates for secure connections
3. **Network Connectivity**: Verify firewall settings for distributed networks
4. **Dependencies**: Run `cargo update` if build fails

### Debug Mode
```bash
# Enable logging
RUST_LOG=debug cargo run --bin output_server

# Verbose mode
cargo run -- --verbose train -c config.toml
```

### Performance Tuning
- **Parallel Processing**: Adjust `RAYON_NUM_THREADS` environment variable
- **Batch Size**: Tune batch_size in configuration
- **Network Topology**: Optimize layer sizes for your use case

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

### Key Concepts to Remember
- **Hebbian Learning**: Unsupervised learning based on correlation
- **NNP Protocol**: Custom binary protocol for neural network communication
- **Distributed Computing**: Networks can span multiple machines
- **Real-time Visualization**: Web interfaces for monitoring and control
- **I/O Integration**: Connect to external systems via TCP/SSL

This repository represents a complete neural network ecosystem with advanced features for research, development, and production deployment.