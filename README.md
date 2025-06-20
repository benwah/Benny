# Benny Neural Network

A high-performance neural network library in Rust featuring Hebbian learning, distributed computing, and comprehensive CLI tools.

## Features

- **6 Hebbian Learning Modes**: Classic, Competitive, Oja, BCM, Anti-Hebbian, Hybrid
- **Multi-core Processing**: Parallel training and inference using Rayon
- **Distributed Networks**: Custom binary protocol (NNP) for network communication
- **I/O Interfaces**: SSL/TCP interfaces for connecting external systems to neural networks
- **Security**: TLS encryption with certificate-based authentication
- **CLI Interface**: Complete command-line tools for training and deployment
- **Server Mode**: HTTP/TCP server for neural network services
- **Flexible Architecture**: Support for any network topology
- **Serialization**: JSON, binary, and TOML formats

## Quick Start

```bash
# Build
cargo build --release

# Create config
./target/release/neural_network init-config -o config.toml

# Train network
./target/release/neural_network train -c config.toml -d data.json -o model.bin

# Run server
./target/release/neural_network server -m model.bin -p 8080
```

## Basic Usage

```rust
use neural_network::{NeuralNetwork, HebbianLearningMode};

// Create network with Hebbian learning
let mut net = NeuralNetwork::with_layers_and_mode(
    &[2, 4, 1], 
    0.1, 
    HebbianLearningMode::Classic
);

// Train (unsupervised)
net.train_unsupervised(&[1.0, 1.0]);

// Train (supervised)
let error = net.train(&[1.0, 0.0], &[1.0]);

// Predict
let output = net.predict(&[1.0, 0.0]);
```

## Hebbian Learning Modes

- **Classic**: Δw = η × pre × post
- **Competitive**: Winner-take-all with lateral inhibition
- **Oja**: Normalized learning with weight decay
- **BCM**: Sliding threshold adaptation
- **Anti-Hebbian**: Decorrelation learning
- **Hybrid**: Combined Hebbian + backpropagation

## Distributed Networks

```rust
use neural_network::DistributedNetwork;

// Create distributed node
let (dist_net, receiver) = DistributedNetwork::new(
    "node1".to_string(),
    "0.0.0.0".to_string(),
    8080,
    neural_network,
);

// Start server
dist_net.start_server().await?;

// Connect to peer
let peer_id = dist_net.connect_to("192.168.1.100", 8080).await?;

// Send neural data
dist_net.send_forward_data(peer_id, layer_id, data).await?;
```

## CLI Commands

```bash
# Configuration
neural_network init-config -o config.toml -n feedforward

# Training
neural_network train -c config.toml -d data.json -o model.bin -e 1000

# Prediction
neural_network predict -m model.bin -i input.json

# Interactive mode
neural_network interactive -c config.toml

# Benchmark
neural_network benchmark -c config.toml -i 100

# Server mode
neural_network server -m model.bin -p 8080 --daemon

# Demo
neural_network demo xor
```

## Configuration Format

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
validation_split = 0.2

[distributed]
enable = true
port = 8080
security = "tls"
```

## Data Formats

**JSON Training Data:**
```json
{
  "samples": [
    {"inputs": [0.0, 0.0], "targets": [0.0]},
    {"inputs": [0.0, 1.0], "targets": [1.0]}
  ]
}
```

**CSV Training Data:**
```csv
input1,input2,target
0.0,0.0,0.0
0.0,1.0,1.0
1.0,0.0,1.0
1.0,1.0,0.0
```

## Security Features

- **TLS 1.3 Encryption**: End-to-end encrypted communication
- **Certificate Authentication**: X.509 certificate validation
- **Capability-based Authorization**: Fine-grained access control
- **Message Integrity**: CRC32 checksums and sequence validation
- **Secure Key Exchange**: RSA/ECDSA key agreement

## Performance

- **Multi-core Optimization**: Automatic parallelization across CPU cores
- **Batch Processing**: Efficient training on multiple samples
- **Memory Efficient**: Optimized data structures and algorithms
- **Network Protocol**: Binary protocol with minimal overhead (22-byte headers)

## Examples

```bash
# Basic examples
cargo run --example simple_example
cargo run --example hebbian_learning

# Performance demos
cargo run --example multi_core_performance
cargo run --example benchmark_parallel

# Distributed computing
cargo run --example distributed_network
cargo run --example secure_distributed_network

# Advanced features
cargo run --example network_composition
cargo run --example online_learning

# I/O interfaces
cargo run --example io_interface_example
```

## I/O Interfaces

Connect neural networks to external systems via SSL/TCP protocols:

### Input Interfaces
Connect data sources to neural network inputs:
```rust
use neural_network::{IoManager, TcpInputInterface, IoConfig};

// Create input interface with TLS
let mut input = TcpInputInterface::with_tls_connector(tls_connector);
let config = IoConfig {
    connection_id: Uuid::new_v4(),
    endpoint: "sensor-data.example.com".to_string(),
    port: 8443,
    use_tls: true,
    buffer_size: 1024,
    timeout_ms: 5000,
    ..Default::default()
};

// Connect and start streaming
input.connect(config).await?;
input.start_streaming(data_sender).await?;
```

### Output Interfaces
Send neural network outputs to external systems:
```rust
use neural_network::{TcpOutputInterface, IoData};

// Create output interface
let mut output = TcpOutputInterface::with_tls_connector(tls_connector);
output.connect(output_config).await?;

// Send processed data
let result = IoData {
    timestamp: current_time(),
    values: vec![0.8, 0.2], // Neural network output
    metadata: HashMap::new(),
};
output.write_data(result).await?;
```

### I/O Manager
Coordinate multiple I/O connections:
```rust
use neural_network::{IoManager, NeuralNetwork};

// Create manager with neural network
let network = NeuralNetwork::new(4, 8, 2, 0.1);
let mut manager = IoManager::new(network);

// Add interfaces
manager.add_input_interface(input_id, Box::new(input_interface));
manager.add_output_interface(output_id, Box::new(output_interface));

// Start processing pipeline
manager.start_processing().await?;
```

### Use Cases
- **IoT Integration**: Connect sensors and actuators
- **API Processing**: Real-time data from web services  
- **Enterprise Systems**: Integration with existing infrastructure
- **Distributed AI**: Coordinate multiple neural networks
- **Edge Computing**: Deploy on resource-constrained devices

## API Reference

### Core Methods
- `NeuralNetwork::new(input, hidden, output, rate)` - Create network
- `train(&mut self, inputs, targets) -> f64` - Supervised training
- `train_unsupervised(&mut self, inputs)` - Hebbian learning
- `predict(&self, inputs) -> Vec<f64>` - Make predictions
- `save(&self, path)` / `load(path)` - Serialization

### Distributed Methods
- `DistributedNetwork::new()` - Create distributed node
- `start_server().await` - Start listening for connections
- `connect_to(host, port).await` - Connect to remote node
- `send_forward_data().await` - Send neural activations
- `send_hebbian_data().await` - Send correlation data

### Security Methods
- `SecureDistributedNetwork::new()` - Create secure node
- `load_certificate(path)` - Load X.509 certificate
- `enable_tls(config)` - Configure TLS settings
- `validate_peer(cert)` - Verify peer certificate

### I/O Interface Methods
- `TcpInputInterface::new()` - Create TCP input interface
- `TcpOutputInterface::new()` - Create TCP output interface
- `connect(config).await` - Connect to external system
- `start_streaming(channel).await` - Begin data streaming
- `IoManager::new(network)` - Create I/O coordinator
- `add_input_interface(id, interface)` - Register input
- `add_output_interface(id, interface)` - Register output
- `start_processing().await` - Begin I/O pipeline

## License

MIT License - see LICENSE file for details.