# 🧠 Hebbian Neural Network in Rust

A biologically-inspired neural network implementation in Rust that makes **Hebbian learning** the central organizing principle. This project demonstrates both classical backpropagation and modern understanding of synaptic plasticity with the famous principle: *"Neurons that fire together, wire together"*.

## 🌟 Core Philosophy

This neural network is built around **Hebbian learning** as the primary learning mechanism, with backpropagation as an optional supplement. This reflects modern neuroscience understanding that biological neural networks primarily learn through correlation-based synaptic plasticity.

## 🧬 Hebbian Learning Features

- **6 Hebbian Learning Modes**:
  - **Classic**: Traditional Hebbian learning (Δw = η × pre × post)
  - **Competitive**: Winner-take-all dynamics with lateral inhibition
  - **Oja**: Normalized Hebbian learning with weight decay
  - **BCM**: Bienenstock-Cooper-Munro rule with sliding threshold
  - **AntiHebbian**: Negative correlation learning for decorrelation
  - **Hybrid**: Combined Hebbian + backpropagation learning

- **Biological Mechanisms**:
  - Correlation matrix tracking for neuron relationships
  - Homeostatic regulation to prevent runaway activation
  - Anti-Hebbian mechanisms for competitive learning
  - Temporal correlation analysis through activation history
  - Weight decay for synaptic stability

## 🚀 Performance Features

- **Multi-Core Optimization**: Utilizes all CPU cores for parallel processing
  - Parallel forward propagation across neurons within layers
  - Parallel backpropagation for error calculation and weight updates
  - Batch processing for training multiple samples simultaneously
  - Optimized matrix operations using Rayon for maximum performance
- **Flexible Architecture**: Support for any number of layers and neurons per layer
- **Network Composition**: Connect multiple neural networks together in complex architectures

## Architecture

The neural network now supports flexible architectures:

```
Simple Network:
Input Layer → Hidden Layer → Output Layer
     ↓             ↓             ↓
   2 nodes      3 nodes       1 node

Deep Network:
Input → Hidden1 → Hidden2 → Hidden3 → Output
  ↓        ↓         ↓         ↓        ↓
4 nodes  8 nodes   6 nodes   4 nodes  2 nodes

Direct Network (no hidden layers):
Input Layer → Output Layer
     ↓             ↓
   3 nodes      2 nodes
```

## Quick Start

### Prerequisites

- Rust (latest stable version)
- Cargo (comes with Rust)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd neural_network
```

2. Build the project:
```bash
cargo build
```

3. Run the examples:
```bash
cargo run
```

4. Run tests:
```bash
cargo test
```

## 🎯 Usage

### 🧬 Hebbian Learning (Primary Approach)

```rust
use neural_network::{NeuralNetwork, HebbianLearningMode};

// Create a network with Classic Hebbian learning
let mut network = NeuralNetwork::new(2, 3, 1, 0.1);

// Unsupervised learning with correlated patterns
let patterns = [
    [1.0, 1.0],  // Both high - should strengthen connections
    [1.0, 0.8],  // High correlation
    [0.9, 1.0],  // High correlation  
    [0.0, 0.0],  // Both low
];

// Train with pure Hebbian learning (no targets needed!)
for epoch in 0..50 {
    for pattern in &patterns {
        network.train_unsupervised(pattern);
    }
}

// Check learned correlations
let correlation = network.get_neuron_correlation(0, 0, 0, 1);
println!("Learned correlation: {:.4}", correlation);
```

### 🔀 Hybrid Learning (Hebbian + Backpropagation)

```rust
use neural_network::NeuralNetwork;

// Create a hybrid network combining both learning types
let mut network = NeuralNetwork::with_hybrid_learning(&[2, 4, 1], 0.05, 0.3);

// XOR problem with hybrid learning
let xor_data = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
];

// Train with both Hebbian and supervised learning
for epoch in 0..1000 {
    for (inputs, targets) in &xor_data {
        let error = network.train(inputs, targets);
    }
}
```

### 🎛️ Different Hebbian Learning Modes

```rust
use neural_network::{NeuralNetwork, HebbianLearningMode};

// Competitive learning (winner-take-all)
let mut competitive_net = NeuralNetwork::with_layers_and_mode(
    &[10, 5, 2], 
    0.1, 
    HebbianLearningMode::Competitive
);

// Oja's rule (normalized Hebbian learning)
let mut oja_net = NeuralNetwork::with_layers_and_mode(
    &[8, 4, 1], 
    0.05, 
    HebbianLearningMode::Oja
);

// Anti-Hebbian learning (decorrelation)
let mut anti_net = NeuralNetwork::with_layers_and_mode(
    &[6, 3, 2], 
    0.02, 
    HebbianLearningMode::AntiHebbian
);
```

### Flexible Architecture Examples

```rust
use neural_network::NeuralNetwork;

// Simple network: 2 inputs → 3 hidden → 1 output
let nn1 = NeuralNetwork::with_layers(&[2, 3, 1], 0.1);

// Deep network: 4 inputs → 8 hidden → 6 hidden → 3 hidden → 2 outputs
let nn2 = NeuralNetwork::with_layers(&[4, 8, 6, 3, 2], 0.05);

// Wide network: 3 inputs → 20 hidden → 1 output
let nn3 = NeuralNetwork::with_layers(&[3, 20, 1], 0.1);

// Direct network (no hidden layers): 5 inputs → 3 outputs
let nn4 = NeuralNetwork::with_layers(&[5, 3], 0.2);

// Get network information
println!("Architecture: {}", nn2.info());
println!("Parameters: {}", nn2.num_parameters());
println!("Hidden layers: {}", nn2.num_hidden_layers());
```

## Neural Network API

### 🏗️ Constructors (Hebbian-Centric)

```rust
// Default Classic Hebbian learning
NeuralNetwork::new(input_size: usize, hidden_size: usize, output_size: usize, hebbian_rate: f64)

// Specify Hebbian learning mode
NeuralNetwork::new_with_mode(input_size: usize, hidden_size: usize, output_size: usize, 
                            hebbian_rate: f64, mode: HebbianLearningMode)

// Multi-layer Classic Hebbian
NeuralNetwork::with_layers(layer_sizes: &[usize], hebbian_rate: f64)

// Multi-layer with mode selection
NeuralNetwork::with_layers_and_mode(layer_sizes: &[usize], hebbian_rate: f64, mode: HebbianLearningMode)

// Hybrid Hebbian + Backpropagation
NeuralNetwork::with_hybrid_learning(layer_sizes: &[usize], hebbian_rate: f64, backprop_rate: f64)
```

### 🧬 Hebbian Learning Methods

- `train(&mut self, inputs: &[f64], targets: &[f64]) -> f64`: Primary training method (Hebbian + optional backprop)
- `train_unsupervised(&mut self, inputs: &[f64])`: Pure Hebbian learning without targets
- `get_neuron_correlation(&self, layer: usize, neuron1: usize, neuron2: usize) -> f64`: Analyze neuron relationships
- `get_average_activation(&self, layer: usize, neuron: usize) -> f64`: Monitor network dynamics
- `get_hebbian_rate(&self) -> f64`: Access Hebbian learning rate

### 🎯 Core Methods

- `predict(&self, inputs: &[f64]) -> Vec<f64>`: Make predictions using the trained network
- `forward(&self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>)`: Perform forward propagation (returns hidden and output activations)
- `forward_all_layers(&self, inputs: &[f64]) -> Vec<Vec<f64>>`: Get activations from all layers

### Information Methods

- `info(&self) -> String`: Get network architecture information
- `get_layers(&self) -> &[usize]`: Get layer sizes
- `num_layers(&self) -> usize`: Get total number of layers
- `num_hidden_layers(&self) -> usize`: Get number of hidden layers
- `num_parameters(&self) -> usize`: Get total number of parameters (weights + biases)

### Parallel Processing Methods

- `train_batch(&mut self, batch: &[(Vec<f64>, Vec<f64>)]) -> f64`: Train on multiple samples in parallel
- `forward_batch(&self, inputs_batch: &[Vec<f64>]) -> Vec<Vec<f64>>`: Process multiple inputs in parallel

## 🌐 Distributed Neural Networks

This library includes a revolutionary **Neural Network Protocol (NNP)** - an optimized binary TCP protocol designed specifically for real-time neural network communication across the internet.

### Key Features

- **🚀 High Performance**: Binary protocol with minimal 22-byte headers
- **🔒 Data Integrity**: CRC32 checksums and message sequencing
- **🌍 Internet Scale**: TCP-based for reliable communication across networks
- **🧬 Neural Optimized**: Specialized for forward/backward propagation and Hebbian data
- **⚡ Real-time**: f32 precision and optimized serialization for speed

### Protocol Capabilities

```rust
// Create distributed neural network nodes
let (dist_net, receiver) = DistributedNetwork::new(
    "MyNetwork".to_string(),
    "0.0.0.0".to_string(),
    8080,
    neural_network,
);

// Start server for incoming connections
dist_net.start_server().await?;

// Connect to remote neural network
let peer_id = dist_net.connect_to("192.168.1.100", 8080).await?;

// Send neural data across the network
dist_net.send_forward_data(peer_id, layer_id, data).await?;
dist_net.send_hebbian_data(peer_id, layer_id, correlations, learning_rate).await?;
```

### Use Cases

- **Federated Learning**: Train on distributed data while preserving privacy
- **Pipeline Processing**: Different layers on different machines
- **Ensemble Networks**: Multiple networks collaborating on same problem
- **Edge Computing**: Coordinate neural networks across IoT devices
- **Research Clusters**: Connect neural networks across institutions

See [DISTRIBUTED_NETWORKING.md](DISTRIBUTED_NETWORKING.md) for complete protocol specification and examples.

## 🎮 Examples

### Run the Examples

```bash
# Simple Hebbian learning demonstration
cargo run --example simple_hebbian

# Comprehensive Hebbian learning showcase
cargo run --example hebbian_learning

# Multi-core performance demonstration
cargo run --example multi_core_performance

# Basic neural network examples
cargo run
```

### Example Outputs

**Simple Hebbian Learning:**
```
🧠 Simple Hebbian Neural Network in Rust
=========================================

📊 Network Info: Neural Network: 2 -> 3 -> 1 (Hebbian rate: 0.1, mode: Classic)

🎯 Training with unsupervised Hebbian learning...
  Epoch 0: Output for [1.0, 1.0] = 0.3640
  Epoch 10: Output for [1.0, 1.0] = 0.9980
  Epoch 20: Output for [1.0, 1.0] = 1.0000

🔗 Final neuron correlations:
  Input[0] <-> Input[1]: 0.9660

✨ Hebbian learning complete! Neurons that fired together are now wired together.
```

**Hebbian Learning Showcase:**
- Demonstrates all 6 Hebbian learning modes
- Shows correlation analysis and neuron dynamics
- Compares pure Hebbian vs hybrid learning approaches
- Includes biological insights and learning principles

## Multi-Core Performance

This neural network implementation is optimized for multi-core systems using the Rayon library for parallel processing.

### Performance Features

- **Parallel Forward Propagation**: Neuron computations within each layer are distributed across CPU cores
- **Parallel Backpropagation**: Error calculations and weight updates utilize all available cores
- **Batch Processing**: Multiple training samples can be processed simultaneously
- **Optimized Matrix Operations**: Inner products and vector operations use parallel iterators

### Performance Examples

```rust
use neural_network::NeuralNetwork;

// Create a large network that benefits from parallelization
let mut nn = NeuralNetwork::with_layers(&[100, 200, 150, 100, 50], 0.01);

// Batch training (processes samples in parallel)
let training_batch = vec![
    (vec![/* 100 inputs */], vec![/* 50 targets */]),
    (vec![/* 100 inputs */], vec![/* 50 targets */]),
    // ... more samples
];
let batch_error = nn.train_batch(&training_batch);

// Batch forward propagation
let input_batch = vec![
    vec![/* 100 inputs */],
    vec![/* 100 inputs */],
    // ... more inputs
];
let outputs = nn.forward_batch(&input_batch);
```

### Performance Tips

- **Use wider networks**: Networks with more neurons per layer benefit most from parallelization
- **Batch processing**: Train multiple samples together for better CPU utilization
- **Large networks**: Bigger networks see greater speedup from parallel processing
- **Multi-core systems**: Performance scales with the number of available CPU cores

### Benchmarking

Run the performance benchmarks to see the multi-core optimizations in action:

```bash
# Comprehensive performance demo
cargo run --example multi_core_performance

# Detailed benchmarks
cargo run --example benchmark_parallel

# Distributed neural network protocol demo
cargo run --example distributed_network

# Secure distributed network with TLS and certificates
cargo run --example secure_distributed_network
```

## Hebbian Learning

This neural network implementation includes biologically-inspired Hebbian learning, following the principle "neurons that fire together, wire together."

### Hebbian Learning Constructor

```rust
// Create a network with Hebbian learning capabilities
let mut nn = NeuralNetwork::with_hebbian_learning(
    &[2, 4, 1],  // Layer sizes
    0.1,         // Backpropagation learning rate
    0.05,        // Hebbian learning rate
    10,          // History size (number of recent activations to remember)
    0.001        // Weight decay rate (prevents unbounded growth)
);
```

### Hebbian Learning Methods

- `train_hebbian(&mut self, inputs: &[f64])`: Pure Hebbian learning (unsupervised)
- `train_hybrid(&mut self, inputs: &[f64], targets: &[f64]) -> f64`: Combine backpropagation + Hebbian learning
- `get_neuron_correlation(&self, layer1: usize, neuron1: usize, layer2: usize, neuron2: usize) -> f64`: Calculate correlation between neurons
- `get_average_activation(&self, layer: usize, neuron: usize) -> f64`: Get average activation for a neuron
- `reset_activation_history(&mut self)`: Reset activation history for fresh experiments

### Hebbian Learning Examples

```rust
use neural_network::NeuralNetwork;

// Create Hebbian network
let mut nn = NeuralNetwork::with_hebbian_learning(&[2, 3, 1], 0.1, 0.05, 10, 0.001);

// Pure Hebbian learning (unsupervised)
for _ in 0..100 {
    nn.train_hebbian(&[1.0, 1.0]); // Train with correlated inputs
}

// Check correlation between input neurons
let correlation = nn.get_neuron_correlation(0, 0, 0, 1);
println!("Input correlation: {:.4}", correlation);

// Hybrid learning (supervised + unsupervised)
let error = nn.train_hybrid(&[1.0, 0.0], &[1.0]);
```

### Key Concepts

- **Correlation-based Learning**: Weights strengthen when neurons are co-active
- **Activation History**: Network remembers recent neuron activations for correlation calculation
- **Weight Decay**: Prevents unbounded weight growth in Hebbian learning
- **Hybrid Training**: Combines supervised (backpropagation) and unsupervised (Hebbian) learning

## Network Composition

Connect multiple neural networks together to create complex architectures like pipelines, fan-out systems, ensembles, and multi-stage processing networks.

### NetworkComposer API

```rust
use neural_network::{NeuralNetwork, NetworkComposer};
use std::collections::HashMap;

// Create a composer to manage multiple networks
let mut composer = NetworkComposer::new();

// Add individual networks
let feature_net = NeuralNetwork::new(4, 6, 3, 0.1);
let classifier_net = NeuralNetwork::new(3, 4, 2, 0.1);

composer.add_network("features".to_string(), feature_net).unwrap();
composer.add_network("classifier".to_string(), classifier_net).unwrap();

// Connect outputs of one network to inputs of another
composer.connect_networks(
    "features",      // Source network
    "classifier",    // Target network
    vec![0, 1, 2],   // Source output indices
    vec![0, 1, 2]    // Target input indices
).unwrap();
```

### Network Composition Methods

- `add_network(name: String, network: NeuralNetwork)`: Add a network to the composition
- `remove_network(name: &str)`: Remove a network from the composition
- `connect_networks(source, target, source_outputs, target_inputs)`: Connect network outputs to inputs
- `forward(inputs: &HashMap<String, Vec<f64>>)`: Forward propagation through entire composition
- `train_network(name, inputs, targets)`: Train a specific network in the composition
- `get_network(name)` / `get_network_mut(name)`: Access individual networks
- `info()`: Get detailed information about the composition

### Composition Architectures

#### 1. Pipeline Architecture
```
Input -> Network1 -> Network2 -> Output
```

#### 2. Fan-out Architecture  
```
Input -> Network1 -> [Network2, Network3]
```

#### 3. Ensemble Architecture
```
Input -> [Network1, Network2, Network3] -> Voting Network -> Output
```

#### 4. Multi-Stage Processing
```
InputA -> NetworkA ↘
                    Fusion -> Output
InputB -> NetworkB ↗
```

### Example: Simple Pipeline

```rust
let mut composer = NetworkComposer::new();

// Feature extraction network: 4 inputs -> 3 features
let feature_extractor = NeuralNetwork::new(4, 6, 3, 0.1);
// Classification network: 3 features -> 2 classes  
let classifier = NeuralNetwork::new(3, 4, 2, 0.1);

composer.add_network("extractor".to_string(), feature_extractor).unwrap();
composer.add_network("classifier".to_string(), classifier).unwrap();

// Connect all feature outputs to classifier inputs
composer.connect_networks("extractor", "classifier", vec![0, 1, 2], vec![0, 1, 2]).unwrap();

// Forward propagation
let mut inputs = HashMap::new();
inputs.insert("extractor".to_string(), vec![0.8, 0.3, 0.9, 0.2]);

let outputs = composer.forward(&inputs).unwrap();
println!("Features: {:?}", outputs["extractor"]);
println!("Classification: {:?}", outputs["classifier"]);
```

### Key Features

- **Automatic Execution Order**: Networks are executed in topological order based on connections
- **Cycle Detection**: Prevents creation of circular dependencies
- **Flexible Connections**: Connect any subset of outputs to any subset of inputs
- **Individual Training**: Train specific networks within the composition
- **Validation**: Comprehensive error checking for network sizes and connection validity

## Distributed Networking

The neural network supports distributed computing across multiple nodes with an optimized binary protocol and enterprise-grade security:

### Neural Network Protocol (NNP)

- **Binary Protocol**: Efficient 22-byte header with magic number, version, message type, length, sequence, and CRC32 checksum
- **Message Types**: Handshake, data forwarding, Hebbian learning synchronization, heartbeat, and error handling
- **TCP Communication**: Async networking with Tokio for high-performance distributed training
- **Network Discovery**: Automatic peer discovery and connection management
- **Load Balancing**: Intelligent distribution of computational workload

### Security Features

- **TLS 1.3 Encryption**: End-to-end encryption for all network communication
- **Certificate-based Authentication**: X.509 certificates with neural network-specific extensions
- **Capability Authorization**: Fine-grained permissions based on certificate capabilities
- **Network Identity Verification**: UUID-based network identification with certificate validation
- **Certificate Authority Support**: Full CA infrastructure for enterprise deployment

### Protocol Features

- **Reliability**: CRC32 checksums ensure data integrity
- **Versioning**: Protocol version negotiation for compatibility
- **Scalability**: Supports large-scale distributed neural networks
- **Performance**: Optimized binary serialization for minimal overhead
- **Security**: Enterprise-grade TLS encryption and certificate-based authentication

See [DISTRIBUTED_NETWORKING.md](DISTRIBUTED_NETWORKING.md) for complete protocol specification and [SECURITY.md](SECURITY.md) for security architecture details.

## Examples

Run the examples to see the neural network in action:

```bash
# Main demo with flexible architectures and logic problems
cargo run

# Simple neural network demo
cargo run --example simple_neural_network

# Simple XOR example
cargo run --example simple_example

# Flexible architecture demonstration
cargo run --example flexible_architecture

# Hebbian learning demonstration
cargo run --example hebbian_learning

# Network composition demonstration
cargo run --example network_composition
```

### Included Problems

1. **XOR Problem**: Non-linearly separable problem requiring hidden layers
2. **AND Problem**: Logical AND function
3. **OR Problem**: Logical OR function
4. **Multi-class Classification**: Demonstrates multi-output networks

## Implementation Details

### Activation Function
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- **Derivative**: `f'(x) = f(x) * (1 - f(x))`

### Training Algorithm
1. **Forward Propagation**: Calculate outputs for given inputs
2. **Error Calculation**: Compute the difference between predicted and target outputs
3. **Backward Propagation**: Calculate gradients using the chain rule
4. **Weight Update**: Adjust weights and biases using gradient descent

### Weight Initialization
Weights and biases are randomly initialized between -1 and 1 using Rust's `rand` crate.

## Performance

The network typically converges within:
- **XOR Problem**: ~10,000 epochs
- **AND Problem**: ~5,000 epochs  
- **OR Problem**: ~5,000 epochs

Final accuracy is typically >99% for all problems.

## Dependencies

- `rand = "0.8"` - For random weight initialization

## Project Structure

```
src/
├── lib.rs              # Library entry point
├── main.rs             # Example demonstrations
└── neural_network.rs   # Core neural network implementation
```

## Testing

Run the test suite with:

```bash
cargo test
```

Tests cover:
- Neural network creation
- Forward propagation
- Prediction functionality

## 🧬 Biological Inspiration

This neural network implementation is inspired by real biological neural networks and implements several key principles from neuroscience:

### 🔬 Hebbian Learning Principle

> *"Neurons that fire together, wire together"* - Donald Hebb (1949)

This fundamental principle of synaptic plasticity is the core of our implementation. When two neurons are active simultaneously, the connection between them strengthens, leading to associative learning without external supervision.

### 🧠 Synaptic Plasticity Mechanisms

- **Long-Term Potentiation (LTP)**: Strengthening of synapses based on recent patterns of activity
- **Long-Term Depression (LTD)**: Weakening of synapses to prevent saturation
- **Homeostatic Plasticity**: Regulation of overall neural activity to maintain stability
- **Competitive Learning**: Winner-take-all dynamics similar to cortical columns

### 🔄 Learning Without Teachers

Unlike traditional backpropagation which requires labeled data, Hebbian learning enables:
- **Unsupervised Learning**: Pattern recognition without explicit targets
- **Self-Organization**: Emergence of structure from input statistics
- **Correlation Detection**: Automatic discovery of input relationships
- **Feature Learning**: Development of useful representations

### 🌟 Modern Neuroscience Integration

Our implementation incorporates recent findings from neuroscience:
- **BCM Rule**: Sliding threshold for bidirectional plasticity
- **Oja's Rule**: Normalized learning to prevent weight explosion
- **Anti-Hebbian Learning**: Decorrelation mechanisms for efficient coding
- **Homeostatic Regulation**: Activity-dependent scaling for stability

This makes the network not just a computational tool, but a model of how biological brains actually learn and adapt.

## 🌐 Server/Daemon Mode

The neural network can run as a server that listens for SSL connections, accepts input activations, and applies Hebbian learning as data flows through:

```bash
# Start a neural network server with Hebbian learning
neural_network server --config network_config.json --port 8080 --hebbian-learning

# Run with SSL/TLS encryption
neural_network server \
  --config network_config.json \
  --port 8080 \
  --cert server.crt \
  --key server.key \
  --hebbian-learning

# Forward outputs to other neural networks
neural_network server \
  --config network_config.json \
  --port 8080 \
  --outputs "192.168.1.100:8081" "192.168.1.101:8082" \
  --hebbian-learning
```

**Server Features:**
- **Neural Network Protocol (NNP)**: Binary protocol for efficient communication
- **SSL/TLS Support**: Secure encrypted connections
- **Hebbian Learning**: Real-time learning on incoming activations
- **Output Forwarding**: Chain multiple networks together
- **Daemon Mode**: Background operation
- **Distributed Architecture**: Integrates with existing distributed network capabilities

See [docs/server_mode.md](docs/server_mode.md) for detailed documentation.

## Future Enhancements

Potential improvements for this Hebbian neural network:

- [x] **Multiple hidden layers** (deep networks) ✅ Implemented
- [x] **Hebbian learning mechanisms** ✅ 6 different modes implemented
- [x] **Multi-core optimization** ✅ Parallel processing with Rayon
- [x] **Batch training support** ✅ Parallel batch processing
- [x] **Correlation analysis** ✅ Neuron relationship tracking
- [x] **Distributed networking** ✅ Optimized TCP protocol for internet-scale neural networks
- [ ] Different activation functions (ReLU, tanh, etc.)
- [ ] Different optimization algorithms (Adam, RMSprop)
- [ ] Regularization techniques (dropout, L1/L2)
- [ ] Save/load trained models
- [ ] More complex datasets (MNIST, etc.)
- [ ] GPU acceleration with CUDA
- [ ] Convolutional layers with Hebbian learning
- [ ] Spike-timing dependent plasticity (STDP)
- [ ] Neuromodulation mechanisms (dopamine, etc.)
- [ ] Recurrent connections and memory

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.