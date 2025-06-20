# 🧠 Simple Neural Network in Rust

A basic implementation of a feedforward neural network written in Rust from scratch. This project demonstrates fundamental concepts of neural networks including forward propagation, backpropagation, and gradient descent.

## Features

- **Multi-Core Optimization**: Utilizes all CPU cores for parallel processing
  - Parallel forward propagation across neurons within layers
  - Parallel backpropagation for error calculation and weight updates
  - Batch processing for training multiple samples simultaneously
  - Optimized matrix operations using Rayon for maximum performance
- **Flexible Architecture**: Support for any number of layers and neurons per layer
- **Network Composition**: Connect multiple neural networks together in complex architectures
- **Hebbian Learning**: Biologically-inspired "neurons that fire together, wire together" learning
- **Activation History**: Track neuron activations over time for correlation-based learning
- **Hybrid Training**: Combine backpropagation with Hebbian learning for enhanced performance
- **Multiple Constructors**: Simple `new()` for basic networks, `with_layers()` for complex architectures, `with_hebbian_learning()` for bio-inspired networks
- **Deep Networks**: Support for multiple hidden layers of varying sizes
- **Multi-Input/Output**: Handle complex classification and regression problems
- **Sigmoid Activation**: Uses sigmoid activation function with its derivative
- **Backpropagation**: Implements backpropagation algorithm for training
- **Multiple Problems**: Demonstrates solving XOR, AND, OR logic problems and flexible architectures
- **Modular Design**: Clean separation between neural network implementation and examples
- **Parameter Analysis**: Built-in methods to count parameters and analyze network complexity
- **Unit Tests**: Comprehensive test coverage for core functionality

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

## Usage

### Basic Example (Backward Compatible)

```rust
use neural_network::NeuralNetwork;

// Create a neural network: 2 inputs, 4 hidden neurons, 1 output
let mut nn = NeuralNetwork::new(2, 4, 1, 0.5);

// Training data for XOR problem
let training_data = vec![
    (vec![0.0, 0.0], vec![0.0]),
    (vec![0.0, 1.0], vec![1.0]),
    (vec![1.0, 0.0], vec![1.0]),
    (vec![1.0, 1.0], vec![0.0]),
];

// Train the network
for epoch in 0..10000 {
    for (inputs, targets) in &training_data {
        nn.train(inputs, targets);
    }
}

// Make predictions
let prediction = nn.predict(&[1.0, 0.0]);
println!("Prediction: {:.4}", prediction[0]);
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

### Constructors

```rust
// Simple constructor (backward compatible)
NeuralNetwork::new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64)

// Flexible constructor for any architecture
NeuralNetwork::with_layers(layer_sizes: &[usize], learning_rate: f64)
```

### Core Methods

- `train(&mut self, inputs: &[f64], targets: &[f64]) -> f64`: Train the network with input-target pairs, returns error
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

## Examples

Run the examples to see the neural network in action:

```bash
# Main demo with flexible architectures and logic problems
cargo run

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

## Future Enhancements

Potential improvements for this simple neural network:

- [ ] Multiple hidden layers (deep networks)
- [ ] Different activation functions (ReLU, tanh, etc.)
- [ ] Different optimization algorithms (Adam, RMSprop)
- [ ] Batch training support
- [ ] Regularization techniques (dropout, L1/L2)
- [ ] Save/load trained models
- [ ] More complex datasets (MNIST, etc.)
- [ ] GPU acceleration
- [ ] Convolutional layers

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.