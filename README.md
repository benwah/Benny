# 🧠 Simple Neural Network in Rust

A basic implementation of a feedforward neural network written in Rust from scratch. This project demonstrates fundamental concepts of neural networks including forward propagation, backpropagation, and gradient descent.

## Features

- **Flexible Architecture**: Support for any number of layers and neurons per layer
- **Multiple Constructors**: Simple `new()` for basic networks, `with_layers()` for complex architectures
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

## Examples

Run the examples to see the neural network in action:

```bash
# Main demo with flexible architectures and logic problems
cargo run

# Simple XOR example
cargo run --example simple_example

# Flexible architecture demonstration
cargo run --example flexible_architecture
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