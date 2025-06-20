# 🧠 Simple Neural Network in Rust

A basic implementation of a feedforward neural network written in Rust from scratch. This project demonstrates fundamental concepts of neural networks including forward propagation, backpropagation, and gradient descent.

## Features

- **Simple Architecture**: Single hidden layer feedforward neural network
- **Sigmoid Activation**: Uses sigmoid activation function with its derivative
- **Backpropagation**: Implements backpropagation algorithm for training
- **Multiple Problems**: Demonstrates solving XOR, AND, and OR logic problems
- **Modular Design**: Clean separation between neural network implementation and examples
- **Unit Tests**: Comprehensive test coverage for core functionality

## Architecture

```
Input Layer → Hidden Layer → Output Layer
     ↓             ↓             ↓
   2 nodes    3-4 nodes      1 node
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

### Basic Example

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

## Neural Network API

### Constructor

```rust
NeuralNetwork::new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64)
```

Creates a new neural network with the specified architecture and learning rate.

### Methods

- `train(&mut self, inputs: &[f64], targets: &[f64]) -> f64`: Train the network with input-target pairs, returns error
- `predict(&self, inputs: &[f64]) -> Vec<f64>`: Make predictions using the trained network
- `forward(&self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>)`: Perform forward propagation (returns hidden and output activations)
- `info(&self) -> String`: Get network architecture information

## Examples

The project includes three classic problems:

### 1. XOR Problem
The XOR (exclusive OR) function is a classic non-linearly separable problem that requires a hidden layer to solve.

### 2. AND Problem
The logical AND function - outputs 1 only when both inputs are 1.

### 3. OR Problem
The logical OR function - outputs 1 when at least one input is 1.

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