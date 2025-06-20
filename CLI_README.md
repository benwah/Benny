# Benny Neural Network CLI

A powerful command-line interface for creating, training, and running neural networks with various learning algorithms.

## Installation

```bash
cargo build --release
```

## Quick Start

1. **Create a configuration file:**
```bash
./target/release/neural_network init-config -o my_config.toml -n feedforward
```

2. **Prepare training data** (JSON or CSV format):
```json
{
  "inputs": [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
  "targets": [[0.0], [1.0], [1.0], [0.0]]
}
```

3. **Train the network:**
```bash
./target/release/neural_network train -c my_config.toml -d training_data.json -o trained_model.json -e 1000
```

4. **Make predictions:**
```bash
./target/release/neural_network predict -m trained_model.json -i "0.5,0.8" -f json
```

## Commands

### `init-config` - Create Configuration Files

Generate sample configuration files for different network types:

```bash
# Create a feedforward network config
neural_network init-config -o feedforward.toml -n feedforward

# Create a Hebbian learning network config  
neural_network init-config -o hebbian.toml -n hebbian

# Create an online learning network config
neural_network init-config -o online.toml -n online
```

**Options:**
- `-o, --output <FILE>`: Output configuration file path (default: network_config.toml)
- `-n, --network-type <TYPE>`: Network type (feedforward, hebbian, online, distributed)

### `train` - Train Neural Networks

Train a neural network using configuration and training data:

```bash
# Basic training
neural_network train -c config.toml -d data.json -o model.json -e 1000

# Training with verbose output
neural_network train -c config.toml -d data.csv -o model.bin -e 500 -v

# Training with custom validation split
neural_network train -c config.toml -d data.json -o model.json -e 1000 --validation-split 0.3
```

**Options:**
- `-c, --config <FILE>`: Configuration file path (required)
- `-d, --data <FILE>`: Training data file path (required)
- `-o, --output <FILE>`: Output model file path (required)
- `-e, --epochs <NUM>`: Number of training epochs (required)
- `-v, --verbose`: Enable verbose output
- `--validation-split <RATIO>`: Validation split ratio (0.0-1.0)

**Data Formats:**

*JSON Format:*
```json
{
  "inputs": [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
  "targets": [[0.0], [1.0], [1.0], [0.0]]
}
```

*CSV Format:*
```csv
input1,input2,target
0.0,0.0,0.0
0.0,1.0,1.0
1.0,0.0,1.0
1.0,1.0,0.0
```

### `predict` - Run Inference

Make predictions using trained models:

```bash
# Predict with saved model
neural_network predict -m model.json -i "0.5,0.8" -f json

# Predict with configuration (creates new network)
neural_network predict -c config.toml -i "0.5,0.8" -f plain

# Predict from input file
neural_network predict -m model.json -i input_data.csv -f csv
```

**Options:**
- `-c, --config <FILE>`: Configuration file path (optional if model provided)
- `-m, --model <FILE>`: Trained model file path (optional if config provided)
- `-i, --input <DATA>`: Input data (comma-separated values or file path)
- `-f, --format <FORMAT>`: Output format (json, csv, plain)

**Output Formats:**

*JSON:*
```json
{
  "timestamp": "2025-06-20T04:27:14.592Z",
  "input": [0.0, 0.0],
  "output": [0.279, 0.478, 0.263, 0.621],
  "confidence": 0.737,
  "processing_time_ms": 0.034
}
```

*Plain:*
```
Output: [0.279, 0.478, 0.263, 0.621]
Confidence: 73.70%
Processing time: 0.03ms
```

### `interactive` - Interactive Mode

Enter interactive mode for real-time experimentation:

```bash
neural_network interactive -c config.toml
```

**Interactive Commands:**
- `predict <input>`: Run prediction (e.g., 'predict 0.5,0.8')
- `train <input> <target>`: Train on single sample
- `info`: Show network information
- `save <file>`: Save network to file
- `load <file>`: Load network from file
- `quit`: Exit interactive mode

### `benchmark` - Performance Testing

Benchmark network performance:

```bash
neural_network benchmark -c config.toml -i 1000
```

**Options:**
- `-c, --config <FILE>`: Configuration file path (required)
- `-i, --iterations <NUM>`: Number of benchmark iterations (default: 1000)

### `demo` - Demonstrations

Run built-in demonstrations:

```bash
neural_network demo
```

Shows examples of:
- XOR problem solving
- Hebbian learning
- Model serialization/deserialization

## Configuration File Format

Configuration files use TOML format:

```toml
# Network architecture (layer sizes)
architecture = [2, 4, 1]

# Learning parameters
learning_rate = 0.1
hebbian_mode = "Classic"  # Classic, Competitive, Oja, BCM
hebbian_rate = 0.05
decay_rate = 0.005

# Backpropagation settings
use_backprop = true
backprop_rate = 0.1

# Online learning
online_learning = false

[training]
batch_size = 32
print_interval = 100
early_stop_threshold = 0.001
early_stop_patience = 50
validation_split = 0.2
```

## Network Types

### Feedforward Networks
- Standard multilayer perceptrons
- Backpropagation training
- Suitable for classification and regression

### Hebbian Learning Networks
- Unsupervised learning
- Multiple Hebbian modes: Classic, Competitive, Oja, BCM
- Self-organizing capabilities

### Online Learning Networks
- Real-time adaptation
- Continuous learning from streaming data
- Combines Hebbian and backpropagation learning

### Distributed Networks
- Advanced distributed processing
- Parallel computation capabilities
- Scalable architecture

## File Formats

### Model Files
- **JSON (.json)**: Human-readable, cross-platform
- **Binary (.bin)**: Compact, faster loading

### Data Files
- **JSON (.json)**: Structured format with inputs/targets
- **CSV (.csv)**: Tabular format with headers

## Examples

### XOR Problem
```bash
# Create config
neural_network init-config -o xor.toml -n feedforward

# Create data file (xor_data.json)
echo '{
  "inputs": [[0,0], [0,1], [1,0], [1,1]],
  "targets": [[0], [1], [1], [0]]
}' > xor_data.json

# Train
neural_network train -c xor.toml -d xor_data.json -o xor_model.json -e 1000 -v

# Test
neural_network predict -m xor_model.json -i "1,0" -f plain
```

### Hebbian Learning
```bash
# Create Hebbian config
neural_network init-config -o hebbian.toml -n hebbian

# Interactive exploration
neural_network interactive -c hebbian.toml
```

### Performance Analysis
```bash
# Benchmark different architectures
neural_network benchmark -c small_net.toml -i 10000
neural_network benchmark -c large_net.toml -i 1000
```

## Tips

1. **Start Small**: Begin with simple architectures and small datasets
2. **Use Verbose Mode**: Add `-v` flag during training to monitor progress
3. **Save Models**: Always specify output paths to save trained models
4. **Experiment**: Use interactive mode to explore network behavior
5. **Benchmark**: Test performance before deploying large networks
6. **Validate**: Use validation splits to prevent overfitting

## Troubleshooting

### Common Issues

**Training not converging:**
- Reduce learning rate
- Increase number of epochs
- Check data quality and normalization

**Memory issues:**
- Reduce batch size
- Use smaller architectures
- Save models in binary format

**Slow performance:**
- Use binary model format
- Reduce network complexity
- Benchmark different configurations

### Getting Help

```bash
# General help
neural_network --help

# Command-specific help
neural_network train --help
neural_network predict --help
```