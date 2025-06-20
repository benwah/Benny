# Neural Network Server Mode

The neural network can run as a server/daemon that listens for SSL connections, accepts input activations, forwards them through the network, and applies Hebbian learning as data flows through.

## Features

- **Neural Network Protocol (NNP)**: Uses the existing binary protocol for efficient communication
- **SSL/TLS Support**: Leverages existing secure networking infrastructure
- **Hebbian Learning**: Applies learning updates on incoming activations
- **Output Forwarding**: Can forward outputs to other connected networks
- **Daemon Mode**: Can run as a background process
- **Distributed Architecture**: Integrates with existing distributed network capabilities

## Usage

### Basic Server

```bash
# Start a basic neural network server
neural_network server --config network_config.json --port 8080
```

### Server with Hebbian Learning

```bash
# Enable Hebbian learning on incoming activations
neural_network server --config network_config.json --port 8080 --hebbian-learning
```

### Secure Server with SSL/TLS

```bash
# Run with SSL/TLS encryption
neural_network server \
  --config network_config.json \
  --port 8080 \
  --cert server.crt \
  --key server.key \
  --hebbian-learning
```

### Server with Output Forwarding

```bash
# Forward outputs to other neural networks
neural_network server \
  --config network_config.json \
  --port 8080 \
  --outputs "192.168.1.100:8081" "192.168.1.101:8082" \
  --hebbian-learning
```

### Daemon Mode

```bash
# Run as a background daemon
neural_network server \
  --config network_config.json \
  --port 8080 \
  --daemon \
  --hebbian-learning
```

### Pre-trained Model

```bash
# Load a pre-trained model instead of creating new network
neural_network server \
  --config network_config.json \
  --model trained_model.bin \
  --port 8080 \
  --hebbian-learning
```

## Command Line Options

- `--config <CONFIG>`: Configuration file path (required)
- `--model <MODEL>`: Pre-trained model file path (optional)
- `--port <PORT>`: Port to listen on (default: 8080)
- `--cert <CERT>`: SSL certificate file path
- `--key <KEY>`: SSL private key file path
- `--outputs <OUTPUTS>`: Output network endpoints (host:port)
- `--daemon`: Run as daemon (background process)
- `--hebbian-learning`: Enable Hebbian learning on activations

## Protocol

The server uses the Neural Network Protocol (NNP), a binary protocol that supports:

- **ForwardData**: Neural network activations
- **HebbianData**: Hebbian learning correlations
- **Handshake**: Network identification and capabilities

### Message Types

1. **ForwardData**: Contains layer ID and activation data
   - Processed through the neural network
   - Outputs forwarded to connected endpoints
   - Hebbian learning applied if enabled

2. **HebbianData**: Contains layer ID, correlations, and learning rate
   - Applied to network weights (future enhancement)

3. **Handshake**: Network identification
   - Automatic response handled by distributed network

## Architecture

The server leverages existing infrastructure:

- **DistributedNetwork**: Handles NNP protocol and connections
- **SecureDistributedNetwork**: Provides SSL/TLS encryption
- **NeuralNetwork**: Processes activations and applies learning
- **NetworkServer**: Coordinates message processing and forwarding

## Example Workflow

1. **Server Startup**: Load configuration and create/load neural network
2. **Listen**: Start NNP server on specified port
3. **Accept Connections**: Handle incoming SSL connections
4. **Process Messages**: 
   - Receive ForwardData messages
   - Process through neural network
   - Apply Hebbian learning if enabled
   - Forward outputs to connected networks
5. **Continuous Operation**: Process messages in daemon mode

## Integration with Distributed Networks

The server integrates seamlessly with the existing distributed network infrastructure:

- Uses the same NNP protocol for consistency
- Leverages SSL/TLS capabilities for security
- Supports network discovery and handshaking
- Can participate in larger distributed neural network topologies

## Future Enhancements

- **Load Balancing**: Distribute incoming requests across multiple networks
- **Health Monitoring**: Network performance and status reporting
- **Dynamic Topology**: Runtime network topology changes
- **Batch Processing**: Efficient batch activation processing
- **Metrics Collection**: Performance and learning metrics