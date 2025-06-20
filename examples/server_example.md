# Neural Network Server Example

This example demonstrates how to set up and use the neural network server mode.

## Setup

1. **Create a configuration file**:
```bash
neural_network init-config --output server_config.json --network-type feedforward
```

2. **Start the server**:
```bash
neural_network server --config server_config.json --port 8080 --hebbian-learning
```

## Expected Output

```
🚀 Starting Neural Network Server
=================================
✅ Loaded configuration from: server_config.json
🆕 Creating new network from configuration
✅ Network ready: Neural Network: 2 -> 4 -> 1 (Hebbian rate: 0.05, mode: Classic)
   Parameters: 17
   Hebbian Learning: true
⚠️  Running without SSL/TLS encryption
🌐 Server will listen on: 0.0.0.0:8080
📡 Using Neural Network Protocol (NNP)

🚀 Neural Network Protocol server listening on 0.0.0.0:8080
📡 Network ID: 486865a9-048f-4992-acb0-2952e34efdc0
🧠 Capabilities: 0x00000077
```

## Testing the Server

The server uses the Neural Network Protocol (NNP) for communication. You can:

1. **Connect another neural network** as a client
2. **Send ForwardData messages** with neural activations
3. **Receive processed outputs** from the network
4. **Apply Hebbian learning** automatically on incoming data

## Advanced Usage

### With SSL/TLS

```bash
# Generate self-signed certificates (for testing)
openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes

# Start secure server
neural_network server \
  --config server_config.json \
  --port 8080 \
  --cert server.crt \
  --key server.key \
  --hebbian-learning
```

### With Output Forwarding

```bash
# Start multiple servers
neural_network server --config server1_config.json --port 8080 --hebbian-learning &
neural_network server --config server2_config.json --port 8081 --hebbian-learning &

# Start a server that forwards to others
neural_network server \
  --config main_config.json \
  --port 8082 \
  --outputs "localhost:8080" "localhost:8081" \
  --hebbian-learning
```

### As Daemon

```bash
# Run in background
neural_network server \
  --config server_config.json \
  --port 8080 \
  --daemon \
  --hebbian-learning

# Check if running
ps aux | grep neural_network
```

## Protocol Messages

The server processes these NNP message types:

1. **ForwardData**: Neural network activations
   - Automatically processed through the network
   - Outputs forwarded to connected endpoints
   - Hebbian learning applied if enabled

2. **HebbianData**: Learning correlations
   - Applied to network weights (future enhancement)

3. **Handshake**: Network identification
   - Automatic response with network capabilities

## Integration

The server can be integrated into larger distributed neural network topologies:

- **Chained Processing**: Output of one network becomes input to another
- **Parallel Processing**: Multiple networks process the same input
- **Hierarchical Networks**: Networks at different abstraction levels
- **Learning Networks**: Networks that learn from each other's outputs

## Monitoring

The server provides detailed logging:

- Connection events
- Message processing
- Hebbian learning updates
- Performance metrics
- Error handling

Use `RUST_LOG=debug` for detailed logging:

```bash
RUST_LOG=debug neural_network server --config server_config.json --port 8080 --hebbian-learning
```