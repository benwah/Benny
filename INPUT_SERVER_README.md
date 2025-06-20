# InputServer - Web Interface for Neural Network Input Control

The InputServer provides a web-based interface for manually controlling inputs to distributed neural networks. It features real-time WebSocket communication, SSL/TLS support, and an intuitive web interface.

## Features

- **Web Interface**: Interactive HTML/CSS/JavaScript interface with real-time input controls
- **WebSocket Communication**: Real-time bidirectional communication between web interface and server
- **Neural Network Integration**: Connects to distributed neural networks via SSL/TLS
- **Multiple Network Support**: Can connect to multiple neural network targets
- **Real-time Logging**: Live status updates and activity logging
- **Input Controls**: Sliders, send/reset/randomize buttons for input manipulation

## Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐    SSL/TLS    ┌─────────────────┐
│   Web Browser   │ ◄──────────────► │   InputServer   │ ◄────────────► │ Neural Network  │
│                 │                 │                 │               │     Server      │
│ • HTML/CSS/JS   │                 │ • HTTP Server   │               │                 │
│ • Input Sliders │                 │ • WebSocket     │               │ • Distributed   │
│ • Real-time UI  │                 │ • NN Client     │               │   Network       │
└─────────────────┘                 └─────────────────┘               └─────────────────┘
```

## Quick Start

### 1. Build the Project

```bash
cargo build --example neural_network_server --bin input_server
```

### 2. Start Neural Network Server

```bash
cargo run --example neural_network_server
```

This starts a neural network server on `127.0.0.1:8001` with a 4-input, 2-hidden, 1-output network.

### 3. Start InputServer

```bash
cargo run --bin input_server -- \
    --network-host 127.0.0.1 \
    --network-port 8001 \
    --web-port 3000 \
    --websocket-port 3001 \
    --input-size 4
```

### 4. Open Web Interface

Navigate to `http://127.0.0.1:3000` in your browser to access the input control interface.

### 5. Automated Test

Use the provided test script for a complete demonstration:

```bash
./test_input_server.sh
```

## Command Line Options

### InputServer Binary

```bash
cargo run --bin input_server -- [OPTIONS]
```

**Options:**
- `--network-host <HOST>`: Neural network server hostname (default: 127.0.0.1)
- `--network-port <PORT>`: Neural network server port (default: 8001)
- `--web-port <PORT>`: Web server port (default: 3000)
- `--websocket-port <PORT>`: WebSocket server port (default: 3001)
- `--input-size <SIZE>`: Number of neural network inputs (default: 4)
- `--use-tls`: Enable TLS for neural network connection
- `--cert-path <PATH>`: Path to TLS certificate file
- `--key-path <PATH>`: Path to TLS private key file

## Web Interface

The web interface provides:

### Input Controls
- **Sliders**: Adjust individual input values (0.0 to 1.0)
- **Value Displays**: Real-time display of current input values
- **Input Labels**: Clear labeling of each input (Input 0, Input 1, etc.)

### Action Buttons
- **Send to Network**: Transmit current input values to neural network
- **Reset All**: Reset all inputs to 0.0
- **Randomize**: Set all inputs to random values

### Status Display
- **Connection Status**: Shows connection state to neural network
- **Activity Log**: Real-time log of actions and network communication
- **Timestamp**: Last update timestamp

### Real-time Features
- **Live Updates**: Input changes are reflected immediately
- **WebSocket Status**: Connection indicator for WebSocket communication
- **Error Handling**: User-friendly error messages and recovery

## API Endpoints

### HTTP API

- `GET /`: Serve main HTML interface
- `POST /api/input/{index}/{value}`: Set specific input value
- `GET /api/status`: Get current system status

### WebSocket API

**Message Format:**
```json
{
  "type": "message_type",
  "data": { ... }
}
```

**Message Types:**
- `input_command`: Update input value
- `state_update`: Broadcast current state
- `status_update`: System status changes
- `error`: Error notifications

## Configuration

### Multiple Neural Networks

The InputServer can connect to multiple neural network targets:

```rust
let config = InputServerConfig {
    web_address: "127.0.0.1".to_string(),
    web_port: 3000,
    websocket_port: 3001,
    neural_networks: vec![
        NeuralNetworkTarget {
            id: "network1".to_string(),
            name: "Primary Network".to_string(),
            address: "127.0.0.1".to_string(),
            port: 8001,
            input_count: 4,
            use_tls: false,
        },
        NeuralNetworkTarget {
            id: "network2".to_string(),
            name: "Secondary Network".to_string(),
            address: "192.168.1.100".to_string(),
            port: 8002,
            input_count: 8,
            use_tls: true,
        },
    ],
    cert_path: Some("certs/client.crt".to_string()),
    key_path: Some("certs/client.key".to_string()),
};
```

### SSL/TLS Configuration

For secure connections to neural networks:

1. **Generate Certificates**:
   ```bash
   mkdir certs
   openssl req -x509 -newkey rsa:4096 -keyout certs/client.key -out certs/client.crt -days 365 -nodes
   ```

2. **Enable TLS**:
   ```bash
   cargo run --bin input_server -- \
       --use-tls \
       --cert-path certs/client.crt \
       --key-path certs/client.key
   ```

## Development

### Project Structure

```
src/
├── input_server.rs          # Main InputServer implementation
├── bin/
│   └── input_server.rs      # Binary executable
└── lib.rs                   # Module exports

examples/
└── neural_network_server.rs # Example neural network server

test_input_server.sh         # Integration test script
```

### Key Components

1. **InputServer**: Main server struct handling HTTP, WebSocket, and neural network connections
2. **WebSocket Handler**: Real-time communication with web interface
3. **HTTP Server**: Serves static content and API endpoints
4. **Neural Network Client**: Connects to distributed neural networks
5. **Configuration System**: Flexible configuration for multiple targets

### Testing

Run the complete test suite:

```bash
# Unit tests
cargo test

# Integration test
./test_input_server.sh

# Manual testing
cargo run --example neural_network_server &
cargo run --bin input_server
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Check what's using the port
   lsof -i :3000
   
   # Use different ports
   cargo run --bin input_server -- --web-port 3001 --websocket-port 3002
   ```

2. **Neural Network Connection Failed**:
   - Ensure neural network server is running
   - Check host/port configuration
   - Verify SSL/TLS settings if using encryption

3. **WebSocket Connection Issues**:
   - Check browser console for errors
   - Verify WebSocket port is accessible
   - Ensure firewall allows connections

4. **Build Errors**:
   ```bash
   # Clean and rebuild
   cargo clean
   cargo build
   ```

### Logging

Enable detailed logging:

```bash
RUST_LOG=debug cargo run --bin input_server
```

Log levels: `error`, `warn`, `info`, `debug`, `trace`

## Future Enhancements

- **Authentication**: User authentication and authorization
- **Persistence**: Save/load input configurations
- **Visualization**: Real-time neural network activity visualization
- **Batch Operations**: Send multiple input sets
- **Recording**: Record and replay input sequences
- **Mobile Interface**: Responsive design for mobile devices
- **Plugin System**: Extensible input sources and processors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the Benny neural network library and follows the same licensing terms.