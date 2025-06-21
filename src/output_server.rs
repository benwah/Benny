use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use futures_util::{SinkExt, StreamExt};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio_tungstenite::{accept_async, tungstenite::Message};


/// Configuration for the OutputServer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputServerConfig {
    /// Address to bind the web server
    pub web_address: String,
    /// Port for the web server
    pub web_port: u16,
    /// Port for WebSocket connections
    pub websocket_port: u16,
    /// Expected output size from neural networks
    pub expected_output_size: usize,
    /// Neural network configurations to listen for
    pub neural_networks: Vec<NeuralNetworkSource>,
    /// SSL certificate path (optional)
    pub cert_path: Option<String>,
    /// SSL key path (optional)
    pub key_path: Option<String>,
}

/// Configuration for a source neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkSource {
    /// Unique identifier for this network
    pub id: String,
    /// Display name
    pub name: String,
    /// Network address to listen on
    pub listen_address: String,
    /// Network port to listen on
    pub listen_port: u16,
    /// Number of outputs this network produces
    pub output_count: usize,
    /// Use TLS for connection
    pub use_tls: bool,
}

/// WebSocket message types for output display
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputWebSocketMessage {
    /// Client requests list of available networks
    GetNetworks,
    /// Server responds with network list
    NetworkList { networks: Vec<OutputNetworkInfo> },
    /// Server sends output activation data
    OutputData {
        network_id: String,
        outputs: Vec<f64>,
        timestamp: u64,
    },
    /// Server sends status update
    StatusUpdate { network_id: String, status: String },
    /// Error message
    Error { message: String },
}

/// Network information for the web interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputNetworkInfo {
    pub id: String,
    pub name: String,
    pub listen_address: String,
    pub listen_port: u16,
    pub output_count: usize,
    pub connected: bool,
    pub use_tls: bool,
}

/// OutputServer manages web interface and neural network output display
pub struct OutputServer {
    config: OutputServerConfig,
    network_status: Arc<RwLock<HashMap<String, bool>>>,
    websocket_clients: Arc<RwLock<Vec<mpsc::UnboundedSender<OutputWebSocketMessage>>>>,
}

impl OutputServer {
    /// Create a new OutputServer
    pub fn new(config: OutputServerConfig) -> Self {
        Self {
            config,
            network_status: Arc::new(RwLock::new(HashMap::new())),
            websocket_clients: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start the OutputServer
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting OutputServer...");

        // Start TCP server for neural network connections
        let tcp_server = self.start_tcp_server();

        // Start WebSocket server
        let websocket_server = self.start_websocket_server();

        // Start HTTP server
        let http_server = self.start_http_server();

        // Run all servers concurrently
        tokio::try_join!(tcp_server, websocket_server, http_server)?;

        Ok(())
    }

    /// Start TCP server for neural network connections
    async fn start_tcp_server(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Use the first neural network source for the TCP server configuration
        let source = &self.config.neural_networks[0];
        let addr = format!("{}:{}", source.listen_address, source.listen_port);
        let listener = TcpListener::bind(&addr).await?;
        println!("🔗 TCP server listening on {} for neural network connections", addr);

        let websocket_clients = Arc::clone(&self.websocket_clients);
        let expected_output_size = self.config.expected_output_size;

        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    println!("🔗 New neural network connection from {}", addr);
                    let websocket_clients = Arc::clone(&websocket_clients);
                    
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_neural_network_connection(
                            stream, 
                            "main-network".to_string(), // Use consistent network ID
                            websocket_clients,
                            expected_output_size,
                        ).await {
                            println!("❌ Error handling connection from {}: {:?}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    println!("❌ Failed to accept connection: {:?}", e);
                }
            }
        }
    }

    /// Handle a connection from a neural network
    async fn handle_neural_network_connection(
        mut stream: TcpStream,
        network_id: String,
        websocket_clients: Arc<RwLock<Vec<mpsc::UnboundedSender<OutputWebSocketMessage>>>>,
        expected_output_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("📡 Started handling connection from: {}", network_id);
        
        let mut buffer = vec![0; 1024];
        
        loop {
            match stream.read(&mut buffer).await {
                Ok(0) => {
                    println!("🔌 Connection closed by {}", network_id);
                    break;
                }
                Ok(n) => {
                    // Parse the received data as JSON
                    let data_str = String::from_utf8_lossy(&buffer[..n]);
                    
                    // Try to parse as a simple JSON array of numbers
                    if let Ok(outputs) = serde_json::from_str::<Vec<f64>>(&data_str.trim()) {
                        if outputs.len() == expected_output_size {
                            let timestamp = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64;

                            let output_message = OutputWebSocketMessage::OutputData {
                                network_id: network_id.clone(),
                                outputs: outputs.clone(),
                                timestamp,
                            };

                            // Broadcast to all connected WebSocket clients
                            let clients = websocket_clients.read().await;
                            for client in clients.iter() {
                                let _ = client.send(output_message.clone());
                            }

                            println!("📊 Received output data from {}: {:?}", network_id, outputs);
                            
                            // Send acknowledgment
                            let ack = "OK\n";
                            let _ = stream.write_all(ack.as_bytes()).await;
                        } else {
                            println!("⚠️ Invalid output size from {}: expected {}, got {}", 
                                network_id, expected_output_size, outputs.len());
                        }
                    } else {
                        println!("⚠️ Invalid data format from {}: {}", network_id, data_str);
                    }
                }
                Err(e) => {
                    println!("❌ Error reading from {}: {:?}", network_id, e);
                    break;
                }
            }
        }
        
        println!("📡 Connection handler for {} ended", network_id);
        Ok(())
    }

    /// Start the WebSocket server
    async fn start_websocket_server(&self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("{}:{}", self.config.web_address, self.config.websocket_port);
        let listener = TcpListener::bind(&addr).await?;
        println!("🌐 WebSocket server listening on {}", addr);

        let network_status = Arc::clone(&self.network_status);
        let websocket_clients = Arc::clone(&self.websocket_clients);
        let config = self.config.clone();

        while let Ok((stream, addr)) = listener.accept().await {
            let network_status = Arc::clone(&network_status);
            let websocket_clients = Arc::clone(&websocket_clients);
            let config = config.clone();

            tokio::spawn(async move {
                if let Err(e) = Self::handle_websocket_connection(
                    stream,
                    addr,
                    network_status,
                    websocket_clients,
                    config,
                )
                .await
                {
                    println!("WebSocket error: {:?}", e);
                }
            });
        }

        Ok(())
    }

    /// Handle a WebSocket connection
    async fn handle_websocket_connection(
        stream: TcpStream,
        addr: SocketAddr,
        network_status: Arc<RwLock<HashMap<String, bool>>>,
        websocket_clients: Arc<RwLock<Vec<mpsc::UnboundedSender<OutputWebSocketMessage>>>>,
        config: OutputServerConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔌 New WebSocket connection from {}", addr);

        let ws_stream = accept_async(stream).await?;
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();

        let (tx, mut rx) = mpsc::unbounded_channel::<OutputWebSocketMessage>();

        // Add client to the list
        {
            let mut clients = websocket_clients.write().await;
            clients.push(tx.clone());
        }

        // Send initial network list
        let networks = if let Some(network) = config.neural_networks.first() {
            vec![OutputNetworkInfo {
                id: network.id.clone(),
                name: network.name.clone(),
                listen_address: network.listen_address.clone(),
                listen_port: network.listen_port,
                output_count: network.output_count,
                connected: true, // Assume connected for simplicity
                use_tls: network.use_tls,
            }]
        } else {
            vec![]
        };
        let network_list_msg = OutputWebSocketMessage::NetworkList { networks };
        if let Ok(msg_text) = serde_json::to_string(&network_list_msg) {
            let _ = ws_sender.send(Message::Text(msg_text)).await;
        }

        // Handle outgoing messages
        let ws_sender_task = tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                if let Ok(msg_text) = serde_json::to_string(&message) {
                    if ws_sender.send(Message::Text(msg_text)).await.is_err() {
                        break;
                    }
                }
            }
        });

        // Handle incoming messages
        let tx_clone = tx.clone();
        let ws_receiver_task = tokio::spawn(async move {
            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if let Ok(ws_msg) = serde_json::from_str::<OutputWebSocketMessage>(&text) {
                            Self::handle_websocket_message(ws_msg, &tx_clone).await;
                        }
                    }
                    Ok(Message::Close(_)) => break,
                    Err(_) => break,
                    _ => {}
                }
            }
        });

        // Wait for either task to complete
        tokio::select! {
            _ = ws_sender_task => {},
            _ = ws_receiver_task => {},
        }

        // Remove client from the list
        {
            let mut clients = websocket_clients.write().await;
            clients.retain(|client| !client.is_closed());
        }

        println!("🔌 WebSocket connection from {} closed", addr);
        Ok(())
    }

    /// Handle WebSocket messages
    async fn handle_websocket_message(
        message: OutputWebSocketMessage,
        _tx: &mpsc::UnboundedSender<OutputWebSocketMessage>,
    ) {
        match message {
            OutputWebSocketMessage::GetNetworks => {
                // This would be handled by sending the network list on connection
                // For now, we don't need to handle this specifically
            }
            _ => {}
        }
    }

    /// Get network information
    async fn get_network_info(
        config: &OutputServerConfig,
        network_status: &Arc<RwLock<HashMap<String, bool>>>,
    ) -> Vec<OutputNetworkInfo> {
        let status = network_status.read().await;
        config
            .neural_networks
            .iter()
            .map(|source| OutputNetworkInfo {
                id: source.id.clone(),
                name: source.name.clone(),
                listen_address: source.listen_address.clone(),
                listen_port: source.listen_port,
                output_count: source.output_count,
                connected: status.get(&source.id).copied().unwrap_or(false),
                use_tls: source.use_tls,
            })
            .collect()
    }



    /// Start the HTTP server for serving the web interface
    async fn start_http_server(&self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("{}:{}", self.config.web_address, self.config.web_port);
        let websocket_port = self.config.websocket_port;

        let make_svc = make_service_fn(move |_conn| {
            let websocket_port = websocket_port;
            async move {
                Ok::<_, Infallible>(service_fn(move |req| {
                    Self::handle_http_request(req, websocket_port)
                }))
            }
        });

        let server = Server::bind(&addr.parse()?).serve(make_svc);
        println!("🌐 HTTP server listening on http://{}", addr);

        server.await?;
        Ok(())
    }



    /// Handle HTTP requests
    async fn handle_http_request(
        req: Request<Body>,
        websocket_port: u16,
    ) -> Result<Response<Body>, Infallible> {
        match req.uri().path() {
            "/" => Ok(Self::serve_index_html(websocket_port)),
            "/style.css" => Ok(Self::serve_css()),
            "/script.js" => Ok(Self::serve_js(websocket_port)),
            _ => Ok(Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from("Not Found"))
                .unwrap()),
        }
    }

    /// Serve the main HTML page
    fn serve_index_html(websocket_port: u16) -> Response<Body> {
        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Output Monitor</title>
    <link rel="stylesheet" href="/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Neural Network Output Monitor</h1>
            <div id="connection-status" class="status disconnected">Disconnected</div>
        </header>
        
        <main>
            <div id="networks-container">
                <h2>Monitored Neural Networks</h2>
                <div id="networks-list"></div>
            </div>
            
            <div id="output-panel">
                <h2>Real-time Output Display</h2>
                <div id="output-visualizations"></div>
            </div>
            
            <div id="log-panel">
                <h2>Activity Log</h2>
                <div id="log-container"></div>
                <button id="clear-log" class="btn-secondary">Clear Log</button>
            </div>
        </main>
    </div>
    
    <script>
        const WEBSOCKET_PORT = {websocket_port};
    </script>
    <script src="/script.js"></script>
</body>
</html>"#
        );

        Response::builder()
            .header("Content-Type", "text/html")
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            .header("Access-Control-Allow-Headers", "Content-Type")
            .body(Body::from(html))
            .unwrap()
    }

    /// Serve CSS styles
    fn serve_css() -> Response<Body> {
        let css = r#"
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

header {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

header h1 {
    color: #333;
    font-size: 2.2em;
}

.status {
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9em;
}

.status.connected {
    background: #4CAF50;
    color: white;
}

.status.disconnected {
    background: #f44336;
    color: white;
}

main {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 20px;
    margin-bottom: 20px;
}

#networks-container, #output-panel {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

#output-panel {
    grid-column: span 2;
}

h2 {
    color: #333;
    margin-bottom: 15px;
    border-bottom: 2px solid #667eea;
    padding-bottom: 5px;
}

.network-item {
    background: #f8f9fa;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}

.network-item.connected {
    border-left-color: #4CAF50;
}

.network-item.disconnected {
    border-left-color: #f44336;
}

.network-name {
    font-weight: bold;
    font-size: 1.1em;
    margin-bottom: 5px;
}

.network-details {
    font-size: 0.9em;
    color: #666;
}

.network-status {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: bold;
    margin-top: 5px;
}

.network-status.connected {
    background: #4CAF50;
    color: white;
}

.network-status.disconnected {
    background: #f44336;
    color: white;
}

.output-visualization {
    background: #f8f9fa;
    padding: 20px;
    margin: 15px 0;
    border-radius: 8px;
    border: 1px solid #ddd;
}

.output-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.output-title {
    font-weight: bold;
    font-size: 1.2em;
    color: #333;
}

.output-timestamp {
    font-size: 0.9em;
    color: #666;
}

.output-bars {
    display: flex;
    gap: 8px;
    height: 120px;
    align-items: flex-end;
    margin: 15px 0;
    padding: 10px;
    background: white;
    border-radius: 5px;
}

.output-bar {
    flex: 1;
    background: linear-gradient(to top, #667eea, #764ba2);
    border-radius: 4px 4px 0 0;
    transition: height 0.3s ease;
    min-height: 3px;
    position: relative;
    min-width: 20px;
}

.output-bar-label {
    position: absolute;
    bottom: -25px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.8em;
    color: #666;
    white-space: nowrap;
}

.output-values {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
    gap: 10px;
    margin-top: 15px;
}

.output-value {
    text-align: center;
    padding: 8px;
    background: white;
    border-radius: 5px;
    border: 1px solid #ddd;
}

.output-value-label {
    font-size: 0.8em;
    color: #666;
    margin-bottom: 3px;
}

.output-value-number {
    font-weight: bold;
    font-size: 1.1em;
    color: #333;
}

#log-panel {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    grid-column: span 2;
}

#log-container {
    max-height: 200px;
    overflow-y: auto;
    background: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    font-family: monospace;
    font-size: 0.9em;
}

.log-entry {
    padding: 3px 0;
    border-bottom: 1px solid #eee;
}

.log-entry:last-child {
    border-bottom: none;
}

.log-timestamp {
    color: #666;
    margin-right: 10px;
}

.btn-secondary {
    background: #6c757d;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 0.9em;
    transition: background-color 0.2s;
}

.btn-secondary:hover {
    background: #5a6268;
}

.no-data {
    text-align: center;
    color: #666;
    font-style: italic;
    padding: 40px;
}

@media (max-width: 1024px) {
    main {
        grid-template-columns: 1fr;
    }
    
    #output-panel {
        grid-column: span 1;
    }
    
    #log-panel {
        grid-column: span 1;
    }
}

@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    header {
        flex-direction: column;
        gap: 10px;
        text-align: center;
    }
    
    header h1 {
        font-size: 1.8em;
    }
    
    .output-values {
        grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
    }
}
"#;

        Response::builder()
            .header("Content-Type", "text/css")
            .header("Access-Control-Allow-Origin", "*")
            .body(Body::from(css))
            .unwrap()
    }

    /// Serve JavaScript
    fn serve_js(_websocket_port: u16) -> Response<Body> {
        let js = format!(
            r#"
class OutputMonitor {{
    constructor() {{
        this.ws = null;
        this.networks = new Map();
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.outputData = new Map(); // Store latest output data for each network
        
        this.init();
    }}

    init() {{
        this.connectWebSocket();
        this.setupEventListeners();
        this.log('OutputMonitor initialized');
    }}

    connectWebSocket() {{
        // Force ws:// for development (since our WebSocket server doesn't support TLS)
        const protocol = 'ws:';
        // Use work-2 domain for WebSocket (port 12001 maps to work-2)
        const wsUrl = `${{protocol}}//localhost:12001`;
        
        this.log(`Connecting to WebSocket: ${{wsUrl}}`);
        
        try {{
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {{
                this.log('WebSocket connected');
                this.connected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus();
            }};
            
            this.ws.onmessage = (event) => {{
                try {{
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                }} catch (e) {{
                    this.log(`Error parsing message: ${{e.message}}`);
                }}
            }};
            
            this.ws.onclose = () => {{
                this.log('WebSocket disconnected');
                this.connected = false;
                this.updateConnectionStatus();
                this.attemptReconnect();
            }};
            
            this.ws.onerror = (error) => {{
                this.log(`WebSocket error: ${{error}}`);
            }};
        }} catch (e) {{
            this.log(`Failed to create WebSocket: ${{e.message}}`);
            this.attemptReconnect();
        }}
    }}

    attemptReconnect() {{
        if (this.reconnectAttempts < this.maxReconnectAttempts) {{
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            this.log(`Attempting to reconnect in ${{delay/1000}}s (attempt ${{this.reconnectAttempts}}/${{this.maxReconnectAttempts}})`);
            
            setTimeout(() => {{
                this.connectWebSocket();
            }}, delay);
        }} else {{
            this.log('Max reconnection attempts reached. Please refresh the page.');
        }}
    }}

    handleMessage(message) {{
        switch (message.type) {{
            case 'NetworkList':
                this.updateNetworkList(message.networks);
                break;
            case 'OutputData':
                this.updateOutputData(message.network_id, message.outputs, message.timestamp);
                break;
            case 'StatusUpdate':
                this.updateNetworkStatus(message.network_id, message.status);
                break;
            case 'Error':
                this.log(`Error: ${{message.message}}`);
                break;
            default:
                this.log(`Unknown message type: ${{message.type}}`);
        }}
    }}

    updateNetworkList(networks) {{
        this.networks.clear();
        networks.forEach(network => {{
            this.networks.set(network.id, network);
        }});
        this.renderNetworkList();
        this.renderOutputVisualizations();
    }}

    updateOutputData(networkId, outputs, timestamp) {{
        this.outputData.set(networkId, {{
            outputs: outputs,
            timestamp: timestamp,
            lastUpdate: new Date()
        }});
        
        this.updateOutputVisualization(networkId);
        this.log(`Received output data from ${{this.networks.get(networkId)?.name || networkId}}: [${{outputs.map(v => v.toFixed(3)).join(', ')}}]`);
    }}

    updateNetworkStatus(networkId, status) {{
        const network = this.networks.get(networkId);
        if (network) {{
            network.connected = status === 'connected';
            this.renderNetworkList();
        }}
        this.log(`Network ${{networkId}} status: ${{status}}`);
    }}

    updateConnectionStatus() {{
        const statusElement = document.getElementById('connection-status');
        if (this.connected) {{
            statusElement.textContent = 'Connected';
            statusElement.className = 'status connected';
        }} else {{
            statusElement.textContent = 'Disconnected';
            statusElement.className = 'status disconnected';
        }}
    }}

    renderNetworkList() {{
        const container = document.getElementById('networks-list');
        container.innerHTML = '';
        
        if (this.networks.size === 0) {{
            container.innerHTML = '<div class="no-data">No neural networks configured</div>';
            return;
        }}
        
        this.networks.forEach(network => {{
            const networkDiv = document.createElement('div');
            networkDiv.className = `network-item ${{network.connected ? 'connected' : 'disconnected'}}`;
            
            networkDiv.innerHTML = `
                <div class="network-name">${{network.name}}</div>
                <div class="network-details">
                    ${{network.listen_address}}:${{network.listen_port}} | 
                    ${{network.output_count}} outputs | 
                    ${{network.use_tls ? 'TLS' : 'Plain'}}
                </div>
                <div class="network-status ${{network.connected ? 'connected' : 'disconnected'}}">
                    ${{network.connected ? 'Connected' : 'Disconnected'}}
                </div>
            `;
            
            container.appendChild(networkDiv);
        }});
    }}

    renderOutputVisualizations() {{
        const container = document.getElementById('output-visualizations');
        container.innerHTML = '';
        
        if (this.networks.size === 0) {{
            container.innerHTML = '<div class="no-data">No neural networks to monitor</div>';
            return;
        }}
        
        this.networks.forEach(network => {{
            const vizDiv = document.createElement('div');
            vizDiv.className = 'output-visualization';
            vizDiv.id = `viz-${{network.id}}`;
            
            vizDiv.innerHTML = `
                <div class="output-header">
                    <div class="output-title">${{network.name}}</div>
                    <div class="output-timestamp" id="timestamp-${{network.id}}">No data yet</div>
                </div>
                <div class="output-bars" id="bars-${{network.id}}">
                    ${{Array.from({{length: network.output_count}}, (_, i) => `
                        <div class="output-bar" id="bar-${{network.id}}-${{i}}" style="height: 3px;">
                            <div class="output-bar-label">O${{i}}</div>
                        </div>
                    `).join('')}}
                </div>
                <div class="output-values" id="values-${{network.id}}">
                    ${{Array.from({{length: network.output_count}}, (_, i) => `
                        <div class="output-value">
                            <div class="output-value-label">Output ${{i}}</div>
                            <div class="output-value-number" id="value-${{network.id}}-${{i}}">0.000</div>
                        </div>
                    `).join('')}}
                </div>
            `;
            
            container.appendChild(vizDiv);
        }});
    }}

    updateOutputVisualization(networkId) {{
        const data = this.outputData.get(networkId);
        const network = this.networks.get(networkId);
        
        if (!data || !network) return;
        
        // Update timestamp
        const timestampElement = document.getElementById(`timestamp-${{networkId}}`);
        if (timestampElement) {{
            timestampElement.textContent = `Last update: ${{data.lastUpdate.toLocaleTimeString()}}`;
        }}
        
        // Update bars and values
        data.outputs.forEach((value, index) => {{
            const barElement = document.getElementById(`bar-${{networkId}}-${{index}}`);
            const valueElement = document.getElementById(`value-${{networkId}}-${{index}}`);
            
            if (barElement) {{
                const height = Math.max(3, value * 100); // Minimum 3px height
                barElement.style.height = `${{height}}%`;
            }}
            
            if (valueElement) {{
                valueElement.textContent = value.toFixed(3);
                
                // Color coding based on value
                if (value > 0.7) {{
                    valueElement.style.color = '#4CAF50'; // Green for high values
                }} else if (value > 0.3) {{
                    valueElement.style.color = '#FF9800'; // Orange for medium values
                }} else {{
                    valueElement.style.color = '#666'; // Gray for low values
                }}
            }}
        }});
    }}

    setupEventListeners() {{
        const clearLogBtn = document.getElementById('clear-log');
        if (clearLogBtn) {{
            clearLogBtn.addEventListener('click', () => {{
                this.clearLog();
            }});
        }}
    }}

    log(message) {{
        const logContainer = document.getElementById('log-container');
        if (!logContainer) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `<span class="log-timestamp">${{timestamp}}</span>${{message}}`;
        
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
        
        // Keep only last 100 log entries
        while (logContainer.children.length > 100) {{
            logContainer.removeChild(logContainer.firstChild);
        }}
    }}

    clearLog() {{
        const logContainer = document.getElementById('log-container');
        if (logContainer) {{
            logContainer.innerHTML = '';
        }}
    }}
}}

// Initialize the output monitor when the page loads
let outputMonitor;
document.addEventListener('DOMContentLoaded', () => {{
    outputMonitor = new OutputMonitor();
}});
"#
        );

        Response::builder()
            .header("Content-Type", "application/javascript")
            .header("Access-Control-Allow-Origin", "*")
            .body(Body::from(js))
            .unwrap()
    }
}
