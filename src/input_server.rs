use crate::io_interface::{InputNode, IoNodeConfig};
use futures_util::{SinkExt, StreamExt};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{RwLock, mpsc};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use uuid::Uuid;

/// Configuration for the InputServer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputServerConfig {
    /// Address to bind the web server
    pub web_address: String,
    /// Port for the web server
    pub web_port: u16,
    /// Port for WebSocket connections
    pub websocket_port: u16,
    /// Neural network configurations to connect to
    pub neural_networks: Vec<NeuralNetworkTarget>,
    /// SSL certificate path (optional)
    pub cert_path: Option<String>,
    /// SSL key path (optional)
    pub key_path: Option<String>,
}

/// Configuration for a target neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkTarget {
    /// Unique identifier for this network
    pub id: String,
    /// Display name
    pub name: String,
    /// Network address
    pub address: String,
    /// Network port
    pub port: u16,
    /// Number of inputs this network expects
    pub input_count: usize,
    /// Use TLS for connection
    pub use_tls: bool,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    /// Client requests list of available networks
    GetNetworks,
    /// Server responds with network list
    NetworkList { networks: Vec<NetworkInfo> },
    /// Client sends input activation
    ActivateInput {
        network_id: String,
        inputs: Vec<f64>,
    },
    /// Server confirms input activation
    InputActivated {
        network_id: String,
        success: bool,
        message: String,
    },
    /// Server sends status update
    StatusUpdate { network_id: String, status: String },
    /// Error message
    Error { message: String },
}

/// Network information for the web interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub id: String,
    pub name: String,
    pub address: String,
    pub port: u16,
    pub input_count: usize,
    pub connected: bool,
    pub use_tls: bool,
}

/// InputServer manages web interface and neural network connections
pub struct InputServer {
    config: InputServerConfig,
    input_nodes: Arc<RwLock<HashMap<String, InputNode>>>,
    network_status: Arc<RwLock<HashMap<String, bool>>>,
    websocket_clients: Arc<RwLock<Vec<mpsc::UnboundedSender<WebSocketMessage>>>>,
}

impl InputServer {
    /// Create a new InputServer
    pub fn new(config: InputServerConfig) -> Self {
        Self {
            config,
            input_nodes: Arc::new(RwLock::new(HashMap::new())),
            network_status: Arc::new(RwLock::new(HashMap::new())),
            websocket_clients: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start the InputServer
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting InputServer...");

        // Initialize neural network connections
        self.initialize_networks().await?;

        // Start WebSocket server
        let websocket_server = self.start_websocket_server();

        // Start HTTP server
        let http_server = self.start_http_server();

        // Run both servers concurrently
        tokio::try_join!(websocket_server, http_server)?;

        Ok(())
    }

    /// Initialize connections to neural networks
    async fn initialize_networks(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut input_nodes = self.input_nodes.write().await;
        let mut network_status = self.network_status.write().await;

        for target in &self.config.neural_networks {
            println!(
                "🔗 Connecting to neural network: {} ({}:{})",
                target.name, target.address, target.port
            );

            let node_config = IoNodeConfig {
                node_id: Uuid::new_v4(),
                name: format!("InputServer-{}", target.name),
                listen_address: "127.0.0.1".to_string(),
                listen_port: 0, // Let the system assign a port
                target_address: Some(target.address.clone()),
                target_port: Some(target.port),
                use_tls: target.use_tls,
                cert_path: self.config.cert_path.clone(),
                key_path: self.config.key_path.clone(),
                data_transformation: None,
            };

            let (mut input_node, _receiver) = InputNode::new(node_config);

            match input_node.start().await {
                Ok(_) => {
                    println!("✅ Connected to {}", target.name);
                    input_nodes.insert(target.id.clone(), input_node);
                    network_status.insert(target.id.clone(), true);
                }
                Err(e) => {
                    println!("❌ Failed to connect to {}: {:?}", target.name, e);
                    network_status.insert(target.id.clone(), false);
                }
            }
        }

        Ok(())
    }

    /// Start the WebSocket server
    async fn start_websocket_server(&self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("{}:{}", self.config.web_address, self.config.websocket_port);
        let listener = TcpListener::bind(&addr).await?;
        println!("🌐 WebSocket server listening on {}", addr);

        let input_nodes = Arc::clone(&self.input_nodes);
        let network_status = Arc::clone(&self.network_status);
        let websocket_clients = Arc::clone(&self.websocket_clients);
        let config = self.config.clone();

        while let Ok((stream, addr)) = listener.accept().await {
            let input_nodes = Arc::clone(&input_nodes);
            let network_status = Arc::clone(&network_status);
            let websocket_clients = Arc::clone(&websocket_clients);
            let config = config.clone();

            tokio::spawn(async move {
                if let Err(e) = Self::handle_websocket_connection(
                    stream,
                    addr,
                    input_nodes,
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
        input_nodes: Arc<RwLock<HashMap<String, InputNode>>>,
        network_status: Arc<RwLock<HashMap<String, bool>>>,
        websocket_clients: Arc<RwLock<Vec<mpsc::UnboundedSender<WebSocketMessage>>>>,
        config: InputServerConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔌 New WebSocket connection from {}", addr);

        let ws_stream = accept_async(stream).await?;
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();

        let (tx, mut rx) = mpsc::unbounded_channel::<WebSocketMessage>();

        // Add client to the list
        {
            let mut clients = websocket_clients.write().await;
            clients.push(tx.clone());
        }

        // Send initial network list
        let networks = Self::get_network_info(&config, &network_status).await;
        let network_list_msg = WebSocketMessage::NetworkList { networks };
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
        let input_nodes_clone = Arc::clone(&input_nodes);
        let tx_clone = tx.clone();
        let ws_receiver_task = tokio::spawn(async move {
            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if let Ok(ws_msg) = serde_json::from_str::<WebSocketMessage>(&text) {
                            Self::handle_websocket_message(ws_msg, &input_nodes_clone, &tx_clone)
                                .await;
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
        message: WebSocketMessage,
        input_nodes: &Arc<RwLock<HashMap<String, InputNode>>>,
        tx: &mpsc::UnboundedSender<WebSocketMessage>,
    ) {
        match message {
            WebSocketMessage::ActivateInput { network_id, inputs } => {
                let mut nodes = input_nodes.write().await;
                if let Some(node) = nodes.get_mut(&network_id) {
                    match node.send_data(inputs.clone()).await {
                        Ok(_) => {
                            let response = WebSocketMessage::InputActivated {
                                network_id: network_id.clone(),
                                success: true,
                                message: format!("Successfully sent {} inputs", inputs.len()),
                            };
                            let _ = tx.send(response);
                            println!("📤 Sent inputs to {}: {:?}", network_id, inputs);
                        }
                        Err(e) => {
                            let response = WebSocketMessage::InputActivated {
                                network_id: network_id.clone(),
                                success: false,
                                message: format!("Failed to send inputs: {:?}", e),
                            };
                            let _ = tx.send(response);
                            println!("❌ Failed to send inputs to {}: {:?}", network_id, e);
                        }
                    }
                } else {
                    let response = WebSocketMessage::Error {
                        message: format!("Network {} not found", network_id),
                    };
                    let _ = tx.send(response);
                }
            }
            _ => {}
        }
    }

    /// Get network information
    async fn get_network_info(
        config: &InputServerConfig,
        network_status: &Arc<RwLock<HashMap<String, bool>>>,
    ) -> Vec<NetworkInfo> {
        let status = network_status.read().await;
        config
            .neural_networks
            .iter()
            .map(|target| NetworkInfo {
                id: target.id.clone(),
                name: target.name.clone(),
                address: target.address.clone(),
                port: target.port,
                input_count: target.input_count,
                connected: status.get(&target.id).copied().unwrap_or(false),
                use_tls: target.use_tls,
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
    fn serve_index_html(_websocket_port: u16) -> Response<Body> {
        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Input Server</title>
    <link rel="stylesheet" href="/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Neural Network Input Server</h1>
            <div id="connection-status" class="status disconnected">Disconnected</div>
        </header>
        
        <main>
            <div id="networks-container">
                <h2>Connected Neural Networks</h2>
                <div id="networks-list"></div>
            </div>
            
            <div id="input-panel">
                <h2>Input Activation</h2>
                <div id="selected-network">
                    <p>Select a neural network to activate inputs</p>
                </div>
                <div id="input-controls" style="display: none;">
                    <div id="input-sliders"></div>
                    <div class="controls">
                        <button id="send-inputs" class="btn-primary">Send Inputs</button>
                        <button id="reset-inputs" class="btn-secondary">Reset</button>
                        <button id="random-inputs" class="btn-secondary">Random</button>
                    </div>
                </div>
            </div>
            
            <div id="log-panel">
                <h2>Activity Log</h2>
                <div id="log-container"></div>
                <button id="clear-log" class="btn-secondary">Clear Log</button>
            </div>
        </main>
    </div>
    
    <script src="/script.js"></script>
</body>
</html>"#
        );

        Response::builder()
            .header("Content-Type", "text/html")
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
    max-width: 1200px;
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

h1 {
    color: #4a5568;
    font-size: 2rem;
}

h2 {
    color: #2d3748;
    margin-bottom: 15px;
    font-size: 1.5rem;
}

.status {
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    text-transform: uppercase;
    font-size: 0.8rem;
}

.status.connected {
    background: #48bb78;
    color: white;
}

.status.disconnected {
    background: #f56565;
    color: white;
}

main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 20px;
}

#networks-container {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

#input-panel {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

#log-panel {
    grid-column: 1 / -1;
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.network-card {
    background: #f7fafc;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.network-card:hover {
    border-color: #4299e1;
    transform: translateY(-2px);
}

.network-card.selected {
    border-color: #3182ce;
    background: #ebf8ff;
}

.network-card.disconnected {
    opacity: 0.6;
    cursor: not-allowed;
}

.network-name {
    font-weight: bold;
    font-size: 1.1rem;
    color: #2d3748;
}

.network-details {
    color: #718096;
    font-size: 0.9rem;
    margin-top: 5px;
}

.network-status {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: bold;
    text-transform: uppercase;
    margin-top: 5px;
}

.network-status.connected {
    background: #c6f6d5;
    color: #22543d;
}

.network-status.disconnected {
    background: #fed7d7;
    color: #742a2a;
}

.input-slider {
    margin-bottom: 15px;
}

.input-slider label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
    color: #4a5568;
}

.slider-container {
    display: flex;
    align-items: center;
    gap: 10px;
}

.slider {
    flex: 1;
    height: 6px;
    border-radius: 3px;
    background: #e2e8f0;
    outline: none;
    -webkit-appearance: none;
}

.slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #4299e1;
    cursor: pointer;
}

.slider::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #4299e1;
    cursor: pointer;
    border: none;
}

.slider-value {
    min-width: 60px;
    text-align: center;
    font-weight: bold;
    color: #2d3748;
}

.controls {
    margin-top: 20px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.btn-primary, .btn-secondary {
    padding: 10px 20px;
    border: none;
    border-radius: 6px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-primary {
    background: #4299e1;
    color: white;
}

.btn-primary:hover {
    background: #3182ce;
}

.btn-secondary {
    background: #e2e8f0;
    color: #4a5568;
}

.btn-secondary:hover {
    background: #cbd5e0;
}

#log-container {
    background: #1a202c;
    color: #e2e8f0;
    padding: 15px;
    border-radius: 6px;
    height: 200px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    margin-bottom: 10px;
}

.log-entry {
    margin-bottom: 5px;
    padding: 2px 0;
}

.log-entry.success {
    color: #68d391;
}

.log-entry.error {
    color: #fc8181;
}

.log-entry.info {
    color: #63b3ed;
}

@media (max-width: 768px) {
    main {
        grid-template-columns: 1fr;
    }
    
    .controls {
        justify-content: center;
    }
}
"#;

        Response::builder()
            .header("Content-Type", "text/css")
            .body(Body::from(css))
            .unwrap()
    }

    /// Serve JavaScript
    fn serve_js(websocket_port: u16) -> Response<Body> {
        let js = format!(
            r#"
class InputServerClient {{
    constructor() {{
        this.ws = null;
        this.networks = [];
        this.selectedNetwork = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.initializeUI();
        this.connect();
    }}
    
    connect() {{
        const wsUrl = `ws://${{window.location.hostname}}:{websocket_port}`;
        console.log('Connecting to:', wsUrl);
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {{
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
            this.reconnectAttempts = 0;
        }};
        
        this.ws.onmessage = (event) => {{
            try {{
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            }} catch (e) {{
                console.error('Failed to parse message:', e);
            }}
        }};
        
        this.ws.onclose = () => {{
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            this.attemptReconnect();
        }};
        
        this.ws.onerror = (error) => {{
            console.error('WebSocket error:', error);
            this.addLogEntry('WebSocket connection error', 'error');
        }};
    }}
    
    attemptReconnect() {{
        if (this.reconnectAttempts < this.maxReconnectAttempts) {{
            this.reconnectAttempts++;
            this.addLogEntry(`Reconnecting... (attempt ${{this.reconnectAttempts}})`, 'info');
            setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
        }} else {{
            this.addLogEntry('Max reconnection attempts reached', 'error');
        }}
    }}
    
    handleMessage(message) {{
        console.log('Received message:', message);
        
        switch (message.type) {{
            case 'NetworkList':
                this.networks = message.networks;
                this.updateNetworksList();
                this.addLogEntry(`Loaded ${{message.networks.length}} neural networks`, 'info');
                break;
                
            case 'InputActivated':
                if (message.success) {{
                    this.addLogEntry(`✅ ${{message.message}} (Network: ${{message.network_id}})`, 'success');
                }} else {{
                    this.addLogEntry(`❌ ${{message.message}} (Network: ${{message.network_id}})`, 'error');
                }}
                break;
                
            case 'StatusUpdate':
                this.addLogEntry(`📊 ${{message.status}} (Network: ${{message.network_id}})`, 'info');
                break;
                
            case 'Error':
                this.addLogEntry(`❌ Error: ${{message.message}}`, 'error');
                break;
        }}
    }}
    
    sendMessage(message) {{
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {{
            this.ws.send(JSON.stringify(message));
        }} else {{
            this.addLogEntry('Cannot send message: WebSocket not connected', 'error');
        }}
    }}
    
    updateConnectionStatus(connected) {{
        const statusElement = document.getElementById('connection-status');
        if (connected) {{
            statusElement.textContent = 'Connected';
            statusElement.className = 'status connected';
        }} else {{
            statusElement.textContent = 'Disconnected';
            statusElement.className = 'status disconnected';
        }}
    }}
    
    updateNetworksList() {{
        const container = document.getElementById('networks-list');
        container.innerHTML = '';
        
        this.networks.forEach(network => {{
            const card = document.createElement('div');
            card.className = `network-card ${{network.connected ? '' : 'disconnected'}}`;
            card.dataset.networkId = network.id;
            
            card.innerHTML = `
                <div class="network-name">${{network.name}}</div>
                <div class="network-details">
                    ${{network.address}}:${{network.port}} | ${{network.input_count}} inputs
                    ${{network.use_tls ? '🔒 TLS' : '🔓 Plain'}}
                </div>
                <div class="network-status ${{network.connected ? 'connected' : 'disconnected'}}">
                    ${{network.connected ? 'Connected' : 'Disconnected'}}
                </div>
            `;
            
            if (network.connected) {{
                card.addEventListener('click', () => this.selectNetwork(network));
            }}
            
            container.appendChild(card);
        }});
    }}
    
    selectNetwork(network) {{
        this.selectedNetwork = network;
        
        // Update UI selection
        document.querySelectorAll('.network-card').forEach(card => {{
            card.classList.remove('selected');
        }});
        document.querySelector(`[data-network-id="${{network.id}}"]`).classList.add('selected');
        
        // Update input panel
        this.updateInputPanel();
        this.addLogEntry(`Selected network: ${{network.name}}`, 'info');
    }}
    
    updateInputPanel() {{
        const selectedNetworkDiv = document.getElementById('selected-network');
        const inputControls = document.getElementById('input-controls');
        
        if (this.selectedNetwork) {{
            selectedNetworkDiv.innerHTML = `
                <h3>${{this.selectedNetwork.name}}</h3>
                <p>${{this.selectedNetwork.input_count}} inputs | ${{this.selectedNetwork.address}}:${{this.selectedNetwork.port}}</p>
            `;
            
            this.createInputSliders();
            inputControls.style.display = 'block';
        }} else {{
            selectedNetworkDiv.innerHTML = '<p>Select a neural network to activate inputs</p>';
            inputControls.style.display = 'none';
        }}
    }}
    
    createInputSliders() {{
        const container = document.getElementById('input-sliders');
        container.innerHTML = '';
        
        for (let i = 0; i < this.selectedNetwork.input_count; i++) {{
            const sliderDiv = document.createElement('div');
            sliderDiv.className = 'input-slider';
            
            sliderDiv.innerHTML = `
                <label for="input-${{i}}">Input ${{i + 1}}</label>
                <div class="slider-container">
                    <input type="range" id="input-${{i}}" class="slider" min="0" max="1" step="0.01" value="0">
                    <span class="slider-value" id="value-${{i}}">0.00</span>
                </div>
            `;
            
            const slider = sliderDiv.querySelector('.slider');
            const valueSpan = sliderDiv.querySelector('.slider-value');
            
            slider.addEventListener('input', (e) => {{
                valueSpan.textContent = parseFloat(e.target.value).toFixed(2);
            }});
            
            container.appendChild(sliderDiv);
        }}
    }}
    
    getInputValues() {{
        const values = [];
        for (let i = 0; i < this.selectedNetwork.input_count; i++) {{
            const slider = document.getElementById(`input-${{i}}`);
            values.push(parseFloat(slider.value));
        }}
        return values;
    }}
    
    setInputValues(values) {{
        for (let i = 0; i < Math.min(values.length, this.selectedNetwork.input_count); i++) {{
            const slider = document.getElementById(`input-${{i}}`);
            const valueSpan = document.getElementById(`value-${{i}}`);
            slider.value = values[i];
            valueSpan.textContent = values[i].toFixed(2);
        }}
    }}
    
    sendInputs() {{
        if (!this.selectedNetwork) {{
            this.addLogEntry('No network selected', 'error');
            return;
        }}
        
        const inputs = this.getInputValues();
        this.sendMessage({{
            type: 'ActivateInput',
            network_id: this.selectedNetwork.id,
            inputs: inputs
        }});
        
        this.addLogEntry(`📤 Sending inputs: [${{inputs.map(v => v.toFixed(2)).join(', ')}}]`, 'info');
    }}
    
    resetInputs() {{
        if (!this.selectedNetwork) return;
        
        const zeros = new Array(this.selectedNetwork.input_count).fill(0);
        this.setInputValues(zeros);
        this.addLogEntry('Reset all inputs to 0', 'info');
    }}
    
    randomizeInputs() {{
        if (!this.selectedNetwork) return;
        
        const randomValues = Array.from({{length: this.selectedNetwork.input_count}}, () => Math.random());
        this.setInputValues(randomValues);
        this.addLogEntry('Randomized all inputs', 'info');
    }}
    
    addLogEntry(message, type = 'info') {{
        const container = document.getElementById('log-container');
        const entry = document.createElement('div');
        entry.className = `log-entry ${{type}}`;
        
        const timestamp = new Date().toLocaleTimeString();
        entry.textContent = `[${{timestamp}}] ${{message}}`;
        
        container.appendChild(entry);
        container.scrollTop = container.scrollHeight;
        
        // Keep only last 100 entries
        while (container.children.length > 100) {{
            container.removeChild(container.firstChild);
        }}
    }}
    
    clearLog() {{
        document.getElementById('log-container').innerHTML = '';
        this.addLogEntry('Log cleared', 'info');
    }}
    
    initializeUI() {{
        // Send inputs button
        document.getElementById('send-inputs').addEventListener('click', () => {{
            this.sendInputs();
        }});
        
        // Reset inputs button
        document.getElementById('reset-inputs').addEventListener('click', () => {{
            this.resetInputs();
        }});
        
        // Random inputs button
        document.getElementById('random-inputs').addEventListener('click', () => {{
            this.randomizeInputs();
        }});
        
        // Clear log button
        document.getElementById('clear-log').addEventListener('click', () => {{
            this.clearLog();
        }});
        
        // Initial log entry
        this.addLogEntry('InputServer client initialized', 'info');
    }}
}}

// Initialize the client when the page loads
document.addEventListener('DOMContentLoaded', () => {{
    window.inputServerClient = new InputServerClient();
}});
"#,
            websocket_port = websocket_port
        );

        Response::builder()
            .header("Content-Type", "application/javascript")
            .body(Body::from(js))
            .unwrap()
    }
}
