use crate::neural_network::NeuralNetwork;
use crate::distributed_network::{DistributedNetwork, NetworkMessage, MessagePayload, ProtocolError};
use std::path::PathBuf;
use tokio::sync::mpsc;
use uuid::Uuid;
use log::{info, warn, error, debug};

/// Server configuration for neural network daemon
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub name: String,
    pub address: String,
    pub port: u16,
    pub cert_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
    pub output_endpoints: Vec<String>,
    pub hebbian_learning: bool,
    pub daemon_mode: bool,
}

/// Neural network server using existing distributed network infrastructure
pub struct NetworkServer {
    distributed_network: DistributedNetwork,
    config: ServerConfig,
    message_receiver: mpsc::UnboundedReceiver<NetworkMessage>,
}

impl NetworkServer {
    pub fn new(
        network: NeuralNetwork,
        config: ServerConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Create distributed network
        let (distributed_network, message_receiver) = DistributedNetwork::new(
            config.name.clone(),
            config.address.clone(),
            config.port,
            network,
        );

        Ok(Self {
            distributed_network,
            config,
            message_receiver,
        })
    }

    /// Start the server using existing distributed network infrastructure
    pub async fn start(mut self) -> Result<(), ProtocolError> {
        info!("🚀 Starting Neural Network Server");
        info!("   Name: {}", self.config.name);
        info!("   Address: {}:{}", self.config.address, self.config.port);
        info!("   Network ID: {}", self.distributed_network.id);
        info!("   Hebbian Learning: {}", self.config.hebbian_learning);
        info!("   Output Endpoints: {:?}", self.config.output_endpoints);
        
        if self.config.daemon_mode {
            info!("   Running in daemon mode");
        }

        // Start the distributed network server
        info!("🌐 Starting neural network server");
        if self.config.cert_path.is_some() && self.config.key_path.is_some() {
            warn!("⚠️  SSL/TLS configuration detected but using basic server for now");
        }
        
        self.distributed_network.start_server().await?;

        // Start message processing loop
        self.start_message_processing().await?;

        Ok(())
    }

    /// Start message processing loop
    async fn start_message_processing(&mut self) -> Result<(), ProtocolError> {
        info!("📡 Starting message processing loop");
        
        while let Some(message) = self.message_receiver.recv().await {
            if let Err(e) = self.process_message(message).await {
                error!("Error processing message: {:?}", e);
            }
        }
        
        Ok(())
    }

    /// Process incoming network messages
    async fn process_message(&self, message: NetworkMessage) -> Result<(), ProtocolError> {
        debug!("Processing message: {:?}", message.msg_type);
        
        match message.payload {
            MessagePayload::ForwardData { layer_id, data } => {
                self.handle_forward_data(layer_id, data).await?;
            }
            MessagePayload::HebbianData { layer_id, correlations, learning_rate } => {
                if self.config.hebbian_learning {
                    self.handle_hebbian_data(layer_id, correlations, learning_rate).await?;
                }
            }
            MessagePayload::Handshake { network_id, name, layers, capabilities } => {
                self.handle_handshake(network_id, name, layers, capabilities).await?;
            }
            MessagePayload::Heartbeat { timestamp } => {
                debug!("Received heartbeat at timestamp: {}", timestamp);
            }
            _ => {
                debug!("Unhandled message type: {:?}", message.msg_type);
            }
        }
        
        Ok(())
    }

    /// Handle forward data (neural network activation)
    async fn handle_forward_data(&self, layer_id: u8, data: Vec<f32>) -> Result<(), ProtocolError> {
        let start_time = std::time::Instant::now();
        
        // Convert f32 to f64 for neural network processing
        let inputs: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        
        info!("📥 Received activation for layer {} with {} inputs", layer_id, inputs.len());
        
        // Process through neural network
        let outputs = {
            let mut network = self.distributed_network.network.lock().unwrap();
            let (outputs, _hidden) = network.forward(&inputs);
            
            // Apply Hebbian learning if enabled
            if self.config.hebbian_learning {
                network.hebbian_update(&inputs);
                info!("🧠 Applied Hebbian learning update");
            }
            
            outputs
        };
        
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        info!("⚡ Processed in {:.2}ms, outputs: {:?}", processing_time, outputs);
        
        // Forward outputs to connected networks
        self.forward_outputs(&outputs).await?;
        
        Ok(())
    }

    /// Handle Hebbian learning data
    async fn handle_hebbian_data(&self, layer_id: u8, _correlations: Vec<f32>, learning_rate: f32) -> Result<(), ProtocolError> {
        info!("🧠 Received Hebbian data for layer {} (rate: {})", layer_id, learning_rate);
        
        // Apply Hebbian correlations to the network
        let _network = self.distributed_network.network.lock().unwrap();
        // Note: This would need to be implemented in the neural network
        // network.apply_hebbian_correlations(layer_id, &correlations, learning_rate);
        
        Ok(())
    }

    /// Handle handshake from new network
    async fn handle_handshake(&self, network_id: Uuid, name: String, layers: Vec<u16>, capabilities: u32) -> Result<(), ProtocolError> {
        info!("🤝 Handshake from network '{}' (ID: {})", name, network_id);
        info!("   Layers: {:?}", layers);
        info!("   Capabilities: 0x{:08X}", capabilities);
        
        // The distributed network will handle the handshake response automatically
        
        Ok(())
    }

    /// Forward outputs to connected networks using NNP protocol
    async fn forward_outputs(&self, outputs: &[f64]) -> Result<(), ProtocolError> {
        if self.config.output_endpoints.is_empty() {
            return Ok(());
        }
        
        // For now, just log the forwarding intent
        // In a full implementation, you'd use the distributed network's send methods
        for endpoint in &self.config.output_endpoints {
            info!("📤 Would forward {} outputs to {}", outputs.len(), endpoint);
            debug!("   Outputs: {:?}", outputs);
        }
        
        Ok(())
    }
}

/// Run server in daemon mode using existing distributed network infrastructure
pub async fn run_daemon(
    network: NeuralNetwork,
    config: ServerConfig,
) -> Result<(), ProtocolError> {
    // Initialize logging
    env_logger::init();
    
    info!("🔄 Starting neural network daemon...");
    
    if config.daemon_mode {
        info!("🔄 Running in daemon mode");
        // In a real implementation, you'd want to:
        // 1. Fork the process
        // 2. Detach from terminal
        // 3. Set up proper logging to files
        // 4. Handle signals for graceful shutdown
        // 5. Create PID file
    }
    
    // Create and start the server
    let server = NetworkServer::new(network, config)
        .map_err(|_| ProtocolError::IoError(std::io::Error::other("Failed to create server")))?;
    server.start().await?;
    
    Ok(())
}

/// Create a client for connecting to other neural networks using NNP protocol
pub struct NetworkClient {
    distributed_network: DistributedNetwork,
}

impl NetworkClient {
    pub fn new(name: String, address: String, port: u16, network: NeuralNetwork) -> Self {
        let (distributed_network, _) = DistributedNetwork::new(name, address, port, network);
        
        Self {
            distributed_network,
        }
    }

    /// Connect to a remote neural network
    pub async fn connect_to_network(
        &self,
        target_address: &str,
        target_port: u16,
    ) -> Result<Uuid, ProtocolError> {
        info!("🔗 Connecting to neural network at {}:{}", target_address, target_port);
        
        // Use the existing distributed network connection functionality
        let peer_id = self.distributed_network.connect_to(target_address, target_port).await?;
        
        Ok(peer_id)
    }

    /// Send forward data to connected networks
    pub async fn send_forward_data(
        &self,
        peer_id: Uuid,
        layer_id: u8,
        data: Vec<f64>,
    ) -> Result<(), ProtocolError> {
        // Use the existing distributed network send method
        self.distributed_network.send_forward_data(peer_id, layer_id, data).await
    }

    /// Send Hebbian learning data
    pub async fn send_hebbian_data(
        &self,
        peer_id: Uuid,
        layer_id: u8,
        correlations: Vec<f64>,
        learning_rate: f64,
    ) -> Result<(), ProtocolError> {
        // Use the existing distributed network send method
        self.distributed_network.send_hebbian_data(peer_id, layer_id, correlations, learning_rate).await
    }
}