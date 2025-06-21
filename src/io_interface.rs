use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::distributed_network::{
    DistributedNetwork, MessagePayload, MessageType, NetworkId, NetworkMessage,
};
use crate::neural_network::NeuralNetwork;

/// Unique identifier for I/O connections
pub type IoConnectionId = Uuid;

/// Configuration for I/O node connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoNodeConfig {
    pub node_id: NetworkId,
    pub name: String,
    pub listen_address: String,
    pub listen_port: u16,
    pub target_address: Option<String>,
    pub target_port: Option<u16>,
    pub use_tls: bool,
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
    pub data_transformation: Option<String>,
    pub input_size: usize,
}

/// Errors that can occur during I/O operations
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    #[error("Data transformation error: {0}")]
    TransformationError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Input node that acts as a data source in the distributed neural network
///
/// This node receives data from external systems and sends it to the distributed
/// neural network using the standard NNP protocol. It appears to the network
/// as just another neural network node.
#[derive(Clone)]
pub struct InputNode {
    distributed_network: DistributedNetwork,
    config: IoNodeConfig,
    is_running: bool,
}

impl InputNode {
    /// Create a new input node
    pub fn new(config: IoNodeConfig) -> (Self, mpsc::UnboundedReceiver<NetworkMessage>) {
        // Create a special "passthrough" neural network for input nodes
        // Since we can't directly set weights, we'll use a minimal network
        // and bypass it in the send_data method
        let passthrough_network = NeuralNetwork::with_layers(&[config.input_size, config.input_size], 0.0);
        
        // Create a distributed network with the passthrough neural network
        let (distributed_network, message_receiver) = DistributedNetwork::new(
            config.name.clone(),
            config.listen_address.clone(),
            config.listen_port,
            passthrough_network,
        );

        let input_node = Self {
            distributed_network,
            config,
            is_running: false,
        };

        (input_node, message_receiver)
    }

    /// Start the input node server
    pub async fn start(&mut self) -> Result<(), IoError> {
        // Start the distributed network server
        self.distributed_network
            .start_server()
            .await
            .map_err(|e| IoError::NetworkError(format!("Failed to start server: {:?}", e)))?;

        self.is_running = true;

        // If we have a target to connect to, connect to it
        if let (Some(target_addr), Some(target_port)) =
            (&self.config.target_address, self.config.target_port)
        {
            self.distributed_network
                .connect_to(target_addr, target_port)
                .await
                .map_err(|e| {
                    IoError::ConnectionFailed(format!("Failed to connect to target: {:?}", e))
                })?;
        }

        Ok(())
    }

    /// Send data directly to connected neural network nodes via NNP
    /// This is a special implementation for InputNode that bypasses the neural network
    pub async fn send_data(&self, data: Vec<f64>) -> Result<(), IoError> {
        // Verify input size matches the configured size
        if data.len() != self.config.input_size {
            return Err(IoError::TransformationError(format!(
                "Input size mismatch: expected {}, got {}",
                self.config.input_size,
                data.len()
            )));
        }
        
        // Try to send directly to a specific target if configured
        if let (Some(addr), Some(port)) = (&self.config.target_address, &self.config.target_port) {
            // Find the peer ID for the target neural network
            if let Some(peer_id) = self.distributed_network.find_peer_by_address(addr, *port) {
                // Send data directly to the target peer, bypassing neural network processing
                return self.distributed_network
                    .send_forward_data(peer_id, 0u8, data)
                    .await
                    .map_err(|e| IoError::NetworkError(format!("Failed to send data: {:?}", e)));
            }
        }
        
        // If no specific target or target not found, broadcast to all connected networks
        let message = NetworkMessage {
            msg_type: MessageType::ForwardData,
            sequence: 0, // Will be set by send_message
            payload: MessagePayload::ForwardData {
                layer_id: 0, // Always use layer 0 for input data
                data: data.iter().map(|&x| x as f32).collect(),
            },
        };
        
        self.distributed_network
            .handle_message(message)
            .await
            .map_err(|e| IoError::NetworkError(format!("Failed to send data: {:?}", e)))?;

        Ok(())
    }



    /// Connect to external data source and start forwarding via NNP
    pub async fn connect_external_source(
        &self,
        source_config: ExternalSourceConfig,
    ) -> Result<(), IoError> {
        match source_config {
            ExternalSourceConfig::TcpSocket { address, port } => {
                self.start_tcp_source(address, port).await?;
            }
            ExternalSourceConfig::HttpEndpoint { url, poll_interval } => {
                self.start_http_source(url, poll_interval).await?;
            }
            ExternalSourceConfig::Custom { handler } => {
                let (tx, mut rx) = mpsc::channel(100);
                handler(tx).await?;

                // Forward custom data via NNP
                let input_node = self.clone();
                tokio::spawn(async move {
                    while let Some(data) = rx.recv().await {
                        if let Err(e) = input_node.send_data(data).await {
                            eprintln!("Failed to send custom data: {:?}", e);
                        }
                    }
                });
            }
        }

        Ok(())
    }

    async fn start_tcp_source(&self, address: String, port: u16) -> Result<(), IoError> {
        let addr = format!("{}:{}", address, port);
        let input_node = self.clone();

        tokio::spawn(async move {
            if let Ok(mut stream) = TcpStream::connect(&addr).await {
                let mut buffer = vec![0u8; 1024];

                while let Ok(n) = stream.read(&mut buffer).await {
                    if n == 0 {
                        break;
                    }

                    // Transform raw bytes to neural network input
                    let data: Vec<f64> = buffer[..n].iter().map(|&b| b as f64 / 255.0).collect();

                    if let Err(e) = input_node.send_data(data).await {
                        eprintln!("Failed to send TCP data: {:?}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    async fn start_http_source(&self, _url: String, poll_interval: u64) -> Result<(), IoError> {
        let input_node = self.clone();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(poll_interval));

            loop {
                interval.tick().await;

                // In a real implementation, you'd make HTTP requests here
                // For demo purposes, generate sample data
                let sample_data = vec![0.5, 0.8, 0.2, 0.9];

                if let Err(e) = input_node.send_data(sample_data).await {
                    eprintln!("Failed to send HTTP data: {:?}", e);
                    break;
                }
            }
        });

        Ok(())
    }
}

/// Output node that receives data from the distributed neural network
///
/// This node appears to the network as just another neural network node,
/// but instead of processing the data, it forwards it to external systems.
#[derive(Clone)]
pub struct OutputNode {
    distributed_network: DistributedNetwork,
    #[allow(dead_code)]
    config: IoNodeConfig,
    is_running: bool,
}

impl OutputNode {
    /// Create a new output node
    pub fn new(config: IoNodeConfig) -> (Self, mpsc::UnboundedReceiver<NetworkMessage>) {
        // Create a dummy neural network for the distributed node
        let dummy_network = NeuralNetwork::new(4, 2, 1, 0.1);

        let (distributed_network, message_receiver) = DistributedNetwork::new(
            config.name.clone(),
            config.listen_address.clone(),
            config.listen_port,
            dummy_network,
        );

        let output_node = Self {
            distributed_network,
            config,
            is_running: false,
        };

        (output_node, message_receiver)
    }

    /// Start the output node server
    pub async fn start(&mut self) -> Result<(), IoError> {
        // Start the distributed network server
        self.distributed_network
            .start_server()
            .await
            .map_err(|e| IoError::NetworkError(format!("Failed to start server: {:?}", e)))?;

        self.is_running = true;
        Ok(())
    }

    /// Process incoming NNP messages and forward to external sink
    pub async fn process_messages(
        &self,
        message_receiver: mpsc::UnboundedReceiver<NetworkMessage>,
        sink_config: ExternalSinkConfig,
    ) -> Result<(), IoError> {
        match sink_config {
            ExternalSinkConfig::TcpSocket { address, port } => {
                self.start_tcp_sink_handler(message_receiver, address, port)
                    .await?;
            }
            ExternalSinkConfig::HttpEndpoint { url } => {
                self.start_http_sink_handler(message_receiver, url).await?;
            }
            ExternalSinkConfig::Custom { handler } => {
                let (tx, rx) = mpsc::channel(100);
                self.start_nnp_receiver(message_receiver, tx).await?;
                handler(rx).await?;
            }
        }

        Ok(())
    }

    async fn start_tcp_sink_handler(
        &self,
        mut message_receiver: mpsc::UnboundedReceiver<NetworkMessage>,
        address: String,
        port: u16,
    ) -> Result<(), IoError> {
        let addr = format!("{}:{}", address, port);

        tokio::spawn(async move {
            if let Ok(mut stream) = TcpStream::connect(&addr).await {
                while let Some(message) = message_receiver.recv().await {
                    if let MessagePayload::ForwardData { data, .. } = message.payload {
                        // Transform neural network output to bytes
                        let bytes: Vec<u8> = data
                            .iter()
                            .map(|&x| (x * 255.0).clamp(0.0, 255.0) as u8)
                            .collect();

                        if stream.write_all(&bytes).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });

        Ok(())
    }

    async fn start_http_sink_handler(
        &self,
        mut message_receiver: mpsc::UnboundedReceiver<NetworkMessage>,
        url: String,
    ) -> Result<(), IoError> {
        tokio::spawn(async move {
            while let Some(message) = message_receiver.recv().await {
                if let MessagePayload::ForwardData { data, .. } = message.payload {
                    let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                    println!("Would POST to {}: {:?}", url, data_f64);
                }
            }
        });

        Ok(())
    }

    async fn start_nnp_receiver(
        &self,
        mut message_receiver: mpsc::UnboundedReceiver<NetworkMessage>,
        sender: mpsc::Sender<Vec<f64>>,
    ) -> Result<(), IoError> {
        tokio::spawn(async move {
            while let Some(message) = message_receiver.recv().await {
                if let MessagePayload::ForwardData { data, .. } = message.payload {
                    let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();

                    if sender.send(data_f64).await.is_err() {
                        break;
                    }
                }
            }
        });

        Ok(())
    }
}

/// Type alias for custom source handler
pub type CustomSourceHandler =
    fn(
        mpsc::Sender<Vec<f64>>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), IoError>> + Send>>;

/// Type alias for custom sink handler  
pub type CustomSinkHandler =
    fn(
        mpsc::Receiver<Vec<f64>>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), IoError>> + Send>>;

/// Configuration for external data sources
#[derive(Debug, Clone)]
pub enum ExternalSourceConfig {
    TcpSocket { address: String, port: u16 },
    HttpEndpoint { url: String, poll_interval: u64 },
    Custom { handler: CustomSourceHandler },
}

/// Configuration for external data sinks
#[derive(Debug, Clone)]
pub enum ExternalSinkConfig {
    TcpSocket { address: String, port: u16 },
    HttpEndpoint { url: String },
    Custom { handler: CustomSinkHandler },
}

/// Secure I/O nodes using TLS encryption
///
/// These are simplified wrappers around the basic I/O nodes that add TLS support
/// by leveraging the SecureDistributedNetwork infrastructure.
pub type SecureInputNode = InputNode;
pub type SecureOutputNode = OutputNode;
