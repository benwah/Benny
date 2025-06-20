use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_rustls::{TlsConnector, client::TlsStream};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use async_trait::async_trait;

use crate::neural_network::NeuralNetwork;

/// Unique identifier for I/O connections
pub type IoConnectionId = Uuid;

/// Data format for neural network I/O
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoData {
    pub timestamp: u64,
    pub values: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

/// Configuration for I/O connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoConfig {
    pub connection_id: IoConnectionId,
    pub endpoint: String,
    pub port: u16,
    pub use_tls: bool,
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
    pub buffer_size: usize,
    pub timeout_ms: u64,
}

/// Errors that can occur during I/O operations
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    #[error("Data format error: {0}")]
    DataFormatError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Timeout error")]
    Timeout,
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Trait for neural network input interfaces
/// 
/// This trait defines how external data sources can connect to neural network inputs.
/// Implementations can handle various input types like sensors, databases, APIs, etc.
#[async_trait]
pub trait InputInterface: Send + Sync {
    /// Connect to an external data source
    async fn connect(&mut self, config: IoConfig) -> Result<(), IoError>;
    
    /// Disconnect from the data source
    async fn disconnect(&mut self) -> Result<(), IoError>;
    
    /// Read data from the connected source
    async fn read_data(&mut self) -> Result<IoData, IoError>;
    
    /// Start streaming data (non-blocking)
    async fn start_streaming(&mut self, sender: mpsc::Sender<IoData>) -> Result<(), IoError>;
    
    /// Stop streaming data
    async fn stop_streaming(&mut self) -> Result<(), IoError>;
    
    /// Check if the connection is active
    fn is_connected(&self) -> bool;
    
    /// Get connection information
    fn get_connection_info(&self) -> Option<IoConfig>;
    
    /// Transform raw data into neural network input format
    fn transform_input(&self, raw_data: &[u8]) -> Result<Vec<f64>, IoError>;
}

/// Trait for neural network output interfaces
/// 
/// This trait defines how neural network outputs can connect to external systems.
/// Implementations can handle various output types like actuators, databases, APIs, etc.
#[async_trait]
pub trait OutputInterface: Send + Sync {
    /// Connect to an external system
    async fn connect(&mut self, config: IoConfig) -> Result<(), IoError>;
    
    /// Disconnect from the external system
    async fn disconnect(&mut self) -> Result<(), IoError>;
    
    /// Write data to the connected system
    async fn write_data(&mut self, data: IoData) -> Result<(), IoError>;
    
    /// Start streaming data to the system (non-blocking)
    async fn start_streaming(&mut self, mut receiver: mpsc::Receiver<IoData>) -> Result<(), IoError>;
    
    /// Stop streaming data
    async fn stop_streaming(&mut self) -> Result<(), IoError>;
    
    /// Check if the connection is active
    fn is_connected(&self) -> bool;
    
    /// Get connection information
    fn get_connection_info(&self) -> Option<IoConfig>;
    
    /// Transform neural network output into external system format
    fn transform_output(&self, nn_output: &[f64]) -> Result<Vec<u8>, IoError>;
}

/// TCP-based input interface using existing SSL/TCP protocols
pub struct TcpInputInterface {
    connection_id: Option<IoConnectionId>,
    config: Option<IoConfig>,
    stream: Option<TlsStream<TcpStream>>,
    is_streaming: bool,
    tls_connector: Option<TlsConnector>,
}

impl TcpInputInterface {
    pub fn new() -> Self {
        Self {
            connection_id: None,
            config: None,
            stream: None,
            is_streaming: false,
            tls_connector: None,
        }
    }
    
    pub fn with_tls_connector(tls_connector: TlsConnector) -> Self {
        Self {
            connection_id: None,
            config: None,
            stream: None,
            is_streaming: false,
            tls_connector: Some(tls_connector),
        }
    }
}

#[async_trait]
impl InputInterface for TcpInputInterface {
    async fn connect(&mut self, config: IoConfig) -> Result<(), IoError> {
        let addr = format!("{}:{}", config.endpoint, config.port);
        let tcp_stream = TcpStream::connect(&addr).await
            .map_err(|e| IoError::ConnectionFailed(e.to_string()))?;
        
        if config.use_tls {
            if let Some(connector) = &self.tls_connector {
                let domain = rustls::ServerName::try_from(config.endpoint.as_str())
                    .map_err(|e| IoError::ConfigError(e.to_string()))?;
                
                let tls_stream = connector.connect(domain, tcp_stream).await
                    .map_err(|e| IoError::ConnectionFailed(e.to_string()))?;
                
                self.stream = Some(tls_stream);
            } else {
                return Err(IoError::ConfigError("TLS requested but no connector provided".to_string()));
            }
        } else {
            return Err(IoError::ConfigError("Non-TLS connections not implemented yet".to_string()));
        }
        
        self.connection_id = Some(config.connection_id);
        self.config = Some(config);
        
        Ok(())
    }
    
    async fn disconnect(&mut self) -> Result<(), IoError> {
        if let Some(stream) = self.stream.take() {
            let (_, mut writer) = tokio::io::split(stream);
            let _ = writer.shutdown().await;
        }
        self.connection_id = None;
        self.config = None;
        self.is_streaming = false;
        Ok(())
    }
    
    async fn read_data(&mut self) -> Result<IoData, IoError> {
        if let Some(stream) = &mut self.stream {
            let mut buffer = vec![0u8; 1024];
            let n = stream.read(&mut buffer).await
                .map_err(|e| IoError::NetworkError(e.to_string()))?;
            
            if n == 0 {
                return Err(IoError::NetworkError("Connection closed".to_string()));
            }
            
            buffer.truncate(n);
            let values = self.transform_input(&buffer)?;
            
            Ok(IoData {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                values,
                metadata: HashMap::new(),
            })
        } else {
            Err(IoError::ConnectionFailed("Not connected".to_string()))
        }
    }
    
    async fn start_streaming(&mut self, sender: mpsc::Sender<IoData>) -> Result<(), IoError> {
        if self.is_streaming {
            return Err(IoError::ConfigError("Already streaming".to_string()));
        }
        
        self.is_streaming = true;
        
        // Spawn a task to continuously read data
        let stream = self.stream.take()
            .ok_or_else(|| IoError::ConnectionFailed("Not connected".to_string()))?;
        
        tokio::spawn(async move {
            let mut stream = stream;
            let mut buffer = vec![0u8; 1024];
            
            while let Ok(n) = stream.read(&mut buffer).await {
                if n == 0 {
                    break;
                }
                
                // Simple transformation for demo - convert bytes to f64
                let values: Vec<f64> = buffer[..n].iter()
                    .map(|&b| b as f64 / 255.0)
                    .collect();
                
                let data = IoData {
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                    values,
                    metadata: HashMap::new(),
                };
                
                if sender.send(data).await.is_err() {
                    break;
                }
            }
        });
        
        Ok(())
    }
    
    async fn stop_streaming(&mut self) -> Result<(), IoError> {
        self.is_streaming = false;
        Ok(())
    }
    
    fn is_connected(&self) -> bool {
        self.stream.is_some()
    }
    
    fn get_connection_info(&self) -> Option<IoConfig> {
        self.config.clone()
    }
    
    fn transform_input(&self, raw_data: &[u8]) -> Result<Vec<f64>, IoError> {
        // Default transformation: normalize bytes to [0, 1] range
        Ok(raw_data.iter().map(|&b| b as f64 / 255.0).collect())
    }
}

/// TCP-based output interface using existing SSL/TCP protocols
pub struct TcpOutputInterface {
    connection_id: Option<IoConnectionId>,
    config: Option<IoConfig>,
    stream: Option<TlsStream<TcpStream>>,
    is_streaming: bool,
    tls_connector: Option<TlsConnector>,
}

impl TcpOutputInterface {
    pub fn new() -> Self {
        Self {
            connection_id: None,
            config: None,
            stream: None,
            is_streaming: false,
            tls_connector: None,
        }
    }
    
    pub fn with_tls_connector(tls_connector: TlsConnector) -> Self {
        Self {
            connection_id: None,
            config: None,
            stream: None,
            is_streaming: false,
            tls_connector: Some(tls_connector),
        }
    }
}

#[async_trait]
impl OutputInterface for TcpOutputInterface {
    async fn connect(&mut self, config: IoConfig) -> Result<(), IoError> {
        let addr = format!("{}:{}", config.endpoint, config.port);
        let tcp_stream = TcpStream::connect(&addr).await
            .map_err(|e| IoError::ConnectionFailed(e.to_string()))?;
        
        if config.use_tls {
            if let Some(connector) = &self.tls_connector {
                let domain = rustls::ServerName::try_from(config.endpoint.as_str())
                    .map_err(|e| IoError::ConfigError(e.to_string()))?;
                
                let tls_stream = connector.connect(domain, tcp_stream).await
                    .map_err(|e| IoError::ConnectionFailed(e.to_string()))?;
                
                self.stream = Some(tls_stream);
            } else {
                return Err(IoError::ConfigError("TLS requested but no connector provided".to_string()));
            }
        } else {
            return Err(IoError::ConfigError("Non-TLS connections not implemented yet".to_string()));
        }
        
        self.connection_id = Some(config.connection_id);
        self.config = Some(config);
        
        Ok(())
    }
    
    async fn disconnect(&mut self) -> Result<(), IoError> {
        if let Some(stream) = self.stream.take() {
            let (_, mut writer) = tokio::io::split(stream);
            let _ = writer.shutdown().await;
        }
        self.connection_id = None;
        self.config = None;
        self.is_streaming = false;
        Ok(())
    }
    
    async fn write_data(&mut self, data: IoData) -> Result<(), IoError> {
        let output_bytes = self.transform_output(&data.values)?;
        if let Some(stream) = &mut self.stream {
            stream.write_all(&output_bytes).await
                .map_err(|e| IoError::NetworkError(e.to_string()))?;
            Ok(())
        } else {
            Err(IoError::ConnectionFailed("Not connected".to_string()))
        }
    }
    
    async fn start_streaming(&mut self, mut receiver: mpsc::Receiver<IoData>) -> Result<(), IoError> {
        if self.is_streaming {
            return Err(IoError::ConfigError("Already streaming".to_string()));
        }
        
        self.is_streaming = true;
        
        // Spawn a task to continuously write data
        let stream = self.stream.take()
            .ok_or_else(|| IoError::ConnectionFailed("Not connected".to_string()))?;
        
        tokio::spawn(async move {
            let mut stream = stream;
            
            while let Some(data) = receiver.recv().await {
                // Simple transformation for demo - convert f64 to bytes
                let output_bytes: Vec<u8> = data.values.iter()
                    .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
                    .collect();
                
                if stream.write_all(&output_bytes).await.is_err() {
                    break;
                }
            }
        });
        
        Ok(())
    }
    
    async fn stop_streaming(&mut self) -> Result<(), IoError> {
        self.is_streaming = false;
        Ok(())
    }
    
    fn is_connected(&self) -> bool {
        self.stream.is_some()
    }
    
    fn get_connection_info(&self) -> Option<IoConfig> {
        self.config.clone()
    }
    
    fn transform_output(&self, nn_output: &[f64]) -> Result<Vec<u8>, IoError> {
        // Default transformation: convert [0, 1] range to bytes
        Ok(nn_output.iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect())
    }
}

/// Manager for neural network I/O connections
pub struct IoManager {
    input_interfaces: HashMap<IoConnectionId, Box<dyn InputInterface>>,
    output_interfaces: HashMap<IoConnectionId, Box<dyn OutputInterface>>,
    neural_network: Arc<Mutex<NeuralNetwork>>,
}

impl IoManager {
    pub fn new(neural_network: NeuralNetwork) -> Self {
        Self {
            input_interfaces: HashMap::new(),
            output_interfaces: HashMap::new(),
            neural_network: Arc::new(Mutex::new(neural_network)),
        }
    }
    
    /// Add an input interface
    pub fn add_input_interface(&mut self, id: IoConnectionId, interface: Box<dyn InputInterface>) {
        self.input_interfaces.insert(id, interface);
    }
    
    /// Add an output interface
    pub fn add_output_interface(&mut self, id: IoConnectionId, interface: Box<dyn OutputInterface>) {
        self.output_interfaces.insert(id, interface);
    }
    
    /// Connect an input interface
    pub async fn connect_input(&mut self, id: IoConnectionId, config: IoConfig) -> Result<(), IoError> {
        if let Some(interface) = self.input_interfaces.get_mut(&id) {
            interface.connect(config).await
        } else {
            Err(IoError::ConfigError("Input interface not found".to_string()))
        }
    }
    
    /// Connect an output interface
    pub async fn connect_output(&mut self, id: IoConnectionId, config: IoConfig) -> Result<(), IoError> {
        if let Some(interface) = self.output_interfaces.get_mut(&id) {
            interface.connect(config).await
        } else {
            Err(IoError::ConfigError("Output interface not found".to_string()))
        }
    }
    
    /// Start processing: read from inputs, process through neural network, send to outputs
    pub async fn start_processing(&mut self) -> Result<(), IoError> {
        let (input_tx, mut input_rx) = mpsc::channel::<IoData>(100);
        
        // Start input streaming
        for (_, interface) in self.input_interfaces.iter_mut() {
            interface.start_streaming(input_tx.clone()).await?;
        }
        
        // Create output channels for each interface
        let mut output_senders = Vec::new();
        for (_, interface) in self.output_interfaces.iter_mut() {
            let (tx, rx) = mpsc::channel::<IoData>(100);
            interface.start_streaming(rx).await?;
            output_senders.push(tx);
        }
        
        // Process neural network in the middle
        let nn = Arc::clone(&self.neural_network);
        tokio::spawn(async move {
            while let Some(input_data) = input_rx.recv().await {
                let output = {
                    let mut network = nn.lock().unwrap();
                    network.predict(&input_data.values)
                };
                
                let output_data = IoData {
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                    values: output,
                    metadata: input_data.metadata,
                };
                
                // Send to all output interfaces
                for sender in &output_senders {
                    let _ = sender.send(output_data.clone()).await;
                }
            }
        });
        
        Ok(())
    }
    
    /// Get status of all connections
    pub fn get_status(&self) -> HashMap<IoConnectionId, bool> {
        let mut status = HashMap::new();
        
        for (id, interface) in &self.input_interfaces {
            status.insert(*id, interface.is_connected());
        }
        
        for (id, interface) in &self.output_interfaces {
            status.insert(*id, interface.is_connected());
        }
        
        status
    }
}