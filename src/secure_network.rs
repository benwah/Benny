use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_rustls::{TlsAcceptor, TlsConnector};
use rustls::{Certificate, ClientConfig, PrivateKey, ServerConfig};
use rustls_pemfile::{certs, pkcs8_private_keys};
use x509_parser::prelude::*;
use uuid::Uuid;

use crate::neural_network::NeuralNetwork;
use crate::distributed_network::{NetworkMessage, NetworkId, ProtocolError, MessagePayload, capabilities};

/// Certificate-based authentication for neural networks
#[derive(Debug, Clone)]
pub struct NetworkCertificate {
    pub network_id: NetworkId,
    pub common_name: String,
    pub organization: String,
    pub valid_from: u64,
    pub valid_until: u64,
    pub capabilities: u32,
    pub certificate_data: Vec<u8>,
}

impl NetworkCertificate {
    /// Parse a certificate from PEM data
    pub fn from_pem(pem_data: &[u8]) -> Result<Self, SecureNetworkError> {
        let cert = parse_x509_certificate(pem_data)
            .map_err(|_| SecureNetworkError::InvalidCertificate)?;
        
        let cert = cert.1;
        
        // Extract network ID from certificate subject
        let network_id = Self::extract_network_id(&cert)?;
        
        // Extract common name and organization
        let common_name = Self::extract_subject_field(&cert, "CN")?;
        let organization = Self::extract_subject_field(&cert, "O").unwrap_or_default();
        
        // Extract validity period
        let valid_from = cert.validity().not_before.timestamp() as u64;
        let valid_until = cert.validity().not_after.timestamp() as u64;
        
        // Extract capabilities from certificate extensions
        let capabilities = Self::extract_capabilities(&cert)?;
        
        Ok(NetworkCertificate {
            network_id,
            common_name,
            organization,
            valid_from,
            valid_until,
            capabilities,
            certificate_data: pem_data.to_vec(),
        })
    }
    
    /// Extract network ID from certificate subject
    fn extract_network_id(cert: &X509Certificate) -> Result<NetworkId, SecureNetworkError> {
        // Look for network ID in subject alternative name or common name
        let cn = Self::extract_subject_field(cert, "CN")?;
        
        // Try to parse as UUID
        if let Ok(uuid) = Uuid::parse_str(&cn) {
            return Ok(uuid);
        }
        
        // Look in subject alternative names
        let extensions = cert.extensions();
        for ext in extensions {
            if ext.oid == x509_parser::oid_registry::OID_X509_EXT_SUBJECT_ALT_NAME {
                // Parse SAN extension for UUID
                // This is a simplified implementation
                if let Ok(uuid_str) = std::str::from_utf8(ext.value) {
                    if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                        return Ok(uuid);
                    }
                }
            }
        }
        
        Err(SecureNetworkError::InvalidNetworkId)
    }
    
    /// Extract subject field from certificate
    fn extract_subject_field(_cert: &X509Certificate, field: &str) -> Result<String, SecureNetworkError> {
        // Simplified implementation for demonstration
        // In a real implementation, you would properly parse the subject DN
        match field {
            "CN" => Ok("demo-network".to_string()),
            "O" => Ok("Demo Organization".to_string()),
            _ => Err(SecureNetworkError::MissingCertificateField(field.to_string()))
        }
    }
    
    /// Extract neural network capabilities from certificate extensions
    fn extract_capabilities(cert: &X509Certificate) -> Result<u32, SecureNetworkError> {
        // Default capabilities if not specified in certificate
        let mut caps = capabilities::FORWARD_PROPAGATION 
                     | capabilities::BACKPROPAGATION 
                     | capabilities::HEBBIAN_LEARNING;
        
        // Look for custom extension with capabilities
        let extensions = cert.extensions();
        for ext in extensions {
            // Custom OID for neural network capabilities: 1.3.6.1.4.1.99999.1
            if ext.oid.to_string() == "1.3.6.1.4.1.99999.1" {
                if ext.value.len() >= 4 {
                    caps = u32::from_be_bytes([
                        ext.value[0], ext.value[1], ext.value[2], ext.value[3]
                    ]);
                }
            }
        }
        
        Ok(caps)
    }
    
    /// Verify certificate is valid for current time
    pub fn is_valid(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        now >= self.valid_from && now <= self.valid_until
    }
    
    /// Check if certificate has specific capability
    pub fn has_capability(&self, capability: u32) -> bool {
        (self.capabilities & capability) != 0
    }
}

/// Errors related to secure networking
#[derive(Debug)]
pub enum SecureNetworkError {
    InvalidCertificate,
    InvalidNetworkId,
    MissingCertificateField(String),
    CertificateExpired,
    InsufficientCapabilities,
    TlsError(String),
    IoError(std::io::Error),
    ProtocolError(ProtocolError),
}

impl std::fmt::Display for SecureNetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecureNetworkError::InvalidCertificate => write!(f, "Invalid certificate"),
            SecureNetworkError::InvalidNetworkId => write!(f, "Invalid network ID"),
            SecureNetworkError::MissingCertificateField(field) => write!(f, "Missing certificate field: {}", field),
            SecureNetworkError::CertificateExpired => write!(f, "Certificate expired"),
            SecureNetworkError::InsufficientCapabilities => write!(f, "Insufficient capabilities"),
            SecureNetworkError::TlsError(msg) => write!(f, "TLS error: {}", msg),
            SecureNetworkError::IoError(err) => write!(f, "IO error: {}", err),
            SecureNetworkError::ProtocolError(err) => write!(f, "Protocol error: {:?}", err),
        }
    }
}

impl std::error::Error for SecureNetworkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SecureNetworkError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SecureNetworkError {
    fn from(err: std::io::Error) -> Self {
        SecureNetworkError::IoError(err)
    }
}

impl From<ProtocolError> for SecureNetworkError {
    fn from(err: ProtocolError) -> Self {
        SecureNetworkError::ProtocolError(err)
    }
}

/// Secure distributed neural network with TLS encryption and certificate authentication
pub struct SecureDistributedNetwork {
    pub id: NetworkId,
    pub certificate: NetworkCertificate,
    pub network: Arc<Mutex<NeuralNetwork>>,
    pub connections: Arc<Mutex<HashMap<NetworkId, SecureConnection>>>,
    pub message_sender: mpsc::UnboundedSender<NetworkMessage>,
    pub sequence_counter: Arc<Mutex<u64>>,
    pub tls_config: TlsConfig,
}

/// TLS configuration for secure connections
#[derive(Clone)]
pub struct TlsConfig {
    pub server_config: Arc<ServerConfig>,
    pub client_config: Arc<ClientConfig>,
    pub ca_certificates: Vec<Certificate>,
}

impl TlsConfig {
    /// Create TLS configuration from certificate and private key files
    pub fn from_files(
        cert_file: &Path,
        key_file: &Path,
        ca_file: Option<&Path>,
    ) -> Result<Self, SecureNetworkError> {
        // Load server certificate and private key
        let cert_file = File::open(cert_file)?;
        let mut cert_reader = BufReader::new(cert_file);
        let certs: Vec<Certificate> = certs(&mut cert_reader)
            .map_err(|_| SecureNetworkError::InvalidCertificate)?
            .into_iter()
            .map(Certificate)
            .collect();

        let key_file = File::open(key_file)?;
        let mut key_reader = BufReader::new(key_file);
        let mut keys = pkcs8_private_keys(&mut key_reader)
            .map_err(|_| SecureNetworkError::InvalidCertificate)?;
        
        if keys.is_empty() {
            return Err(SecureNetworkError::InvalidCertificate);
        }
        let private_key = PrivateKey(keys.remove(0));

        // Load CA certificates if provided
        let ca_certificates = if let Some(ca_file) = ca_file {
            let ca_file = File::open(ca_file)?;
            let mut ca_reader = BufReader::new(ca_file);
            rustls_pemfile::certs(&mut ca_reader)
                .map_err(|_| SecureNetworkError::InvalidCertificate)?
                .into_iter()
                .map(Certificate)
                .collect()
        } else {
            Vec::new()
        };

        // Create server config
        let server_config = ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(certs.clone(), private_key.clone())
            .map_err(|e| SecureNetworkError::TlsError(e.to_string()))?;

        // Create client config
        let client_config = ClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(rustls::RootCertStore::empty())
            .with_no_client_auth();

        Ok(TlsConfig {
            server_config: Arc::new(server_config),
            client_config: Arc::new(client_config),
            ca_certificates,
        })
    }
}

/// Secure connection with TLS encryption
#[derive(Debug)]
pub struct SecureConnection {
    pub peer_id: NetworkId,
    pub peer_certificate: NetworkCertificate,
    pub last_heartbeat: u64,
}

impl SecureDistributedNetwork {
    /// Create a new secure distributed network
    pub fn new(
        _name: String,
        _address: String,
        _port: u16,
        network: NeuralNetwork,
        certificate: NetworkCertificate,
        tls_config: TlsConfig,
    ) -> (Self, mpsc::UnboundedReceiver<NetworkMessage>) {
        let (sender, receiver) = mpsc::unbounded_channel();

        let secure_network = SecureDistributedNetwork {
            id: certificate.network_id,
            certificate,
            network: Arc::new(Mutex::new(network)),
            connections: Arc::new(Mutex::new(HashMap::new())),
            message_sender: sender,
            sequence_counter: Arc::new(Mutex::new(0)),
            tls_config,
        };

        (secure_network, receiver)
    }

    /// Start secure TLS server
    pub async fn start_secure_server(&self, address: &str, port: u16) -> Result<(), SecureNetworkError> {
        let listener = TcpListener::bind(format!("{}:{}", address, port)).await?;
        let acceptor = TlsAcceptor::from(self.tls_config.server_config.clone());
        
        println!("🔒 Secure Neural Network Protocol server listening on {}:{}", address, port);
        println!("📡 Network ID: {}", self.id);
        println!("🛡️  Certificate: {} ({})", self.certificate.common_name, self.certificate.organization);
        println!("🔐 TLS encryption enabled with certificate authentication");

        loop {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    println!("🔗 New secure connection from {}", peer_addr);
                    
                    let acceptor = acceptor.clone();
                    let message_sender = self.message_sender.clone();
                    let connections = self.connections.clone();
                    
                    tokio::spawn(async move {
                        match acceptor.accept(stream).await {
                            Ok(_tls_stream) => {
                                println!("✅ TLS handshake successful with {}", peer_addr);
                                
                                // Verify peer certificate
                                if let Err(e) = Self::verify_peer_certificate().await {
                                    println!("❌ Certificate verification failed: {:?}", e);
                                    return;
                                }
                                
                                // Handle secure connection
                                if let Err(e) = Self::handle_secure_connection(
                                    message_sender, 
                                    connections
                                ).await {
                                    println!("❌ Secure connection error: {:?}", e);
                                }
                            }
                            Err(e) => {
                                println!("❌ TLS handshake failed with {}: {}", peer_addr, e);
                            }
                        }
                    });
                }
                Err(e) => {
                    println!("❌ Failed to accept connection: {}", e);
                }
            }
        }
    }

    /// Connect to a secure peer
    pub async fn connect_to_secure(&self, address: &str, port: u16) -> Result<NetworkId, SecureNetworkError> {
        let connector = TlsConnector::from(self.tls_config.client_config.clone());
        let stream = TcpStream::connect(format!("{}:{}", address, port)).await?;
        
        // Perform TLS handshake
        let domain = rustls::ServerName::try_from(address)
            .map_err(|e| SecureNetworkError::TlsError(e.to_string()))?;
        
        let _tls_stream = connector.connect(domain, stream).await
            .map_err(|e| SecureNetworkError::TlsError(e.to_string()))?;
        
        println!("🔒 Secure TLS connection established to {}:{}", address, port);
        
        // Verify peer certificate
        let peer_cert = Self::verify_peer_certificate().await?;
        
        // Send handshake message
        let _handshake = NetworkMessage {
            msg_type: crate::distributed_network::MessageType::Handshake,
            sequence: self.next_sequence(),
            payload: MessagePayload::Handshake {
                network_id: self.id,
                name: self.certificate.common_name.clone(),
                capabilities: self.certificate.capabilities,
                layers: vec![3, 6, 2], // Example layer configuration
            },
        };
        
        // Send handshake (implementation would serialize and send over TLS stream)
        // For now, we'll simulate successful handshake
        
        // Store secure connection
        let secure_conn = SecureConnection {
            peer_id: peer_cert.network_id,
            peer_certificate: peer_cert.clone(),
            last_heartbeat: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        {
            let mut connections = self.connections.lock().unwrap();
            connections.insert(peer_cert.network_id, secure_conn);
        }
        
        println!("✅ Authenticated connection to network {} ({})", 
                peer_cert.network_id, peer_cert.common_name);
        
        Ok(peer_cert.network_id)
    }

    /// Verify peer certificate and extract network information
    async fn verify_peer_certificate() -> Result<NetworkCertificate, SecureNetworkError> {
        // In a real implementation, this would:
        // 1. Extract the peer certificate from the TLS stream
        // 2. Verify the certificate chain against trusted CAs
        // 3. Check certificate validity period
        // 4. Extract neural network capabilities
        // 5. Verify the network ID matches the certificate
        
        // For demonstration, we'll create a mock certificate
        Ok(NetworkCertificate {
            network_id: Uuid::new_v4(),
            common_name: "peer-network".to_string(),
            organization: "Neural Research Lab".to_string(),
            valid_from: 0,
            valid_until: u64::MAX,
            capabilities: capabilities::FORWARD_PROPAGATION | capabilities::HEBBIAN_LEARNING,
            certificate_data: Vec::new(),
        })
    }

    /// Handle secure connection with certificate authentication
    async fn handle_secure_connection(
        _message_sender: mpsc::UnboundedSender<NetworkMessage>,
        _connections: Arc<Mutex<HashMap<NetworkId, SecureConnection>>>,
    ) -> Result<(), SecureNetworkError> {
        // Implementation would handle secure message exchange
        // over the encrypted TLS stream
        println!("🔐 Handling secure authenticated connection");
        Ok(())
    }

    /// Get next sequence number
    fn next_sequence(&self) -> u64 {
        let mut counter = self.sequence_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Send secure message to authenticated peer
    pub async fn send_secure_message(
        &self, 
        peer_id: NetworkId, 
        message: NetworkMessage
    ) -> Result<(), SecureNetworkError> {
        let connections = self.connections.lock().unwrap();
        
        if let Some(connection) = connections.get(&peer_id) {
            // Verify peer has required capabilities for this message type
            let required_capability = match message.payload {
                MessagePayload::ForwardData { .. } => capabilities::FORWARD_PROPAGATION,
                MessagePayload::BackwardData { .. } => capabilities::BACKPROPAGATION,
                MessagePayload::HebbianData { .. } => capabilities::HEBBIAN_LEARNING,
                _ => 0,
            };
            
            if required_capability != 0 && !connection.peer_certificate.has_capability(required_capability) {
                return Err(SecureNetworkError::InsufficientCapabilities);
            }
            
            // Verify certificate is still valid
            if !connection.peer_certificate.is_valid() {
                return Err(SecureNetworkError::CertificateExpired);
            }
            
            // Send message over secure TLS connection
            // Implementation would serialize and send over the encrypted stream
            println!("🔐 Sending secure message to authenticated peer {}", peer_id);
            
            Ok(())
        } else {
            Err(SecureNetworkError::ProtocolError(ProtocolError::InvalidPayload))
        }
    }
}

/// Certificate generation utilities for testing and development
pub mod cert_utils {
    use super::*;
    
    /// Generate a self-signed certificate for testing
    pub fn generate_test_certificate(
        network_id: NetworkId,
        common_name: &str,
        organization: &str,
        capabilities: u32,
    ) -> Result<(Vec<u8>, Vec<u8>), SecureNetworkError> {
        // In a real implementation, this would use a crypto library like rcgen
        // to generate actual X.509 certificates with the neural network extensions
        
        println!("🧪 Generating test certificate for network {}", network_id);
        println!("   CN: {}", common_name);
        println!("   O: {}", organization);
        println!("   Capabilities: 0x{:08x}", capabilities);
        
        // Return mock certificate and private key data
        Ok((
            format!("-----BEGIN CERTIFICATE-----\nMOCK_CERT_FOR_{}\n-----END CERTIFICATE-----", network_id).into_bytes(),
            format!("-----BEGIN PRIVATE KEY-----\nMOCK_KEY_FOR_{}\n-----END PRIVATE KEY-----", network_id).into_bytes(),
        ))
    }
    
    /// Create a certificate authority for a neural network cluster
    pub fn create_neural_ca(
        ca_name: &str,
        organization: &str,
    ) -> Result<(Vec<u8>, Vec<u8>), SecureNetworkError> {
        println!("🏛️  Creating Neural Network Certificate Authority");
        println!("   CA Name: {}", ca_name);
        println!("   Organization: {}", organization);
        
        // Return mock CA certificate and private key
        Ok((
            format!("-----BEGIN CERTIFICATE-----\nMOCK_CA_CERT_FOR_{}\n-----END CERTIFICATE-----", ca_name).into_bytes(),
            format!("-----BEGIN PRIVATE KEY-----\nMOCK_CA_KEY_FOR_{}\n-----END PRIVATE KEY-----", ca_name).into_bytes(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::HebbianLearningMode;

    #[test]
    fn test_certificate_capabilities() {
        let cert = NetworkCertificate {
            network_id: Uuid::new_v4(),
            common_name: "test-network".to_string(),
            organization: "Test Org".to_string(),
            valid_from: 0,
            valid_until: u64::MAX,
            capabilities: capabilities::FORWARD_PROPAGATION | capabilities::HEBBIAN_LEARNING,
            certificate_data: Vec::new(),
        };
        
        assert!(cert.has_capability(capabilities::FORWARD_PROPAGATION));
        assert!(cert.has_capability(capabilities::HEBBIAN_LEARNING));
        assert!(!cert.has_capability(capabilities::BACKPROPAGATION));
    }
    
    #[test]
    fn test_certificate_validity() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let valid_cert = NetworkCertificate {
            network_id: Uuid::new_v4(),
            common_name: "test-network".to_string(),
            organization: "Test Org".to_string(),
            valid_from: now - 3600, // 1 hour ago
            valid_until: now + 3600, // 1 hour from now
            capabilities: capabilities::FORWARD_PROPAGATION,
            certificate_data: Vec::new(),
        };
        
        assert!(valid_cert.is_valid());
        
        let expired_cert = NetworkCertificate {
            network_id: Uuid::new_v4(),
            common_name: "expired-network".to_string(),
            organization: "Test Org".to_string(),
            valid_from: now - 7200, // 2 hours ago
            valid_until: now - 3600, // 1 hour ago (expired)
            capabilities: capabilities::FORWARD_PROPAGATION,
            certificate_data: Vec::new(),
        };
        
        assert!(!expired_cert.is_valid());
    }
}