use crate::neural_network::NeuralNetwork;
use byteorder::{BigEndian, ByteOrder};
use crc32fast::Hasher;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Unique identifier for a neural network node
pub type NetworkId = Uuid;

/// Neural Network Protocol (NNP) - Optimized Binary Protocol
///
/// Protocol Structure:
/// [MAGIC][VERSION][MSG_TYPE][LENGTH][SEQUENCE][CHECKSUM][PAYLOAD]
///
/// MAGIC: 4 bytes - "NNP\0" (0x4E4E5000)
/// VERSION: 1 byte - Protocol version (currently 1)
/// MSG_TYPE: 1 byte - Message type identifier
/// LENGTH: 4 bytes - Payload length (big-endian)
/// SEQUENCE: 8 bytes - Message sequence number (big-endian)
/// CHECKSUM: 4 bytes - CRC32 of payload (big-endian)
/// PAYLOAD: Variable length - Message data
const PROTOCOL_MAGIC: [u8; 4] = [0x4E, 0x4E, 0x50, 0x00]; // "NNP\0"
const PROTOCOL_VERSION: u8 = 1;
const HEADER_SIZE: usize = 22; // 4 + 1 + 1 + 4 + 8 + 4

/// Message types for the neural network protocol
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum MessageType {
    Handshake = 0x01,
    HandshakeAck = 0x02,
    ForwardData = 0x10,
    BackwardData = 0x11,
    HebbianData = 0x12,
    WeightSync = 0x13,
    Heartbeat = 0x20,
    Disconnect = 0x21,
    Error = 0xFF,
}

impl From<u8> for MessageType {
    fn from(value: u8) -> Self {
        match value {
            0x01 => MessageType::Handshake,
            0x02 => MessageType::HandshakeAck,
            0x10 => MessageType::ForwardData,
            0x11 => MessageType::BackwardData,
            0x12 => MessageType::HebbianData,
            0x13 => MessageType::WeightSync,
            0x20 => MessageType::Heartbeat,
            0x21 => MessageType::Disconnect,
            _ => MessageType::Error,
        }
    }
}

/// Optimized message structure for neural network communication
#[derive(Debug, Clone)]
pub struct NetworkMessage {
    pub msg_type: MessageType,
    pub sequence: u64,
    pub payload: MessagePayload,
}

#[derive(Debug, Clone)]
pub enum MessagePayload {
    /// Handshake with network information
    Handshake {
        network_id: NetworkId,
        name: String,
        layers: Vec<u16>,
        capabilities: u32, // Bitfield for capabilities
    },
    /// Acknowledgment of handshake
    HandshakeAck {
        network_id: NetworkId,
        accepted: bool,
    },
    /// Forward propagation data (highly optimized)
    ForwardData {
        layer_id: u8,
        data: Vec<f32>, // Using f32 for better network performance
    },
    /// Backpropagation gradients
    BackwardData { layer_id: u8, gradients: Vec<f32> },
    /// Hebbian correlation data
    HebbianData {
        layer_id: u8,
        correlations: Vec<f32>,
        learning_rate: f32,
    },
    /// Weight synchronization
    WeightSync {
        layer_id: u8,
        weights: Vec<f32>,
        biases: Vec<f32>,
    },
    /// Heartbeat
    Heartbeat { timestamp: u64 },
    /// Disconnect notification
    Disconnect { reason: String },
    /// Error message
    Error { code: u16, message: String },
}

/// Capability flags for neural networks
pub mod capabilities {
    pub const FORWARD_PROPAGATION: u32 = 1 << 0;
    pub const BACKPROPAGATION: u32 = 1 << 1;
    pub const HEBBIAN_LEARNING: u32 = 1 << 2;
    pub const WEIGHT_SYNC: u32 = 1 << 3;
    pub const CORRELATION_ANALYSIS: u32 = 1 << 4;
    pub const MULTI_LAYER: u32 = 1 << 5;
    pub const REAL_TIME: u32 = 1 << 6;
    pub const COMPRESSION: u32 = 1 << 7;
}

/// Information about a neural network node
#[derive(Debug, Clone)]
pub struct NetworkInfo {
    pub id: NetworkId,
    pub name: String,
    pub address: String,
    pub port: u16,
    pub layers: Vec<u16>,
    pub capabilities: u32,
    pub status: NetworkStatus,
}

/// Connection between two neural networks
#[derive(Debug)]
pub struct NetworkConnection {
    pub peer_id: NetworkId,
    pub stream: Option<TcpStream>,
    pub capabilities: u32,
    pub last_heartbeat: u64,
    pub sequence_counter: u64,
}

/// Status of a network node
#[derive(Debug, Clone, Copy)]
pub enum NetworkStatus {
    Online,
    Training,
    Idle,
    Offline,
}

/// Binary protocol implementation for neural network messages
impl NetworkMessage {
    /// Serialize message to binary format
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload_bytes = self.payload.to_bytes();
        let payload_len = payload_bytes.len() as u32;

        // Calculate CRC32 checksum
        let mut hasher = Hasher::new();
        hasher.update(&payload_bytes);
        let checksum = hasher.finalize();

        let mut buffer = Vec::with_capacity(HEADER_SIZE + payload_bytes.len());

        // Header: MAGIC + VERSION + MSG_TYPE + LENGTH + SEQUENCE + CHECKSUM
        buffer.extend_from_slice(&PROTOCOL_MAGIC);
        buffer.push(PROTOCOL_VERSION);
        buffer.push(self.msg_type as u8);

        let mut len_bytes = [0u8; 4];
        BigEndian::write_u32(&mut len_bytes, payload_len);
        buffer.extend_from_slice(&len_bytes);

        let mut seq_bytes = [0u8; 8];
        BigEndian::write_u64(&mut seq_bytes, self.sequence);
        buffer.extend_from_slice(&seq_bytes);

        let mut checksum_bytes = [0u8; 4];
        BigEndian::write_u32(&mut checksum_bytes, checksum);
        buffer.extend_from_slice(&checksum_bytes);

        // Payload
        buffer.extend_from_slice(&payload_bytes);

        buffer
    }

    /// Deserialize message from binary format
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProtocolError> {
        if bytes.len() < HEADER_SIZE {
            return Err(ProtocolError::InvalidLength);
        }

        // Verify magic number
        if bytes[0..4] != PROTOCOL_MAGIC {
            return Err(ProtocolError::InvalidMagic);
        }

        // Check version
        if bytes[4] != PROTOCOL_VERSION {
            return Err(ProtocolError::UnsupportedVersion);
        }

        let msg_type = MessageType::from(bytes[5]);
        let payload_len = BigEndian::read_u32(&bytes[6..10]) as usize;
        let sequence = BigEndian::read_u64(&bytes[10..18]);
        let expected_checksum = BigEndian::read_u32(&bytes[18..22]);

        if bytes.len() != HEADER_SIZE + payload_len {
            return Err(ProtocolError::InvalidLength);
        }

        let payload_bytes = &bytes[HEADER_SIZE..];

        // Verify checksum
        let mut hasher = Hasher::new();
        hasher.update(payload_bytes);
        let actual_checksum = hasher.finalize();

        if actual_checksum != expected_checksum {
            return Err(ProtocolError::ChecksumMismatch);
        }

        let payload = MessagePayload::from_bytes(msg_type, payload_bytes)?;

        Ok(NetworkMessage {
            msg_type,
            sequence,
            payload,
        })
    }
}

impl MessagePayload {
    /// Serialize payload to bytes
    fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();

        match self {
            MessagePayload::Handshake {
                network_id,
                name,
                layers,
                capabilities,
            } => {
                buffer.extend_from_slice(network_id.as_bytes());

                let name_bytes = name.as_bytes();
                buffer.push(name_bytes.len() as u8);
                buffer.extend_from_slice(name_bytes);

                buffer.push(layers.len() as u8);
                for &layer_size in layers {
                    let mut layer_bytes = [0u8; 2];
                    BigEndian::write_u16(&mut layer_bytes, layer_size);
                    buffer.extend_from_slice(&layer_bytes);
                }

                let mut cap_bytes = [0u8; 4];
                BigEndian::write_u32(&mut cap_bytes, *capabilities);
                buffer.extend_from_slice(&cap_bytes);
            }

            MessagePayload::HandshakeAck {
                network_id,
                accepted,
            } => {
                buffer.extend_from_slice(network_id.as_bytes());
                buffer.push(if *accepted { 1 } else { 0 });
            }

            MessagePayload::ForwardData { layer_id, data } => {
                buffer.push(*layer_id);

                let mut len_bytes = [0u8; 4];
                BigEndian::write_u32(&mut len_bytes, data.len() as u32);
                buffer.extend_from_slice(&len_bytes);

                for &value in data {
                    let mut value_bytes = [0u8; 4];
                    BigEndian::write_u32(&mut value_bytes, value.to_bits());
                    buffer.extend_from_slice(&value_bytes);
                }
            }

            MessagePayload::BackwardData {
                layer_id,
                gradients,
            } => {
                buffer.push(*layer_id);

                let mut len_bytes = [0u8; 4];
                BigEndian::write_u32(&mut len_bytes, gradients.len() as u32);
                buffer.extend_from_slice(&len_bytes);

                for &gradient in gradients {
                    let mut grad_bytes = [0u8; 4];
                    BigEndian::write_u32(&mut grad_bytes, gradient.to_bits());
                    buffer.extend_from_slice(&grad_bytes);
                }
            }

            MessagePayload::HebbianData {
                layer_id,
                correlations,
                learning_rate,
            } => {
                buffer.push(*layer_id);

                let mut rate_bytes = [0u8; 4];
                BigEndian::write_u32(&mut rate_bytes, learning_rate.to_bits());
                buffer.extend_from_slice(&rate_bytes);

                let mut len_bytes = [0u8; 4];
                BigEndian::write_u32(&mut len_bytes, correlations.len() as u32);
                buffer.extend_from_slice(&len_bytes);

                for &correlation in correlations {
                    let mut corr_bytes = [0u8; 4];
                    BigEndian::write_u32(&mut corr_bytes, correlation.to_bits());
                    buffer.extend_from_slice(&corr_bytes);
                }
            }

            MessagePayload::WeightSync {
                layer_id,
                weights,
                biases,
            } => {
                buffer.push(*layer_id);

                let mut weights_len_bytes = [0u8; 4];
                BigEndian::write_u32(&mut weights_len_bytes, weights.len() as u32);
                buffer.extend_from_slice(&weights_len_bytes);

                for &weight in weights {
                    let mut weight_bytes = [0u8; 4];
                    BigEndian::write_u32(&mut weight_bytes, weight.to_bits());
                    buffer.extend_from_slice(&weight_bytes);
                }

                let mut biases_len_bytes = [0u8; 4];
                BigEndian::write_u32(&mut biases_len_bytes, biases.len() as u32);
                buffer.extend_from_slice(&biases_len_bytes);

                for &bias in biases {
                    let mut bias_bytes = [0u8; 4];
                    BigEndian::write_u32(&mut bias_bytes, bias.to_bits());
                    buffer.extend_from_slice(&bias_bytes);
                }
            }

            MessagePayload::Heartbeat { timestamp } => {
                let mut time_bytes = [0u8; 8];
                BigEndian::write_u64(&mut time_bytes, *timestamp);
                buffer.extend_from_slice(&time_bytes);
            }

            MessagePayload::Disconnect { reason } => {
                let reason_bytes = reason.as_bytes();
                buffer.push(reason_bytes.len() as u8);
                buffer.extend_from_slice(reason_bytes);
            }

            MessagePayload::Error { code, message } => {
                let mut code_bytes = [0u8; 2];
                BigEndian::write_u16(&mut code_bytes, *code);
                buffer.extend_from_slice(&code_bytes);

                let message_bytes = message.as_bytes();
                buffer.push(message_bytes.len() as u8);
                buffer.extend_from_slice(message_bytes);
            }
        }

        buffer
    }

    /// Deserialize payload from bytes
    fn from_bytes(msg_type: MessageType, bytes: &[u8]) -> Result<Self, ProtocolError> {
        let mut offset = 0;

        match msg_type {
            MessageType::Handshake => {
                if bytes.len() < 16 {
                    return Err(ProtocolError::InvalidPayload);
                }

                let network_id =
                    Uuid::from_bytes_le(bytes[offset..offset + 16].try_into().unwrap());
                offset += 16;

                let name_len = bytes[offset] as usize;
                offset += 1;

                if bytes.len() < offset + name_len {
                    return Err(ProtocolError::InvalidPayload);
                }
                let name = String::from_utf8_lossy(&bytes[offset..offset + name_len]).to_string();
                offset += name_len;

                let layers_count = bytes[offset] as usize;
                offset += 1;

                let mut layers = Vec::with_capacity(layers_count);
                for _ in 0..layers_count {
                    if bytes.len() < offset + 2 {
                        return Err(ProtocolError::InvalidPayload);
                    }
                    let layer_size = BigEndian::read_u16(&bytes[offset..offset + 2]);
                    layers.push(layer_size);
                    offset += 2;
                }

                if bytes.len() < offset + 4 {
                    return Err(ProtocolError::InvalidPayload);
                }
                let capabilities = BigEndian::read_u32(&bytes[offset..offset + 4]);

                Ok(MessagePayload::Handshake {
                    network_id,
                    name,
                    layers,
                    capabilities,
                })
            }

            MessageType::ForwardData => {
                if bytes.len() < 5 {
                    return Err(ProtocolError::InvalidPayload);
                }

                let layer_id = bytes[0];
                let data_len = BigEndian::read_u32(&bytes[1..5]) as usize;

                if bytes.len() != 5 + data_len * 4 {
                    return Err(ProtocolError::InvalidPayload);
                }

                let mut data = Vec::with_capacity(data_len);
                for i in 0..data_len {
                    let start = 5 + i * 4;
                    let bits = BigEndian::read_u32(&bytes[start..start + 4]);
                    data.push(f32::from_bits(bits));
                }

                Ok(MessagePayload::ForwardData { layer_id, data })
            }

            MessageType::HandshakeAck => {
                if bytes.len() != 17 {
                    return Err(ProtocolError::InvalidPayload);
                }
                let network_id = Uuid::from_bytes_le(bytes[0..16].try_into().unwrap());
                let accepted = bytes[16] != 0;
                Ok(MessagePayload::HandshakeAck {
                    network_id,
                    accepted,
                })
            }

            MessageType::Heartbeat => {
                if bytes.len() != 8 {
                    return Err(ProtocolError::InvalidPayload);
                }
                let timestamp = BigEndian::read_u64(&bytes[0..8]);
                Ok(MessagePayload::Heartbeat { timestamp })
            }

            MessageType::BackwardData => {
                if bytes.len() < 5 {
                    return Err(ProtocolError::InvalidPayload);
                }

                let layer_id = bytes[0];
                let data_len = BigEndian::read_u32(&bytes[1..5]) as usize;

                if bytes.len() != 5 + data_len * 4 {
                    return Err(ProtocolError::InvalidPayload);
                }

                let mut gradients = Vec::with_capacity(data_len);
                for i in 0..data_len {
                    let start = 5 + i * 4;
                    let bits = BigEndian::read_u32(&bytes[start..start + 4]);
                    gradients.push(f32::from_bits(bits));
                }

                Ok(MessagePayload::BackwardData {
                    layer_id,
                    gradients,
                })
            }

            MessageType::HebbianData => {
                if bytes.len() < 9 {
                    return Err(ProtocolError::InvalidPayload);
                }

                let layer_id = bytes[0];
                let learning_rate = f32::from_bits(BigEndian::read_u32(&bytes[1..5]));
                let data_len = BigEndian::read_u32(&bytes[5..9]) as usize;

                if bytes.len() != 9 + data_len * 4 {
                    return Err(ProtocolError::InvalidPayload);
                }

                let mut correlations = Vec::with_capacity(data_len);
                for i in 0..data_len {
                    let start = 9 + i * 4;
                    let bits = BigEndian::read_u32(&bytes[start..start + 4]);
                    correlations.push(f32::from_bits(bits));
                }

                Ok(MessagePayload::HebbianData {
                    layer_id,
                    correlations,
                    learning_rate,
                })
            }

            // Add other message type deserializations...
            _ => Err(ProtocolError::UnsupportedMessageType),
        }
    }
}

/// Protocol errors
#[derive(Debug)]
pub enum ProtocolError {
    InvalidMagic,
    UnsupportedVersion,
    InvalidLength,
    ChecksumMismatch,
    InvalidPayload,
    UnsupportedMessageType,
    IoError(std::io::Error),
}

impl From<std::io::Error> for ProtocolError {
    fn from(error: std::io::Error) -> Self {
        ProtocolError::IoError(error)
    }
}

/// Distributed Neural Network Node with optimized TCP protocol
#[derive(Clone)]
pub struct DistributedNetwork {
    pub id: NetworkId,
    pub info: NetworkInfo,
    pub network: Arc<Mutex<NeuralNetwork>>,
    pub connections: Arc<Mutex<HashMap<NetworkId, NetworkConnection>>>,
    pub message_sender: mpsc::UnboundedSender<NetworkMessage>,
    pub sequence_counter: Arc<Mutex<u64>>,
}

impl DistributedNetwork {
    /// Create a new distributed neural network node
    pub fn new(
        name: String,
        address: String,
        port: u16,
        network: NeuralNetwork,
    ) -> (Self, mpsc::UnboundedReceiver<NetworkMessage>) {
        let id = Uuid::new_v4();
        let layers: Vec<u16> = network.get_layers().iter().map(|&x| x as u16).collect();

        let capabilities = capabilities::FORWARD_PROPAGATION
            | capabilities::BACKPROPAGATION
            | capabilities::HEBBIAN_LEARNING
            | capabilities::CORRELATION_ANALYSIS
            | capabilities::MULTI_LAYER
            | capabilities::REAL_TIME;

        let info = NetworkInfo {
            id,
            name,
            address,
            port,
            layers,
            capabilities,
            status: NetworkStatus::Online,
        };

        let (sender, receiver) = mpsc::unbounded_channel();

        let distributed_network = DistributedNetwork {
            id,
            info,
            network: Arc::new(Mutex::new(network)),
            connections: Arc::new(Mutex::new(HashMap::new())),
            message_sender: sender,
            sequence_counter: Arc::new(Mutex::new(0)),
        };

        (distributed_network, receiver)
    }

    /// Get next sequence number for message ordering
    fn next_sequence(&self) -> u64 {
        let mut counter = self.sequence_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Start the TCP server for incoming connections
    pub async fn start_server(&self) -> Result<(), ProtocolError> {
        let addr = format!("{}:{}", self.info.address, self.info.port);
        let listener = TcpListener::bind(&addr).await?;

        println!("🚀 Neural Network Protocol server listening on {}", addr);
        println!("📡 Network ID: {}", self.id);
        println!("🧠 Capabilities: 0x{:08X}", self.info.capabilities);

        let connections = self.connections.clone();
        let message_sender = self.message_sender.clone();
        let network_id = self.id;

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        println!("🔗 New connection from {}", peer_addr);

                        let connections_clone = connections.clone();
                        let sender_clone = message_sender.clone();

                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_connection(
                                stream,
                                connections_clone,
                                sender_clone,
                                network_id,
                            )
                            .await
                            {
                                println!("❌ Connection error: {:?}", e);
                            }
                        });
                    }
                    Err(e) => {
                        println!("❌ Failed to accept connection: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Handle incoming TCP connection
    async fn handle_connection(
        mut stream: TcpStream,
        connections: Arc<Mutex<HashMap<NetworkId, NetworkConnection>>>,
        message_sender: mpsc::UnboundedSender<NetworkMessage>,
        our_network_id: NetworkId,
    ) -> Result<(), ProtocolError> {
        let mut buffer = vec![0u8; 8192]; // 8KB buffer for incoming messages

        loop {
            // Read message header first
            let mut header_buf = [0u8; HEADER_SIZE];
            match stream.read_exact(&mut header_buf).await {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    println!("🔌 Connection closed by peer");
                    break;
                }
                Err(e) => return Err(ProtocolError::IoError(e)),
            }

            // Parse header to get payload length
            if header_buf[0..4] != PROTOCOL_MAGIC {
                println!("❌ Invalid magic number received");
                continue;
            }

            let payload_len = BigEndian::read_u32(&header_buf[6..10]) as usize;

            // Read the complete message
            let total_len = HEADER_SIZE + payload_len;
            if buffer.len() < total_len {
                buffer.resize(total_len, 0);
            }

            buffer[..HEADER_SIZE].copy_from_slice(&header_buf);
            stream
                .read_exact(&mut buffer[HEADER_SIZE..total_len])
                .await?;

            // Parse the complete message
            match NetworkMessage::from_bytes(&buffer[..total_len]) {
                Ok(message) => {
                    println!("📥 Received message: {:?}", message.msg_type);

                    // Handle handshake messages specially
                    if let MessagePayload::Handshake { network_id, .. } = &message.payload {
                        // Store connection info
                        let connection = NetworkConnection {
                            peer_id: *network_id,
                            stream: None, // We'll store the stream separately for outgoing messages
                            capabilities: 0, // Will be updated from handshake
                            last_heartbeat: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                            sequence_counter: 0,
                        };

                        {
                            let mut conns = connections.lock().unwrap();
                            conns.insert(*network_id, connection);
                        }

                        // Send handshake acknowledgment
                        let ack_message = NetworkMessage {
                            msg_type: MessageType::HandshakeAck,
                            sequence: 1,
                            payload: MessagePayload::HandshakeAck {
                                network_id: our_network_id,
                                accepted: true,
                            },
                        };

                        let ack_bytes = ack_message.to_bytes();
                        stream.write_all(&ack_bytes).await?;
                        println!("📤 Sent handshake acknowledgment");
                    }

                    // Forward message to main handler
                    if message_sender.send(message).is_err() {
                        println!("❌ Failed to forward message to handler");
                        break;
                    }
                }
                Err(e) => {
                    println!("❌ Failed to parse message: {:?}", e);
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Connect to a remote neural network
    pub async fn connect_to(&self, address: &str, port: u16) -> Result<NetworkId, ProtocolError> {
        let addr = format!("{}:{}", address, port);
        println!("🔗 Connecting to neural network at {}", addr);

        let mut stream = TcpStream::connect(&addr).await?;

        // Send handshake
        let handshake = NetworkMessage {
            msg_type: MessageType::Handshake,
            sequence: self.next_sequence(),
            payload: MessagePayload::Handshake {
                network_id: self.id,
                name: self.info.name.clone(),
                layers: self.info.layers.clone(),
                capabilities: self.info.capabilities,
            },
        };

        let handshake_bytes = handshake.to_bytes();
        stream.write_all(&handshake_bytes).await?;

        // Wait for handshake acknowledgment
        let mut header_buf = [0u8; HEADER_SIZE];
        stream.read_exact(&mut header_buf).await?;

        let payload_len = BigEndian::read_u32(&header_buf[6..10]) as usize;
        let mut full_message = vec![0u8; HEADER_SIZE + payload_len];
        full_message[..HEADER_SIZE].copy_from_slice(&header_buf);
        stream.read_exact(&mut full_message[HEADER_SIZE..]).await?;

        match NetworkMessage::from_bytes(&full_message) {
            Ok(ack_message) => {
                if let MessagePayload::HandshakeAck {
                    network_id,
                    accepted,
                } = ack_message.payload
                {
                    if accepted {
                        println!("✅ Connected to network {}", network_id);

                        // Store connection
                        let connection = NetworkConnection {
                            peer_id: network_id,
                            stream: Some(stream),
                            capabilities: 0, // Will be updated
                            last_heartbeat: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                            sequence_counter: 0,
                        };

                        {
                            let mut connections = self.connections.lock().unwrap();
                            connections.insert(network_id, connection);
                        }

                        return Ok(network_id);
                    } else {
                        return Err(ProtocolError::InvalidPayload);
                    }
                }
            }
            Err(e) => return Err(e),
        }

        Err(ProtocolError::InvalidPayload)
    }

    /// Send forward propagation data to a connected network
    pub async fn send_forward_data(
        &self,
        peer_id: NetworkId,
        layer_id: u8,
        data: Vec<f64>,
    ) -> Result<(), ProtocolError> {
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

        let message = NetworkMessage {
            msg_type: MessageType::ForwardData,
            sequence: self.next_sequence(),
            payload: MessagePayload::ForwardData {
                layer_id,
                data: data_f32,
            },
        };

        self.send_message_to_peer(peer_id, message).await
    }

    /// Send Hebbian correlation data to a connected network
    pub async fn send_hebbian_data(
        &self,
        peer_id: NetworkId,
        layer_id: u8,
        correlations: Vec<f64>,
        learning_rate: f64,
    ) -> Result<(), ProtocolError> {
        let correlations_f32: Vec<f32> = correlations.iter().map(|&x| x as f32).collect();

        let message = NetworkMessage {
            msg_type: MessageType::HebbianData,
            sequence: self.next_sequence(),
            payload: MessagePayload::HebbianData {
                layer_id,
                correlations: correlations_f32,
                learning_rate: learning_rate as f32,
            },
        };

        self.send_message_to_peer(peer_id, message).await
    }

    /// Send a message to a specific peer
    async fn send_message_to_peer(
        &self,
        peer_id: NetworkId,
        message: NetworkMessage,
    ) -> Result<(), ProtocolError> {
        let _message_bytes = message.to_bytes();

        // For now, we'll need to establish a new connection for each message
        // In a production system, you'd maintain persistent connections
        let connections = self.connections.lock().unwrap();
        if let Some(_connection) = connections.get(&peer_id) {
            // TODO: Use persistent connection
            println!(
                "📤 Would send {:?} message to {}",
                message.msg_type, peer_id
            );
        }

        Ok(())
    }

    /// Process incoming network message
    pub async fn handle_message(&self, message: NetworkMessage) -> Result<(), ProtocolError> {
        match message.payload {
            MessagePayload::ForwardData { layer_id, data } => {
                println!(
                    "📥 Received forward data for layer {}: {} values",
                    layer_id,
                    data.len()
                );

                // Convert f32 back to f64 for processing
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();

                // Process through our network
                let mut network = self.network.lock().unwrap();
                let (_, output) = network.forward(&data_f64);
                drop(network);

                println!("🧠 Processed data, output: {:?}", output);
            }

            MessagePayload::HebbianData {
                layer_id,
                correlations,
                learning_rate,
            } => {
                println!(
                    "🧬 Received Hebbian data for layer {}: {} correlations (rate: {})",
                    layer_id,
                    correlations.len(),
                    learning_rate
                );

                // Could integrate this correlation data into our own learning
            }

            MessagePayload::Heartbeat { timestamp } => {
                println!("💓 Heartbeat received: {}", timestamp);
            }

            _ => {
                println!("📨 Received message: {:?}", message.msg_type);
            }
        }

        Ok(())
    }
}
