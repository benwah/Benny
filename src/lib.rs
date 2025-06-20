pub mod neural_network;
pub mod network_composer;
pub mod distributed_network;
pub mod secure_network;
pub mod cli;
pub mod runner;

pub use neural_network::{NeuralNetwork, HebbianLearningMode};
pub use network_composer::{NetworkComposer, NetworkConnection};
pub use distributed_network::{DistributedNetwork, NetworkMessage, MessagePayload, MessageType, ProtocolError, capabilities};
pub use secure_network::{SecureDistributedNetwork, NetworkCertificate, TlsConfig, SecureNetworkError};