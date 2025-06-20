pub mod neural_network;
pub mod network_composer;
pub mod distributed_network;

pub use neural_network::{NeuralNetwork, HebbianLearningMode};
pub use network_composer::{NetworkComposer, NetworkConnection};
pub use distributed_network::{DistributedNetwork, NetworkMessage, MessagePayload, MessageType, ProtocolError, capabilities};