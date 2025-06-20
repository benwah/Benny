use neural_network::{DistributedNetwork, NeuralNetwork};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("🧠 Starting Neural Network Server");

    // Create a simple neural network (4 inputs, 2 hidden, 1 output)
    let network = NeuralNetwork::with_layers(&[4, 2, 1], 0.01);

    // Create distributed network wrapper
    let (distributed_network, mut message_receiver) = DistributedNetwork::new(
        "NeuralNetworkServer".to_string(),
        "127.0.0.1".to_string(),
        8001,
        network,
    );

    // Start the server
    if let Err(e) = distributed_network.start_server().await {
        eprintln!("❌ Failed to start server: {:?}", e);
        return Ok(());
    }
    println!("✅ Neural Network server started on 127.0.0.1:8001");
    println!("📡 Network ID: {}", distributed_network.id);
    println!(
        "🧠 Capabilities: 0x{:08x}",
        distributed_network.info.capabilities
    );

    // Process incoming messages
    tokio::spawn(async move {
        while let Some(message) = message_receiver.recv().await {
            println!("📥 Received message: {:?}", message.msg_type);

            match message.payload {
                neural_network::MessagePayload::ForwardData { layer_id, data } => {
                    println!(
                        "📊 Processing input data for layer {}: {:?}",
                        layer_id, data
                    );
                    // In a real implementation, you would process this data through the neural network
                    // For now, we'll just acknowledge receipt
                }
                _ => {
                    println!("📋 Received other message type: {:?}", message.payload);
                }
            }
        }
    });

    println!("🔄 Server running... Press Ctrl+C to stop");

    // Keep the server running
    loop {
        sleep(Duration::from_secs(1)).await;
    }
}
