use neural_network::{DistributedNetwork, NeuralNetwork, ProtocolError};
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), ProtocolError> {
    println!("🧪 Starting test client for OutputServer");

    // Create a simple neural network
    let network = NeuralNetwork::new(4, 2, 1, 0.1);
    
    // Create a distributed network that will connect to the output server
    let (mut distributed_network, _receiver) = DistributedNetwork::new(
        "TestClient".to_string(),
        "127.0.0.1".to_string(),
        8003, // Use a different port for this client
        network,
    );

    // Start the client
    distributed_network.start_server().await?;
    println!("✅ Test client started on port 8003");

    // Connect to the output server
    println!("🔗 Connecting to OutputServer at 127.0.0.1:8002");
    distributed_network.connect_to("127.0.0.1", 8002).await?;
    println!("✅ Connected to OutputServer");

    // Send some test data periodically
    for i in 0..20 {
        let test_data = vec![
            (i as f32 * 0.1) % 1.0,
            ((i as f32 * 0.15) % 1.0).sin().abs(),
            ((i as f32 * 0.2) % 1.0).cos().abs(),
            (i as f32 * 0.05) % 1.0,
        ];
        
        println!("📤 Sending test data #{}: {:?}", i + 1, test_data);
        
        // Send the data using the distributed network's forward data mechanism
        // We'll create a simple forward message
        use neural_network::{NetworkMessage, MessageType, MessagePayload};
        
        let message = NetworkMessage {
            msg_type: MessageType::ForwardData,
            sequence: i as u64,
            payload: MessagePayload::ForwardData {
                layer_id: 0,
                data: test_data,
            },
        };
        
        distributed_network.handle_message(message).await?;
        
        // Wait 2 seconds before sending next data
        sleep(Duration::from_secs(2)).await;
    }

    println!("🏁 Test completed");
    Ok(())
}