use neural_network::{DistributedNetwork, NeuralNetwork, NetworkMessage, MessageType, MessagePayload};
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Starting Neural Network that sends outputs to OutputServer");

    // Create a neural network
    let network = NeuralNetwork::new(4, 2, 1, 0.1);
    
    // Create a distributed network
    let (mut distributed_network, _receiver) = DistributedNetwork::new(
        "OutputProducer".to_string(),
        "127.0.0.1".to_string(),
        8004, // Use a different port
        network,
    );

    // Start the network
    distributed_network.start_server().await.map_err(|e| format!("Failed to start server: {:?}", e))?;
    println!("✅ Neural Network started on port 8004");

    // Connect to the output server
    println!("🔗 Connecting to OutputServer at 127.0.0.1:8002");
    distributed_network.connect_to("127.0.0.1", 8002).await.map_err(|e| format!("Failed to connect: {:?}", e))?;
    println!("✅ Connected to OutputServer");

    // Simulate neural network processing and sending outputs
    for i in 0..50 {
        // Simulate some inputs to the neural network (4 inputs as expected)
        let inputs = vec![
            (i as f32 * 0.1) % 1.0,
            ((i as f32 * 0.15) % 1.0).sin().abs(),
            ((i as f32 * 0.2) % 1.0).cos().abs(),
            (i as f32 * 0.05) % 1.0,
        ];
        
        println!("🧠 Processing inputs #{}: {:?}", i + 1, inputs);
        
        // Send the inputs to the output server (which will process them and display the outputs)
        distributed_network.send_forward_data(0, inputs).await.map_err(|e| format!("Failed to send data: {:?}", e))?;
        
        // Wait 1 second before sending next inputs
        sleep(Duration::from_secs(1)).await;
    }

    println!("🏁 Neural Network simulation completed");
    Ok(())
}