use neural_network::{DistributedNetwork, HebbianLearningMode, NeuralNetwork};
use tokio::time::{Duration, sleep};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Distributed Neural Network Protocol Demo");
    println!("==========================================");
    println!("Optimized TCP-based protocol for neural network communication");
    println!();

    // Create two neural networks that will communicate
    let network1 =
        NeuralNetwork::with_layers_and_mode(&[2, 4, 2], 0.1, HebbianLearningMode::Classic);

    let network2 =
        NeuralNetwork::with_layers_and_mode(&[2, 3, 1], 0.1, HebbianLearningMode::Competitive);

    // Create distributed network nodes
    let (dist_net1, mut receiver1) = DistributedNetwork::new(
        "AlphaNet".to_string(),
        "127.0.0.1".to_string(),
        8001,
        network1,
    );

    let (dist_net2, mut receiver2) = DistributedNetwork::new(
        "BetaNet".to_string(),
        "127.0.0.1".to_string(),
        8002,
        network2,
    );

    println!("🚀 Starting Neural Network Protocol servers...");
    println!("📡 AlphaNet: {} (ID: {})", "127.0.0.1:8001", dist_net1.id);
    println!("📡 BetaNet: {} (ID: {})", "127.0.0.1:8002", dist_net2.id);
    println!("🧠 Protocol: NNP v1 (Neural Network Protocol)");
    println!();

    // Start servers
    let server1_handle = {
        let dist_net1_clone = dist_net1.clone();
        tokio::spawn(async move {
            if let Err(e) = dist_net1_clone.start_server().await {
                println!("❌ AlphaNet server error: {:?}", e);
            }
        })
    };

    let server2_handle = {
        let dist_net2_clone = dist_net2.clone();
        tokio::spawn(async move {
            if let Err(e) = dist_net2_clone.start_server().await {
                println!("❌ BetaNet server error: {:?}", e);
            }
        })
    };

    // Give servers time to start
    sleep(Duration::from_millis(100)).await;

    // Start message handlers
    let handler1 = {
        let dist_net1_clone = dist_net1.clone();
        tokio::spawn(async move {
            while let Some(message) = receiver1.recv().await {
                if let Err(e) = dist_net1_clone.handle_message(message).await {
                    println!("❌ AlphaNet message handling error: {:?}", e);
                }
            }
        })
    };

    let handler2 = {
        let dist_net2_clone = dist_net2.clone();
        tokio::spawn(async move {
            while let Some(message) = receiver2.recv().await {
                if let Err(e) = dist_net2_clone.handle_message(message).await {
                    println!("❌ BetaNet message handling error: {:?}", e);
                }
            }
        })
    };

    // Give servers more time to fully initialize
    sleep(Duration::from_millis(500)).await;

    println!("🔗 Establishing connections...");

    // Connect AlphaNet to BetaNet
    match dist_net1.connect_to("127.0.0.1", 8002).await {
        Ok(peer_id) => {
            println!("✅ AlphaNet connected to BetaNet (ID: {})", peer_id);
        }
        Err(e) => {
            println!("❌ Connection failed: {:?}", e);
            return Ok(());
        }
    }

    sleep(Duration::from_millis(200)).await;

    println!();
    println!("📊 Protocol Performance Characteristics:");
    println!("   • Binary protocol with CRC32 checksums");
    println!("   • f32 precision for network efficiency");
    println!("   • Message sequencing for ordering");
    println!("   • Capability negotiation");
    println!("   • Optimized for real-time neural data");
    println!();

    // Demonstrate data exchange
    println!("🧬 Testing neural data exchange...");

    // Send forward propagation data
    let test_data = vec![0.5, 0.8];
    println!("📤 AlphaNet sending forward data: {:?}", test_data);

    if let Err(e) = dist_net1
        .send_forward_data(
            dist_net2.id,
            0, // layer 0
            test_data.clone(),
        )
        .await
    {
        println!("❌ Failed to send forward data: {:?}", e);
    }

    sleep(Duration::from_millis(100)).await;

    // Send Hebbian correlation data
    let correlations = vec![0.7, 0.3, 0.9, 0.1];
    println!(
        "📤 AlphaNet sending Hebbian correlations: {:?}",
        correlations
    );

    if let Err(e) = dist_net1
        .send_hebbian_data(
            dist_net2.id,
            1, // layer 1
            correlations,
            0.1, // learning rate
        )
        .await
    {
        println!("❌ Failed to send Hebbian data: {:?}", e);
    }

    sleep(Duration::from_millis(100)).await;

    println!();
    println!("🔬 Protocol Analysis:");
    println!("   • Header size: 22 bytes (magic + version + type + length + sequence + checksum)");
    println!("   • Payload: Variable length, optimized binary encoding");
    println!("   • Checksum: CRC32 for data integrity");
    println!("   • Sequence: 64-bit counter for message ordering");
    println!("   • Magic: 'NNP\\0' for protocol identification");
    println!();

    println!("🎯 Capabilities Demonstrated:");
    println!("   • Forward propagation data streaming");
    println!("   • Hebbian correlation sharing");
    println!("   • Real-time neural network communication");
    println!("   • Distributed learning coordination");
    println!("   • Network topology discovery");
    println!();

    println!("🌟 Use Cases:");
    println!("   • Distributed neural network training");
    println!("   • Real-time inference across machines");
    println!("   • Federated Hebbian learning");
    println!("   • Neural network clustering");
    println!("   • Edge computing coordination");
    println!();

    // Let the demo run for a bit
    sleep(Duration::from_secs(2)).await;

    println!("🎉 Distributed Neural Network Protocol demo complete!");
    println!("💡 Networks can now communicate across the internet using optimized binary protocol");

    // Clean shutdown
    server1_handle.abort();
    server2_handle.abort();
    handler1.abort();
    handler2.abort();

    Ok(())
}
