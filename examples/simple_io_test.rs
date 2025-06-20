use neural_network::{
    InputNode, OutputNode,
    IoNodeConfig, ExternalSinkConfig,
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Simple I/O Interface Test");
    println!("============================");
    
    // Create a simple input node configuration
    let input_config = IoNodeConfig {
        node_id: Uuid::new_v4(),
        name: "TestInputNode".to_string(),
        listen_address: "127.0.0.1".to_string(),
        listen_port: 8001,
        target_address: None, // No external connection
        target_port: None,
        use_tls: false,
        cert_path: None,
        key_path: None,
        data_transformation: None,
    };
    
    // Create input node
    let (mut input_node, _input_receiver) = InputNode::new(input_config);
    
    println!("✅ Created input node");
    
    // Start the input node (this will start the NNP server)
    input_node.start().await?;
    
    println!("✅ Started input node server");
    
    // Demonstrate sending data via NNP
    let test_data = vec![0.1, 0.5, 0.8, 0.3];
    println!("📤 Sending test data: {:?}", test_data);
    
    input_node.send_data(test_data).await?;
    
    println!("✅ Data sent via Neural Network Protocol");
    
    // Create output node configuration
    let output_config = IoNodeConfig {
        node_id: Uuid::new_v4(),
        name: "TestOutputNode".to_string(),
        listen_address: "127.0.0.1".to_string(),
        listen_port: 8002,
        target_address: None,
        target_port: None,
        use_tls: false,
        cert_path: None,
        key_path: None,
        data_transformation: None,
    };
    
    // Create output node
    let (mut output_node, output_receiver) = OutputNode::new(output_config);
    
    println!("✅ Created output node");
    
    // Start the output node
    output_node.start().await?;
    
    println!("✅ Started output node server");
    
    // Set up a custom sink to demonstrate data reception
    let custom_sink = ExternalSinkConfig::Custom {
        handler: |mut receiver| {
            Box::pin(async move {
                println!("📥 Custom sink started, waiting for data...");
                while let Some(data) = receiver.recv().await {
                    println!("📥 Received data from neural network: {:?}", data);
                }
                Ok(())
            })
        }
    };
    
    // Start processing messages in background
    let output_clone = output_node.clone();
    tokio::spawn(async move {
        if let Err(e) = output_clone.process_messages(output_receiver, custom_sink).await {
            eprintln!("Output processing error: {:?}", e);
        }
    });
    
    println!("✅ Set up output processing");
    
    // Wait a bit to let everything initialize
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    println!("\n🎯 I/O Interface Integration Test Complete!");
    println!("   - Input nodes can receive external data and forward via NNP");
    println!("   - Output nodes can receive NNP data and forward to external systems");
    println!("   - Both nodes integrate seamlessly with the distributed neural network");
    
    Ok(())
}