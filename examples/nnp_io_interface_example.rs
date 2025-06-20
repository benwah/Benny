use neural_network::{
    InputNode, OutputNode, SecureInputNode,
    IoNodeConfig, ExternalSourceConfig, ExternalSinkConfig,
};
use neural_network::distributed_network::NetworkId;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Network Protocol (NNP) I/O Interface Example");
    println!("=====================================================");
    
    // Create configurations for I/O nodes
    let input_config = IoNodeConfig {
        node_id: Uuid::new_v4(),
        name: "SensorInputNode".to_string(),
        listen_address: "127.0.0.1".to_string(),
        listen_port: 8001,
        target_address: Some("127.0.0.1".to_string()),
        target_port: Some(8000), // Connect to main neural network
        use_tls: false,
        cert_path: None,
        key_path: None,
        data_transformation: None,
    };
    
    let output_config = IoNodeConfig {
        node_id: Uuid::new_v4(),
        name: "ActuatorOutputNode".to_string(),
        listen_address: "127.0.0.1".to_string(),
        listen_port: 8002,
        target_address: None,
        target_port: None,
        use_tls: false,
        cert_path: None,
        key_path: None,
        data_transformation: None,
    };
    
    println!("✅ Created I/O node configurations");
    
    // Create input node that will receive sensor data and forward via NNP
    let (mut input_node, _input_receiver) = InputNode::new(input_config);
    
    // Create output node that will receive NNP data and forward to actuators
    let (mut output_node, output_receiver) = OutputNode::new(output_config);
    
    println!("✅ Created input and output nodes");
    
    // Start the nodes
    input_node.start().await?;
    output_node.start().await?;
    
    println!("✅ Started I/O nodes on ports 8001 and 8002");
    
    // Configure external data sources and sinks
    let sensor_source = ExternalSourceConfig::HttpEndpoint {
        url: "http://sensor-api.example.com/data".to_string(),
        poll_interval: 1000, // Poll every second
    };
    
    let actuator_sink = ExternalSinkConfig::TcpSocket {
        address: "actuator-controller.local".to_string(),
        port: 9000,
    };
    
    // Connect external systems
    input_node.connect_external_source(sensor_source).await?;
    
    // Start output processing in background
    let output_node_clone = output_node.clone();
    tokio::spawn(async move {
        if let Err(e) = output_node_clone.process_messages(output_receiver, actuator_sink).await {
            eprintln!("Output processing error: {:?}", e);
        }
    });
    
    println!("✅ Connected external sensor and actuator systems");
    
    println!("🔄 Started data processing pipeline");
    
    // Demonstrate the data flow
    println!("\n📊 Data Flow Architecture:");
    println!("  External Sensor → InputNode → NNP → DistributedNetwork → NNP → OutputNode → External Actuator");
    println!("  
    ┌─────────────┐    NNP     ┌──────────────────┐    NNP     ┌─────────────┐
    │ Sensor Data │ ────────→  │ Distributed NN   │ ────────→  │ Actuator    │
    │ (HTTP/TCP)  │  Protocol  │ (Main Processing) │  Protocol  │ Control     │
    └─────────────┘            └──────────────────┘            └─────────────┘
    ");
    
    // Show how this integrates with existing distributed neural networks
    println!("\n🌐 Integration with Distributed Neural Networks:");
    println!("  • Input nodes appear as regular NN nodes to the distributed network");
    println!("  • Output nodes receive data just like any other NN node");
    println!("  • Uses existing NNP protocol for all communication");
    println!("  • Leverages existing security and reliability features");
    println!("  • Can be mixed with regular neural network nodes seamlessly");
    
    // Demonstrate secure version
    println!("\n🔒 Secure I/O Nodes (TLS/SSL):");
    
    let secure_input_config = IoNodeConfig {
        node_id: Uuid::new_v4(),
        name: "SecureSensorNode".to_string(),
        listen_address: "127.0.0.1".to_string(),
        listen_port: 8443,
        target_address: Some("127.0.0.1".to_string()),
        target_port: Some(8000),
        use_tls: true,
        cert_path: Some("certs/sensor.crt".to_string()),
        key_path: Some("certs/sensor.key".to_string()),
        data_transformation: None,
    };
    
    let (_secure_input, _secure_receiver) = SecureInputNode::new(secure_input_config);
    
    // In a real deployment, you would use SecureDistributedNetwork for TLS
    
    println!("✅ Secure I/O nodes configured (would use TLS certificates in production)");
    
    // Show use cases
    println!("\n🎯 Real-World Use Cases:");
    println!("  
    IoT Deployment:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Temperature │    │ Distributed │    │ HVAC        │
    │ Sensors     │ ──→│ Neural Net  │──→ │ Controller  │
    └─────────────┘    └─────────────┘    └─────────────┘
    
    Industrial Control:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Process     │    │ Predictive  │    │ Valve/Motor │
    │ Monitoring  │ ──→│ Control NN  │──→ │ Control     │
    └─────────────┘    └─────────────┘    └─────────────┘
    
    Edge Computing:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Camera/     │    │ Edge Neural │    │ Alert/      │
    │ Microphone  │ ──→│ Network     │──→ │ Response    │
    └─────────────┘    └─────────────┘    └─────────────┘
    ");
    
    println!("\n⚡ Key Benefits:");
    println!("  • Seamless integration with existing distributed NN infrastructure");
    println!("  • Uses proven NNP protocol for reliability and performance");
    println!("  • Maintains security model with TLS/SSL support");
    println!("  • No changes needed to existing neural network code");
    println!("  • Can scale horizontally with multiple I/O nodes");
    println!("  • Supports both input and output scenarios");
    
    // Keep the example running for a bit to show data flow
    println!("\n⏱️  Running for 10 seconds to demonstrate data flow...");
    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
    
    println!("✅ Example completed successfully!");
    
    Ok(())
}

/// Example of creating a custom data transformation
async fn custom_sensor_transformation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Custom Data Transformation Example:");
    
    // Create a custom handler for specialized sensor data
    let custom_source = ExternalSourceConfig::Custom {
        handler: |sender| Box::pin(async move {
            // Simulate specialized sensor that sends structured data
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(500));
            
            for i in 0..20 {
                interval.tick().await;
                
                // Transform complex sensor data into neural network input format
                let temperature = 20.0 + (i as f64 * 0.5);
                let humidity = 50.0 + (i as f64 * 0.3);
                let pressure = 1013.25 + (i as f64 * 0.1);
                
                // Normalize to [0, 1] range for neural network
                let normalized_data = vec![
                    (temperature - 15.0) / 30.0,  // Temperature range 15-45°C
                    humidity / 100.0,             // Humidity 0-100%
                    (pressure - 1000.0) / 50.0,   // Pressure range 1000-1050 hPa
                ];
                
                if sender.send(normalized_data).await.is_err() {
                    break;
                }
            }
            
            Ok(())
        }),
    };
    
    println!("  ✅ Custom sensor transformation configured");
    println!("  📊 Transforms: Temperature, Humidity, Pressure → Neural Network Input");
    
    Ok(())
}