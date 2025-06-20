use neural_network::{
    NeuralNetwork, HebbianLearningMode,
    IoManager, TcpInputInterface, TcpOutputInterface, IoConfig, IoData
};
use tokio_rustls::{TlsConnector, rustls::{ClientConfig, RootCertStore}};
use std::sync::Arc;
use uuid::Uuid;
use std::collections::HashMap;

/// Example demonstrating how to use the I/O interfaces for distributed neural networks
/// 
/// This example shows:
/// 1. Creating a neural network
/// 2. Setting up TLS-enabled input and output interfaces
/// 3. Connecting to external systems via SSL/TCP
/// 4. Processing data through the neural network
/// 5. Managing multiple I/O connections
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Network I/O Interface Example");
    println!("========================================");
    
    // Create a simple neural network
    let network = NeuralNetwork::new_with_mode(4, 8, 2, 0.1, HebbianLearningMode::Oja);
    
    println!("✅ Created neural network with architecture: [4, 8, 2]");
    
    // Create I/O manager
    let mut io_manager = IoManager::new(network);
    
    // Setup TLS configuration (in a real scenario, you'd load proper certificates)
    let mut root_store = RootCertStore::empty();
    // In production, add your CA certificates here
    // root_store.add_server_trust_anchors(webpki_roots::TLS_SERVER_ROOTS.0.iter().map(|ta| {
    //     rustls::OwnedTrustAnchor::from_subject_spki_name_constraints(
    //         ta.subject, ta.spki, ta.name_constraints,
    //     )
    // }));
    
    let config = ClientConfig::builder()
        .with_safe_defaults()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    
    let tls_connector = TlsConnector::from(Arc::new(config));
    
    // Create input interfaces
    let input_id_1 = Uuid::new_v4();
    let input_interface_1 = TcpInputInterface::with_tls_connector(tls_connector.clone());
    io_manager.add_input_interface(input_id_1, Box::new(input_interface_1));
    
    let input_id_2 = Uuid::new_v4();
    let input_interface_2 = TcpInputInterface::with_tls_connector(tls_connector.clone());
    io_manager.add_input_interface(input_id_2, Box::new(input_interface_2));
    
    // Create output interfaces
    let output_id_1 = Uuid::new_v4();
    let output_interface_1 = TcpOutputInterface::with_tls_connector(tls_connector.clone());
    io_manager.add_output_interface(output_id_1, Box::new(output_interface_1));
    
    let output_id_2 = Uuid::new_v4();
    let output_interface_2 = TcpOutputInterface::with_tls_connector(tls_connector.clone());
    io_manager.add_output_interface(output_id_2, Box::new(output_interface_2));
    
    println!("✅ Created 2 input and 2 output interfaces");
    
    // Example configurations for connecting to external systems
    // Note: These are example endpoints - in practice you'd connect to real systems
    let input_config_1 = IoConfig {
        connection_id: input_id_1,
        endpoint: "sensor-data.example.com".to_string(),
        port: 8443,
        use_tls: true,
        cert_path: Some("/path/to/client.crt".to_string()),
        key_path: Some("/path/to/client.key".to_string()),
        buffer_size: 1024,
        timeout_ms: 5000,
    };
    
    let input_config_2 = IoConfig {
        connection_id: input_id_2,
        endpoint: "api-gateway.example.com".to_string(),
        port: 9443,
        use_tls: true,
        cert_path: Some("/path/to/client.crt".to_string()),
        key_path: Some("/path/to/client.key".to_string()),
        buffer_size: 2048,
        timeout_ms: 3000,
    };
    
    let output_config_1 = IoConfig {
        connection_id: output_id_1,
        endpoint: "actuator-control.example.com".to_string(),
        port: 8444,
        use_tls: true,
        cert_path: Some("/path/to/client.crt".to_string()),
        key_path: Some("/path/to/client.key".to_string()),
        buffer_size: 512,
        timeout_ms: 2000,
    };
    
    let output_config_2 = IoConfig {
        connection_id: output_id_2,
        endpoint: "data-logger.example.com".to_string(),
        port: 8445,
        use_tls: true,
        cert_path: Some("/path/to/client.crt".to_string()),
        key_path: Some("/path/to/client.key".to_string()),
        buffer_size: 1024,
        timeout_ms: 4000,
    };
    
    println!("📋 Configured connection parameters for external systems");
    
    // In a real scenario, you would connect to actual endpoints:
    // 
    // // Connect input interfaces
    // match io_manager.connect_input(input_id_1, input_config_1).await {
    //     Ok(_) => println!("✅ Connected input interface 1 to sensor data source"),
    //     Err(e) => println!("❌ Failed to connect input interface 1: {}", e),
    // }
    // 
    // match io_manager.connect_input(input_id_2, input_config_2).await {
    //     Ok(_) => println!("✅ Connected input interface 2 to API gateway"),
    //     Err(e) => println!("❌ Failed to connect input interface 2: {}", e),
    // }
    // 
    // // Connect output interfaces
    // match io_manager.connect_output(output_id_1, output_config_1).await {
    //     Ok(_) => println!("✅ Connected output interface 1 to actuator control"),
    //     Err(e) => println!("❌ Failed to connect output interface 1: {}", e),
    // }
    // 
    // match io_manager.connect_output(output_id_2, output_config_2).await {
    //     Ok(_) => println!("✅ Connected output interface 2 to data logger"),
    //     Err(e) => println!("❌ Failed to connect output interface 2: {}", e),
    // }
    // 
    // // Start processing data through the neural network
    // match io_manager.start_processing().await {
    //     Ok(_) => println!("🚀 Started neural network I/O processing"),
    //     Err(e) => println!("❌ Failed to start processing: {}", e),
    // }
    
    // For demonstration, show the connection status
    let status = io_manager.get_status();
    println!("\n📊 Connection Status:");
    for (id, connected) in status {
        let status_icon = if connected { "🟢" } else { "🔴" };
        println!("  {} Connection {}: {}", status_icon, id, if connected { "Connected" } else { "Disconnected" });
    }
    
    // Demonstrate manual data processing
    println!("\n🔄 Demonstrating manual data processing:");
    
    // Simulate input data
    let sample_input = IoData {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        values: vec![0.5, 0.8, 0.2, 0.9],
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("source".to_string(), "sensor_array_1".to_string());
            meta.insert("type".to_string(), "environmental_data".to_string());
            meta
        },
    };
    
    println!("📥 Input data: {:?}", sample_input.values);
    
    // Process through neural network (simulated)
    // In the real IoManager, this would happen automatically
    // Here we demonstrate the concept manually
    println!("🧠 Processing through neural network...");
    
    // Simulate output data
    let sample_output = IoData {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        values: vec![0.7, 0.3], // Neural network output
        metadata: sample_input.metadata.clone(),
    };
    
    println!("📤 Output data: {:?}", sample_output.values);
    
    println!("\n🎯 Use Cases for I/O Interfaces:");
    println!("  • Sensor data ingestion from IoT devices");
    println!("  • Real-time API data processing");
    println!("  • Actuator control based on neural network decisions");
    println!("  • Data logging and monitoring");
    println!("  • Integration with existing enterprise systems");
    println!("  • Distributed neural network coordination");
    
    println!("\n🔒 Security Features:");
    println!("  • TLS/SSL encryption for all connections");
    println!("  • Certificate-based authentication");
    println!("  • Configurable timeouts and buffer sizes");
    println!("  • Connection status monitoring");
    
    println!("\n⚡ Performance Features:");
    println!("  • Asynchronous, non-blocking I/O");
    println!("  • Multiple concurrent connections");
    println!("  • Streaming data processing");
    println!("  • Configurable buffer management");
    
    Ok(())
}

/// Example of a custom input interface implementation
/// This shows how you could extend the system for specific use cases
pub struct CustomSensorInterface {
    // Custom fields for your specific sensor type
    sensor_type: String,
    calibration_data: Vec<f64>,
    // ... other fields
}

impl CustomSensorInterface {
    pub fn new(sensor_type: String, calibration_data: Vec<f64>) -> Self {
        Self {
            sensor_type,
            calibration_data,
        }
    }
    
    // Custom transformation logic for your specific sensor
    fn apply_calibration(&self, raw_values: &[f64]) -> Vec<f64> {
        raw_values.iter()
            .zip(self.calibration_data.iter())
            .map(|(raw, cal)| raw * cal)
            .collect()
    }
}

// You would implement the InputInterface trait for your custom interface
// This allows seamless integration with the IoManager system