use neural_network::{InputNode, IoNodeConfig};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test with matching input size
    println!("Test 1: Matching input size (should succeed)");
    let config = IoNodeConfig {
        node_id: Uuid::new_v4(),
        name: "TestInputNode".to_string(),
        listen_address: "127.0.0.1".to_string(),
        listen_port: 0, // Let the OS assign a port
        target_address: None,
        target_port: None,
        use_tls: false,
        cert_path: None,
        key_path: None,
        data_transformation: None,
        input_size: 4,
    };
    
    let (input_node, _) = InputNode::new(config);
    
    // This should succeed
    match input_node.send_data(vec![0.1, 0.2, 0.3, 0.4]).await {
        Ok(_) => println!("✅ Test 1 passed: Successfully sent data with matching input size"),
        Err(e) => println!("❌ Test 1 failed: {}", e),
    }
    
    // Wait a bit
    sleep(Duration::from_millis(500)).await;
    
    // Test with mismatched input size
    println!("\nTest 2: Mismatched input size (should fail)");
    let config = IoNodeConfig {
        node_id: Uuid::new_v4(),
        name: "TestInputNode".to_string(),
        listen_address: "127.0.0.1".to_string(),
        listen_port: 0, // Let the OS assign a port
        target_address: None,
        target_port: None,
        use_tls: false,
        cert_path: None,
        key_path: None,
        data_transformation: None,
        input_size: 4,
    };
    
    let (input_node, _) = InputNode::new(config);
    
    // This should fail with an input size mismatch error
    match input_node.send_data(vec![0.1, 0.2, 0.3, 0.4, 0.5]).await {
        Ok(_) => println!("❌ Test 2 failed: Should have rejected mismatched input size"),
        Err(e) => println!("✅ Test 2 passed: Correctly rejected mismatched input size: {}", e),
    }
    
    // Test with different input size
    println!("\nTest 3: Different input size configuration (should succeed)");
    let config = IoNodeConfig {
        node_id: Uuid::new_v4(),
        name: "TestInputNode".to_string(),
        listen_address: "127.0.0.1".to_string(),
        listen_port: 0, // Let the OS assign a port
        target_address: None,
        target_port: None,
        use_tls: false,
        cert_path: None,
        key_path: None,
        data_transformation: None,
        input_size: 16,
    };
    
    let (input_node, _) = InputNode::new(config);
    
    // Create a vector with 16 elements
    let data: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
    
    // This should succeed
    match input_node.send_data(data).await {
        Ok(_) => println!("✅ Test 3 passed: Successfully sent data with input_size=16"),
        Err(e) => println!("❌ Test 3 failed: {}", e),
    }
    
    println!("\nAll tests completed!");
    
    Ok(())
}