use tokio::net::TcpStream;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Starting simple neural network output test");

    // Connect to the output server
    println!("🔗 Connecting to OutputServer at 127.0.0.1:8002");
    let mut stream = TcpStream::connect("127.0.0.1:8002").await?;
    println!("✅ Connected to OutputServer");

    // Simulate neural network outputs
    for i in 0..20 {
        // Generate some test outputs (2 values as expected by the output server)
        let outputs = vec![
            (i as f64 * 0.1).sin().abs(),
            (i as f64 * 0.15).cos().abs(),
        ];
        
        // Convert to JSON
        let json_data = serde_json::to_string(&outputs)?;
        println!("📤 Sending outputs #{}: {}", i + 1, json_data);
        
        // Send the JSON data
        stream.write_all(json_data.as_bytes()).await?;
        stream.write_all(b"\n").await?; // Add newline for clarity
        
        // Read acknowledgment
        let mut buffer = vec![0; 64];
        match stream.read(&mut buffer).await {
            Ok(n) if n > 0 => {
                let response = String::from_utf8_lossy(&buffer[..n]);
                println!("📥 Server response: {}", response.trim());
            }
            _ => {
                println!("⚠️ No response from server");
            }
        }
        
        // Wait 1 second before sending next outputs
        sleep(Duration::from_secs(1)).await;
    }

    println!("🏁 Test completed");
    Ok(())
}