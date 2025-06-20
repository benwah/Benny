use neural_network::{DistributedNetwork, NeuralNetwork};
use std::time::Duration;
use tokio::time::sleep;
use clap::{Parser, ValueEnum};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "topology_tester")]
#[command(about = "Test neural network topologies in Docker containers")]
pub struct Args {
    /// Topology to test
    #[arg(short, long, default_value = "linear")]
    topology: TopologyType,
    
    /// Test duration in seconds
    #[arg(short, long, default_value = "60")]
    duration: u64,
    
    /// Data injection rate (messages per second)
    #[arg(short, long, default_value = "1.0")]
    rate: f64,
    
    /// Entry point nodes (comma-separated host:port)
    #[arg(short, long)]
    entry_points: Option<String>,
    
    /// Test data pattern
    #[arg(short, long, default_value = "random")]
    pattern: DataPattern,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(ValueEnum, Clone, Debug)]
enum TopologyType {
    Linear,
    Star,
    Mesh,
    Ring,
    Custom,
}

#[derive(ValueEnum, Clone, Debug)]
enum DataPattern {
    Random,
    Sine,
    Step,
    Pulse,
    XOR,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    println!("🧪 Neural Network Topology Tester");
    println!("==================================");
    println!("Topology: {:?}", args.topology);
    println!("Duration: {} seconds", args.duration);
    println!("Data rate: {} Hz", args.rate);
    println!("Pattern: {:?}", args.pattern);
    println!();

    // Determine entry points based on topology
    let entry_points = if let Some(custom_points) = args.entry_points {
        parse_entry_points(&custom_points)?
    } else {
        get_default_entry_points(&args.topology)
    };

    println!("📡 Entry points:");
    for (name, addr) in &entry_points {
        println!("  - {}: {}", name, addr);
    }
    println!();

    // Create test client
    let test_network = NeuralNetwork::with_layers(&[4, 2, 1], 0.01);
    let (test_client, mut _receiver) = DistributedNetwork::new(
        "topology-tester".to_string(),
        "0.0.0.0".to_string(),
        9999, // Use a different port for the test client
        test_network,
    );

    // Connect to entry points
    let mut connections = HashMap::new();
    for (name, addr) in &entry_points {
        let parts: Vec<&str> = addr.split(':').collect();
        if parts.len() != 2 {
            eprintln!("❌ Invalid address format: {}", addr);
            continue;
        }
        
        let host = parts[0];
        let port: u16 = parts[1].parse()?;
        
        println!("🔗 Connecting to {}: {}:{}", name, host, port);
        
        // Wait a bit for the servers to be ready
        sleep(Duration::from_secs(2)).await;
        
        match test_client.connect_to(host, port).await {
            Ok(peer_id) => {
                println!("✅ Connected to {} (ID: {})", name, peer_id);
                connections.insert(name.clone(), peer_id);
            }
            Err(e) => {
                println!("❌ Failed to connect to {}: {:?}", name, e);
            }
        }
    }

    if connections.is_empty() {
        eprintln!("❌ No connections established. Exiting.");
        return Ok(());
    }

    println!();
    println!("🚀 Starting data injection test...");
    
    let interval = Duration::from_secs_f64(1.0 / args.rate);
    let end_time = std::time::Instant::now() + Duration::from_secs(args.duration);
    let mut message_count = 0;

    while std::time::Instant::now() < end_time {
        // Generate test data based on pattern
        let test_data = generate_test_data(&args.pattern, message_count);
        
        if args.verbose {
            println!("📤 Sending data: {:?}", test_data);
        }

        // Send data to all connected entry points
        for (name, peer_id) in &connections {
            if let Err(e) = test_client.send_forward_data(*peer_id, 0, test_data.clone()).await {
                eprintln!("❌ Failed to send data to {}: {:?}", name, e);
            } else if args.verbose {
                println!("✅ Sent to {}", name);
            }
        }

        message_count += 1;
        if message_count % 10 == 0 {
            println!("📊 Sent {} messages", message_count);
        }

        sleep(interval).await;
    }

    println!();
    println!("🎉 Test completed!");
    println!("📊 Total messages sent: {}", message_count);
    println!("⏱️  Test duration: {} seconds", args.duration);
    println!("📈 Average rate: {:.2} Hz", message_count as f64 / args.duration as f64);

    Ok(())
}

fn parse_entry_points(input: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut points = HashMap::new();
    for (i, addr) in input.split(',').enumerate() {
        let addr = addr.trim();
        if !addr.is_empty() {
            points.insert(format!("entry-{}", i + 1), addr.to_string());
        }
    }
    Ok(points)
}

fn get_default_entry_points(topology: &TopologyType) -> HashMap<String, String> {
    let mut points = HashMap::new();
    
    match topology {
        TopologyType::Linear => {
            points.insert("node-1".to_string(), "localhost:8081".to_string());
        }
        TopologyType::Star => {
            points.insert("node-1".to_string(), "localhost:8081".to_string());
            points.insert("node-2".to_string(), "localhost:8082".to_string());
            points.insert("node-3".to_string(), "localhost:8083".to_string());
            points.insert("node-4".to_string(), "localhost:8084".to_string());
        }
        TopologyType::Mesh => {
            points.insert("node-1".to_string(), "localhost:8081".to_string());
            points.insert("node-2".to_string(), "localhost:8082".to_string());
        }
        TopologyType::Ring => {
            points.insert("node-1".to_string(), "localhost:8081".to_string());
        }
        TopologyType::Custom => {
            points.insert("custom-1".to_string(), "localhost:8081".to_string());
        }
    }
    
    points
}

fn generate_test_data(pattern: &DataPattern, count: usize) -> Vec<f64> {
    match pattern {
        DataPattern::Random => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            (0..4).map(|_| rng.gen_range(-1.0..1.0)).collect()
        }
        DataPattern::Sine => {
            let t = count as f64 * 0.1;
            vec![
                (t).sin(),
                (t * 1.5).sin(),
                (t * 2.0).sin(),
                (t * 0.5).sin(),
            ]
        }
        DataPattern::Step => {
            let step = (count / 10) % 4;
            let mut data = vec![0.0; 4];
            data[step] = 1.0;
            data
        }
        DataPattern::Pulse => {
            if count % 20 < 5 {
                vec![1.0, 1.0, 1.0, 1.0]
            } else {
                vec![0.0, 0.0, 0.0, 0.0]
            }
        }
        DataPattern::XOR => {
            let patterns = [
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0, 1.0],
                vec![1.0, 1.0, 0.0, 0.0],
            ];
            patterns[count % patterns.len()].clone()
        }
    }
}