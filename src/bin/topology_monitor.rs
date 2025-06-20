use clap::Parser;
// use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use warp::Filter;

#[derive(Parser)]
#[command(name = "topology_monitor")]
#[command(about = "Monitor neural network topologies in real-time")]
pub struct Args {
    /// Web server port
    #[arg(short, long, default_value = "3000")]
    port: u16,
    
    /// Neural network nodes to monitor (comma-separated host:port)
    #[arg(short, long)]
    nodes: Option<String>,
    
    /// Topology configuration file
    #[arg(short, long)]
    topology: Option<String>,
    
    /// Update interval in seconds
    #[arg(short, long, default_value = "1")]
    interval: u64,
}

#[derive(Clone, serde::Serialize)]
struct NodeStatus {
    name: String,
    address: String,
    status: String,
    last_seen: String,
    connections: Vec<String>,
    message_count: u64,
    error_count: u64,
}

#[derive(Clone, serde::Serialize)]
struct TopologyStatus {
    nodes: HashMap<String, NodeStatus>,
    connections: Vec<(String, String)>,
    total_messages: u64,
    uptime: u64,
}

type SharedState = Arc<RwLock<TopologyStatus>>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    println!("📊 Neural Network Topology Monitor");
    println!("==================================");
    println!("Web interface: http://localhost:{}", args.port);
    println!("Update interval: {} seconds", args.interval);
    println!();

    // Initialize shared state
    let state: SharedState = Arc::new(RwLock::new(TopologyStatus {
        nodes: HashMap::new(),
        connections: Vec::new(),
        total_messages: 0,
        uptime: 0,
    }));

    // Parse nodes to monitor
    let nodes = if let Some(nodes_str) = args.nodes {
        parse_nodes(&nodes_str)?
    } else {
        get_default_nodes()
    };

    println!("🔍 Monitoring nodes:");
    for (name, addr) in &nodes {
        println!("  - {}: {}", name, addr);
    }
    println!();

    // Start monitoring task
    let monitor_state = state.clone();
    let monitor_nodes = nodes.clone();
    tokio::spawn(async move {
        monitor_topology(monitor_state, monitor_nodes, args.interval).await;
    });

    // Setup web routes
    let state_filter = warp::any().map(move || state.clone());

    let api_status = warp::path("api")
        .and(warp::path("status"))
        .and(warp::path::end())
        .and(state_filter.clone())
        .and_then(get_status);

    let api_nodes = warp::path("api")
        .and(warp::path("nodes"))
        .and(warp::path::end())
        .and(state_filter.clone())
        .and_then(get_nodes);

    let static_files = warp::path("static")
        .and(warp::fs::dir("static"));

    let index = warp::path::end()
        .map(|| warp::reply::html(include_str!("../../static/topology_monitor.html")));

    let routes = api_status
        .or(api_nodes)
        .or(static_files)
        .or(index)
        .with(warp::cors().allow_any_origin());

    println!("🌐 Starting web server on port {}", args.port);
    warp::serve(routes)
        .run(([0, 0, 0, 0], args.port))
        .await;

    Ok(())
}

async fn monitor_topology(
    state: SharedState,
    nodes: HashMap<String, String>,
    interval: u64,
) {
    let mut message_count = 0;
    let start_time = std::time::Instant::now();

    loop {
        let mut status = state.write().await;
        
        // Update uptime
        status.uptime = start_time.elapsed().as_secs();
        
        // Check each node
        for (name, addr) in &nodes {
            let node_status = check_node_status(name, addr).await;
            status.nodes.insert(name.clone(), node_status);
        }
        
        // Update total message count (simulated for now)
        message_count += nodes.len() as u64;
        status.total_messages = message_count;
        
        drop(status);
        
        tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;
    }
}

async fn check_node_status(name: &str, addr: &str) -> NodeStatus {
    // For now, simulate node checking
    // In a real implementation, this would ping the neural network server
    let status = if rand::random::<f64>() > 0.1 {
        "online"
    } else {
        "offline"
    };

    NodeStatus {
        name: name.to_string(),
        address: addr.to_string(),
        status: status.to_string(),
        last_seen: chrono::Utc::now().format("%H:%M:%S").to_string(),
        connections: vec![], // Would be populated by querying the node
        message_count: rand::random::<u64>() % 1000,
        error_count: rand::random::<u64>() % 10,
    }
}

async fn get_status(state: SharedState) -> Result<impl warp::Reply, warp::Rejection> {
    let status = state.read().await;
    Ok(warp::reply::json(&*status))
}

async fn get_nodes(state: SharedState) -> Result<impl warp::Reply, warp::Rejection> {
    let status = state.read().await;
    let nodes: Vec<_> = status.nodes.values().collect();
    Ok(warp::reply::json(&nodes))
}

fn parse_nodes(input: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut nodes = HashMap::new();
    for (i, addr) in input.split(',').enumerate() {
        let addr = addr.trim();
        if !addr.is_empty() {
            nodes.insert(format!("node-{}", i + 1), addr.to_string());
        }
    }
    Ok(nodes)
}

fn get_default_nodes() -> HashMap<String, String> {
    let mut nodes = HashMap::new();
    nodes.insert("node-1".to_string(), "localhost:8081".to_string());
    nodes.insert("node-2".to_string(), "localhost:8082".to_string());
    nodes.insert("node-3".to_string(), "localhost:8083".to_string());
    nodes.insert("node-4".to_string(), "localhost:8084".to_string());
    nodes
}