use clap::{Arg, Command};
use neural_network::{InputServer, InputServerConfig, NeuralNetworkTarget};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let matches = Command::new("InputServer")
        .version("1.0")
        .about("Web-based input interface for distributed neural networks")
        .arg(
            Arg::new("network-host")
                .long("network-host")
                .value_name("HOST")
                .help("Neural network server hostname")
                .default_value("127.0.0.1"),
        )
        .arg(
            Arg::new("network-port")
                .long("network-port")
                .value_name("PORT")
                .help("Neural network server port")
                .default_value("8001"),
        )
        .arg(
            Arg::new("web-port")
                .long("web-port")
                .value_name("PORT")
                .help("Web server port")
                .default_value("3000"),
        )
        .arg(
            Arg::new("websocket-port")
                .long("websocket-port")
                .value_name("PORT")
                .help("WebSocket server port")
                .default_value("3001"),
        )
        .arg(
            Arg::new("input-size")
                .long("input-size")
                .value_name("SIZE")
                .help("Number of neural network inputs")
                .default_value("4"),
        )
        .arg(
            Arg::new("use-tls")
                .long("use-tls")
                .help("Use TLS for neural network connection")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("cert-path")
                .long("cert-path")
                .value_name("PATH")
                .help("Path to TLS certificate file"),
        )
        .arg(
            Arg::new("key-path")
                .long("key-path")
                .value_name("PATH")
                .help("Path to TLS private key file"),
        )
        .get_matches();

    let network_host = matches.get_one::<String>("network-host").unwrap().clone();
    let network_port: u16 = matches.get_one::<String>("network-port").unwrap().parse()?;
    let web_port: u16 = matches.get_one::<String>("web-port").unwrap().parse()?;
    let websocket_port: u16 = matches
        .get_one::<String>("websocket-port")
        .unwrap()
        .parse()?;
    let input_size: usize = matches.get_one::<String>("input-size").unwrap().parse()?;
    let use_tls = matches.get_flag("use-tls");
    let cert_path = matches.get_one::<String>("cert-path").cloned();
    let key_path = matches.get_one::<String>("key-path").cloned();

    println!("🚀 Starting InputServer");
    println!("   Neural Network: {}:{}", network_host, network_port);
    println!("   Web Interface: http://127.0.0.1:{}", web_port);
    println!("   WebSocket: ws://127.0.0.1:{}", websocket_port);
    println!("   Input Size: {}", input_size);
    println!("   TLS: {}", if use_tls { "Enabled" } else { "Disabled" });

    // Create neural network target configuration
    let neural_network = NeuralNetworkTarget {
        id: "main-network".to_string(),
        name: "Main Neural Network".to_string(),
        address: network_host,
        port: network_port,
        input_count: input_size,
        use_tls,
    };

    // Create InputServer configuration
    let config = InputServerConfig {
        web_address: "127.0.0.1".to_string(),
        web_port,
        websocket_port,
        neural_networks: vec![neural_network],
        cert_path,
        key_path,
    };

    // Create and start the InputServer
    let server = InputServer::new(config);
    server.start().await?;

    Ok(())
}
