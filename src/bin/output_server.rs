use clap::{Arg, Command};
use neural_network::{OutputServer, OutputServerConfig, NeuralNetworkSource};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let matches = Command::new("OutputServer")
        .version("1.0")
        .about("Web-based output monitor for distributed neural networks")
        .arg(
            Arg::new("listen-host")
                .long("listen-host")
                .value_name("HOST")
                .help("Host address to listen for neural network connections")
                .default_value("0.0.0.0"),
        )
        .arg(
            Arg::new("listen-port")
                .long("listen-port")
                .value_name("PORT")
                .help("Port to listen for neural network connections")
                .default_value("8002"),
        )
        .arg(
            Arg::new("web-host")
                .long("web-host")
                .value_name("HOST")
                .help("Web server hostname")
                .default_value("0.0.0.0"),
        )
        .arg(
            Arg::new("web-port")
                .long("web-port")
                .value_name("PORT")
                .help("Web server port")
                .default_value("12000"),
        )
        .arg(
            Arg::new("websocket-port")
                .long("websocket-port")
                .value_name("PORT")
                .help("WebSocket server port")
                .default_value("12001"),
        )
        .arg(
            Arg::new("output-size")
                .long("output-size")
                .value_name("SIZE")
                .help("Number of neural network outputs to expect")
                .default_value("2"),
        )
        .arg(
            Arg::new("network-name")
                .long("network-name")
                .value_name("NAME")
                .help("Display name for the neural network")
                .default_value("Main Neural Network"),
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

    let listen_host = matches.get_one::<String>("listen-host").unwrap().clone();
    let listen_port: u16 = matches.get_one::<String>("listen-port").unwrap().parse()?;
    let web_host = matches.get_one::<String>("web-host").unwrap().clone();
    let web_port: u16 = matches.get_one::<String>("web-port").unwrap().parse()?;
    let websocket_port: u16 = matches
        .get_one::<String>("websocket-port")
        .unwrap()
        .parse()?;
    let output_size: usize = matches.get_one::<String>("output-size").unwrap().parse()?;
    let network_name = matches.get_one::<String>("network-name").unwrap().clone();
    let use_tls = matches.get_flag("use-tls");
    let cert_path = matches.get_one::<String>("cert-path").cloned();
    let key_path = matches.get_one::<String>("key-path").cloned();

    println!("🚀 Starting OutputServer");
    println!("   Listening for Neural Networks: {}:{}", listen_host, listen_port);
    println!("   Web Interface: http://{}:{}", web_host, web_port);
    println!("   WebSocket: ws://{}:{}", web_host, websocket_port);
    println!("   Expected Output Size: {}", output_size);
    println!("   Network Name: {}", network_name);
    println!("   TLS: {}", if use_tls { "Enabled" } else { "Disabled" });

    // Create neural network source configuration
    let neural_network = NeuralNetworkSource {
        id: "main-network".to_string(),
        name: network_name,
        listen_address: listen_host,
        listen_port,
        output_count: output_size,
        use_tls,
    };

    // Create OutputServer configuration
    let config = OutputServerConfig {
        web_address: web_host,
        web_port,
        websocket_port,
        expected_output_size: output_size,
        neural_networks: vec![neural_network],
        cert_path,
        key_path,
    };

    // Create and start the OutputServer
    let server = OutputServer::new(config);
    server.start().await?;

    Ok(())
}