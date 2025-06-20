use clap::Parser;
use neural_network::cli::{Cli, Commands};
use neural_network::runner::*;

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Train {
            config,
            data,
            output,
            epochs,
            verbose,
        } => run_training(config, data, output, epochs, verbose),
        Commands::Predict {
            config,
            input,
            model,
            format,
        } => run_prediction(config, input, model, format),
        Commands::InitConfig {
            output,
            network_type,
        } => create_sample_config(output, network_type),
        Commands::Interactive { config } => run_interactive_mode(config),
        Commands::Benchmark { config, iterations } => run_benchmark(config, iterations),
        Commands::Server {
            config,
            model,
            port,
            cert,
            key,
            outputs,
            daemon,
            hebbian_learning,
        } => run_server(
            config,
            model,
            port,
            cert,
            key,
            outputs,
            daemon,
            hebbian_learning,
        ),
        Commands::Demo { demo_type } => run_demo(demo_type),
    };

    if let Err(e) = result {
        eprintln!("❌ Error: {}", e);
        std::process::exit(1);
    }
}
