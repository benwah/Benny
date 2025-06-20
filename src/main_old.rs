use neural_network::{NeuralNetwork, HebbianLearningMode};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};

#[derive(Parser)]
#[command(name = "benny")]
#[command(about = "A configurable neural network runner")]
#[command(version = "1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a neural network
    Train {
        /// Configuration file path
        #[arg(short, long)]
        config: PathBuf,
        /// Training data file path
        #[arg(short, long)]
        data: PathBuf,
        /// Output model file path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Number of epochs
        #[arg(short, long, default_value = "1000")]
        epochs: usize,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run inference on a trained network
    Predict {
        /// Configuration file path
        #[arg(short, long)]
        config: PathBuf,
        /// Input data file path or single input values
        #[arg(short, long)]
        input: String,
        /// Model file path (if available)
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// Output format
        #[arg(short, long, default_value = "json")]
        format: OutputFormat,
    },
    /// Create a sample configuration file
    InitConfig {
        /// Output configuration file path
        #[arg(short, long, default_value = "network_config.toml")]
        output: PathBuf,
        /// Network type
        #[arg(short, long, default_value = "feedforward")]
        network_type: NetworkType,
    },
    /// Run interactive mode
    Interactive {
        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    /// Benchmark network performance
    Benchmark {
        /// Configuration file path
        #[arg(short, long)]
        config: PathBuf,
        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: usize,
    },
}

#[derive(ValueEnum, Clone)]
enum OutputFormat {
    Json,
    Csv,
    Plain,
}

#[derive(ValueEnum, Clone)]
enum NetworkType {
    Feedforward,
    Hebbian,
    Online,
    Distributed,
}

#[derive(Serialize, Deserialize, Clone)]
struct NetworkConfig {
    /// Network architecture (layer sizes)
    pub architecture: Vec<usize>,
    /// Learning rate
    pub learning_rate: f64,
    /// Hebbian learning mode
    pub hebbian_mode: HebbianLearningModeConfig,
    /// Hebbian learning rate
    pub hebbian_rate: f64,
    /// Anti-Hebbian learning rate
    pub anti_hebbian_rate: f64,
    /// Weight decay rate
    pub decay_rate: f64,
    /// Homeostatic learning rate
    pub homeostatic_rate: f64,
    /// Target activity level
    pub target_activity: f64,
    /// Activation history size
    pub history_size: usize,
    /// Enable backpropagation
    pub use_backprop: bool,
    /// Backpropagation learning rate
    pub backprop_rate: f64,
    /// Enable online learning
    pub online_learning: bool,
    /// Training configuration
    pub training: TrainingConfig,
}

#[derive(Serialize, Deserialize, Clone)]
enum HebbianLearningModeConfig {
    Classic,
    Competitive,
    Oja,
    BCM,
    AntiHebbian,
    Hybrid,
}

#[derive(Serialize, Deserialize, Clone)]
struct TrainingConfig {
    /// Batch size for training
    pub batch_size: usize,
    /// Print progress every N epochs
    pub print_interval: usize,
    /// Early stopping threshold
    pub early_stop_threshold: f64,
    /// Early stopping patience (epochs)
    pub early_stop_patience: usize,
    /// Validation split ratio
    pub validation_split: f64,
}

#[derive(Serialize, Deserialize)]
struct TrainingData {
    pub inputs: Vec<Vec<f64>>,
    pub targets: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
struct PredictionResult {
    pub timestamp: DateTime<Utc>,
    pub input: Vec<f64>,
    pub output: Vec<f64>,
    pub confidence: f64,
    pub processing_time_ms: f64,
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Train { config, data, output, epochs, verbose } => {
            if let Err(e) = run_training(config, data, output, epochs, verbose) {
                eprintln!("❌ Training failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Predict { config, input, model, format } => {
            if let Err(e) = run_prediction(config, input, model, format) {
                eprintln!("❌ Prediction failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::InitConfig { output, network_type } => {
            if let Err(e) = create_sample_config(output, network_type) {
                eprintln!("❌ Config creation failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Interactive { config } => {
            if let Err(e) = run_interactive_mode(config) {
                eprintln!("❌ Interactive mode failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Benchmark { config, iterations } => {
            if let Err(e) = run_benchmark(config, iterations) {
                eprintln!("❌ Benchmark failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn demonstrate_flexible_architecture() {
    println!("\n🏗️  Flexible Architecture Demo");
    println!("-----------------------------");
    
    // Show different network configurations
    let configs = vec![
        ("Simple", vec![2, 3, 1]),
        ("Deep", vec![2, 6, 4, 2, 1]),
        ("Wide", vec![2, 10, 1]),
        ("Multi-output", vec![3, 5, 3]),
    ];
    
    for (name, layers) in configs {
        let nn = NeuralNetwork::with_layers(&layers, 0.1);
        println!("  {}: {} ({} parameters)", name, nn.info(), nn.num_parameters());
    }
    println!();
}

fn solve_xor_problem() {
    println!("\n🔀 XOR Problem");
    println!("--------------");
    
    let mut nn = NeuralNetwork::new(2, 4, 1, 0.5);
    println!("{}", nn.info());
    
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    println!("\nTraining...");
    train_network(&mut nn, &training_data, 10000);
    
    println!("\nResults:");
    test_network(&mut nn, &training_data);
}

fn solve_and_problem() {
    println!("\n🔗 AND Problem");
    println!("---------------");
    
    let mut nn = NeuralNetwork::new(2, 3, 1, 0.3);
    println!("{}", nn.info());
    
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];
    
    println!("\nTraining...");
    train_network(&mut nn, &training_data, 5000);
    
    println!("\nResults:");
    test_network(&mut nn, &training_data);
}

fn solve_or_problem() {
    println!("\n🔀 OR Problem");
    println!("-------------");
    
    let mut nn = NeuralNetwork::new(2, 3, 1, 0.3);
    println!("{}", nn.info());
    
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];
    
    println!("\nTraining...");
    train_network(&mut nn, &training_data, 5000);
    
    println!("\nResults:");
    test_network(&mut nn, &training_data);
}

fn demonstrate_hebbian_learning() {
    println!("\n🧠 Hebbian Learning Preview");
    println!("---------------------------");
    println!("\"Neurons that fire together, wire together\"");
    println!();
    
    // Show different Hebbian learning modes
    println!("🎯 Available Hebbian Learning Modes:");
    println!("  • Classic: Basic Hebbian rule (default)");
    println!("  • Competitive: Winner-take-all learning");
    println!("  • Oja: Normalized Hebbian learning");
    println!("  • BCM: Bienenstock-Cooper-Munro rule");
    println!("  • AntiHebbian: Negative correlation learning");
    println!("  • Hybrid: Combines Hebbian + Backpropagation");
    println!();
    
    // Demonstrate Classic Hebbian learning (default)
    let mut hebbian_nn = NeuralNetwork::with_layers(&[2, 3, 1], 0.05);
    println!("🔬 Classic Hebbian Network: {}", hebbian_nn.info());
    println!("   Hebbian rate: {}", hebbian_nn.get_hebbian_rate());
    println!("   Learning mode: Classic (default)");
    
    // Show weight before Hebbian learning
    let initial_weight = hebbian_nn.get_weight(0, 0, 0);
    println!("   Initial weight (input[0] -> hidden[0]): {:.4}", initial_weight);
    
    // Apply Hebbian learning with correlated inputs
    println!("\n⚡ Training with correlated inputs [1.0, 1.0]...");
    for _ in 0..20 {
        hebbian_nn.train_unsupervised(&[1.0, 1.0]); // Both inputs high - pure Hebbian learning
    }
    
    let final_weight = hebbian_nn.get_weight(0, 0, 0);
    let correlation = hebbian_nn.get_neuron_correlation(0, 0, 0, 1);
    
    println!("📊 After Hebbian training:");
    println!("   Final weight: {:.4}", final_weight);
    println!("   Input correlation: {:.4}", correlation);
    println!("   Weight change: {:.4}", final_weight - initial_weight);
    
    // Demonstrate different learning mode
    println!("\n🔄 Anti-Hebbian Learning Mode:");
    let mut anti_nn = NeuralNetwork::with_layers_and_mode(
        &[2, 3, 1], 
        0.05, 
        HebbianLearningMode::AntiHebbian
    );
    println!("   Network: {}", anti_nn.info());
    
    // Train with anti-correlated patterns
    for _ in 0..10 {
        anti_nn.train_unsupervised(&[1.0, 0.0]); // High-Low
        anti_nn.train_unsupervised(&[0.0, 1.0]); // Low-High
    }
    
    let anti_correlation = anti_nn.get_neuron_correlation(0, 0, 0, 1);
    println!("   Anti-correlation learned: {:.4}", anti_correlation);
    
    println!("\n💡 Run 'cargo run --example hebbian_learning' for full demonstration!");
    
    println!("\n{}", "=".repeat(50));
    println!("\n🔗 Network Composition Preview");
    println!("------------------------------");
    println!("Connect multiple neural networks together");
    println!("Creating a simple pipeline: Feature Extractor -> Classifier");
    println!();
    println!("Example code:");
    println!("  let mut composer = NetworkComposer::new();");
    println!("  let feature_net = NeuralNetwork::new(2, 4, 2, 0.1);");
    println!("  let classifier_net = NeuralNetwork::new(2, 3, 1, 0.1);");
    println!();
    println!("  composer.add_network(\"features\".to_string(), feature_net).unwrap();");
    println!("  composer.add_network(\"classifier\".to_string(), classifier_net).unwrap();");
    println!("  composer.connect_networks(\"features\", \"classifier\", vec![0, 1], vec![0, 1]).unwrap();");
    println!();
    println!("  let mut inputs = HashMap::new();");
    println!("  inputs.insert(\"features\".to_string(), vec![0.8, 0.3]);");
    println!("  let outputs = composer.forward(&inputs).unwrap();");
    println!();
    println!("💡 Run 'cargo run --example network_composition' for full demonstration!

==================================================

🚀 Multi-Core Performance Preview
---------------------------------
All neural network operations now utilize multiple CPU cores!

🖥️  System cores: 4
⚡ Parallel forward propagation: Neuron computations across cores
🔄 Parallel backpropagation: Error calculation and weight updates
📦 Batch processing: Multiple samples processed simultaneously

Example batch processing:
  let training_batch = vec![
      (vec![0.0, 0.0], vec![0.0]),
      (vec![0.0, 1.0], vec![1.0]),
      (vec![1.0, 0.0], vec![1.0]),
      (vec![1.0, 1.0], vec![0.0]),
  ];
  let batch_error = nn.train_batch(&training_batch);
  let batch_outputs = nn.forward_batch(&inputs);

💡 Run 'cargo run --example multi_core_performance' for benchmarks!");
}

fn train_network(nn: &mut NeuralNetwork, training_data: &[(Vec<f64>, Vec<f64>)], epochs: usize) {
    let print_interval = epochs / 5;
    
    for epoch in 0..epochs {
        let mut total_error = 0.0;
        
        for (inputs, targets) in training_data {
            let error = nn.train(inputs, targets);
            total_error += error;
        }
        
        if epoch % print_interval == 0 {
            println!("  Epoch {}: Error = {:.6}", epoch, total_error);
        }
    }
}

fn test_network(nn: &mut NeuralNetwork, test_data: &[(Vec<f64>, Vec<f64>)]) {
    for (inputs, expected) in test_data {
        let prediction = nn.predict(inputs);
        let predicted_binary = if prediction[0] > 0.5 { 1.0 } else { 0.0 };
        let accuracy = if (predicted_binary - expected[0]).abs() < 0.1 { "✅" } else { "❌" };
        
        println!(
            "  Input: [{:.0}, {:.0}] -> Expected: {:.0}, Predicted: {:.4} ({:.0}) {}",
            inputs[0], inputs[1], expected[0], prediction[0], predicted_binary, accuracy
        );
    }
}
