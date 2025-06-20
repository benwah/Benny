use crate::neural_network::{NeuralNetwork, HebbianLearningMode};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use chrono::{DateTime, Utc};

#[derive(Parser)]
#[command(name = "benny")]
#[command(about = "A configurable neural network runner")]
#[command(version = "1.0")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
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
        config: Option<PathBuf>,
        /// Input data file path or single input values (comma-separated)
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
    /// Show demo of different network capabilities
    Demo {
        /// Demo type to run
        #[arg(short, long, default_value = "all")]
        demo_type: DemoType,
    },
}

#[derive(ValueEnum, Clone)]
pub enum OutputFormat {
    Json,
    Csv,
    Plain,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum NetworkType {
    Feedforward,
    Hebbian,
    Online,
    Distributed,
}

#[derive(ValueEnum, Clone)]
pub enum DemoType {
    All,
    Xor,
    Hebbian,
    Serialization,
    MultiCore,
    Composition,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NetworkConfig {
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum HebbianLearningModeConfig {
    Classic,
    Competitive,
    Oja,
    BCM,
    AntiHebbian,
    Hybrid,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingConfig {
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
pub struct TrainingData {
    pub inputs: Vec<Vec<f64>>,
    pub targets: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
pub struct PredictionResult {
    pub timestamp: DateTime<Utc>,
    pub input: Vec<f64>,
    pub output: Vec<f64>,
    pub confidence: f64,
    pub processing_time_ms: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            architecture: vec![2, 4, 1],
            learning_rate: 0.1,
            hebbian_mode: HebbianLearningModeConfig::Classic,
            hebbian_rate: 0.05,
            anti_hebbian_rate: 0.0,
            decay_rate: 0.005,
            homeostatic_rate: 0.005,
            target_activity: 0.2,
            history_size: 20,
            use_backprop: false,
            backprop_rate: 0.0,
            online_learning: false,
            training: TrainingConfig::default(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            print_interval: 100,
            early_stop_threshold: 0.001,
            early_stop_patience: 50,
            validation_split: 0.2,
        }
    }
}

impl From<HebbianLearningModeConfig> for HebbianLearningMode {
    fn from(config: HebbianLearningModeConfig) -> Self {
        match config {
            HebbianLearningModeConfig::Classic => HebbianLearningMode::Classic,
            HebbianLearningModeConfig::Competitive => HebbianLearningMode::Competitive,
            HebbianLearningModeConfig::Oja => HebbianLearningMode::Oja,
            HebbianLearningModeConfig::BCM => HebbianLearningMode::BCM,
            HebbianLearningModeConfig::AntiHebbian => HebbianLearningMode::AntiHebbian,
            HebbianLearningModeConfig::Hybrid => HebbianLearningMode::Hybrid,
        }
    }
}

impl NetworkConfig {
    pub fn create_network(&self) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut nn = if self.online_learning {
            NeuralNetwork::with_online_learning(
                &self.architecture,
                self.learning_rate,
                self.hebbian_mode.clone().into(),
            )
        } else {
            NeuralNetwork::with_layers_and_mode(
                &self.architecture,
                self.learning_rate,
                self.hebbian_mode.clone().into(),
            )
        };

        // Configure additional parameters
        nn.set_hebbian_rate(self.hebbian_rate);
        nn.set_decay_rate(self.decay_rate);
        
        if self.use_backprop {
            nn.set_backprop_enabled(true, self.backprop_rate);
        }

        Ok(nn)
    }

    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: NetworkConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}

impl TrainingData {
    pub fn load_from_csv<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for result in reader.records() {
            let record = result?;
            let values: Vec<f64> = record.iter()
                .map(|s| s.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()?;
            
            if values.len() < 2 {
                return Err("CSV must have at least 2 columns (input and target)".into());
            }
            
            // Assume last column is target, rest are inputs
            let input = values[..values.len()-1].to_vec();
            let target = vec![values[values.len()-1]];
            
            inputs.push(input);
            targets.push(target);
        }

        Ok(TrainingData { inputs, targets })
    }

    pub fn load_from_json<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let data: TrainingData = serde_json::from_str(&content)?;
        Ok(data)
    }

    pub fn save_to_json<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}

pub fn parse_input_string(input: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if input.ends_with(".csv") || input.ends_with(".json") {
        // It's a file path
        if input.ends_with(".csv") {
            let data = TrainingData::load_from_csv(input)?;
            if data.inputs.is_empty() {
                return Err("No data found in file".into());
            }
            Ok(data.inputs[0].clone()) // Return first input for single prediction
        } else {
            let data = TrainingData::load_from_json(input)?;
            if data.inputs.is_empty() {
                return Err("No data found in file".into());
            }
            Ok(data.inputs[0].clone()) // Return first input for single prediction
        }
    } else {
        // Parse as comma-separated values
        let values: Result<Vec<f64>, _> = input
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect();
        Ok(values?)
    }
}