use crate::cli::*;
use crate::neural_network::{NeuralNetwork, HebbianLearningMode};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use std::io::{self, Write};
use chrono::Utc;
use rand::Rng;

pub fn run_training(
    config_path: PathBuf,
    data_path: PathBuf,
    output_path: Option<PathBuf>,
    epochs: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Starting Neural Network Training");
    println!("==================================");

    // Load configuration
    let config = NetworkConfig::load_from_file(&config_path)?;
    println!("✅ Loaded configuration from: {}", config_path.display());
    
    // Load training data
    let training_data = if data_path.extension().and_then(|s| s.to_str()) == Some("csv") {
        TrainingData::load_from_csv(&data_path)?
    } else {
        TrainingData::load_from_json(&data_path)?
    };
    println!("✅ Loaded {} training samples from: {}", training_data.inputs.len(), data_path.display());

    // Create network
    let mut network = config.create_network()?;
    println!("✅ Created network: {}", network.info());
    println!("   Parameters: {}", network.num_parameters());
    
    if verbose {
        println!("   Architecture: {:?}", config.architecture);
        println!("   Learning rate: {}", config.learning_rate);
        println!("   Hebbian mode: {:?}", config.hebbian_mode);
        println!("   Online learning: {}", config.online_learning);
    }

    // Split data for validation if specified
    let validation_split = config.training.validation_split;
    let split_index = ((1.0 - validation_split) * training_data.inputs.len() as f64) as usize;
    
    let train_inputs = &training_data.inputs[..split_index];
    let train_targets = &training_data.targets[..split_index];
    let val_inputs = &training_data.inputs[split_index..];
    let val_targets = &training_data.targets[split_index..];

    println!("\n📊 Training Configuration:");
    println!("   Training samples: {}", train_inputs.len());
    println!("   Validation samples: {}", val_inputs.len());
    println!("   Epochs: {}", epochs);
    println!("   Batch size: {}", config.training.batch_size);

    // Training loop
    println!("\n🚀 Starting training...");
    let start_time = Instant::now();
    let mut best_val_error = f64::INFINITY;
    let mut patience_counter = 0;

    for epoch in 0..epochs {
        let mut total_train_error = 0.0;
        let mut _batch_count = 0;

        // Training batches
        for batch_start in (0..train_inputs.len()).step_by(config.training.batch_size) {
            let batch_end = (batch_start + config.training.batch_size).min(train_inputs.len());
            let mut batch_error = 0.0;

            for i in batch_start..batch_end {
                let error = network.train(&train_inputs[i], &train_targets[i]);
                batch_error += error;
            }

            total_train_error += batch_error;
            _batch_count += 1;
        }

        let avg_train_error = total_train_error / train_inputs.len() as f64;

        // Validation
        let mut total_val_error = 0.0;
        if !val_inputs.is_empty() {
            for i in 0..val_inputs.len() {
                let (output, _) = network.forward(&val_inputs[i]);
                let error: f64 = output.iter()
                    .zip(val_targets[i].iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum();
                total_val_error += error;
            }
            total_val_error /= val_inputs.len() as f64;
        }

        // Print progress
        if epoch % config.training.print_interval == 0 || verbose {
            if val_inputs.is_empty() {
                println!("   Epoch {}: Train Error = {:.6}", epoch, avg_train_error);
            } else {
                println!("   Epoch {}: Train Error = {:.6}, Val Error = {:.6}", 
                        epoch, avg_train_error, total_val_error);
            }
        }

        // Early stopping
        if !val_inputs.is_empty() {
            if total_val_error < best_val_error - config.training.early_stop_threshold {
                best_val_error = total_val_error;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= config.training.early_stop_patience {
                    println!("🛑 Early stopping at epoch {} (best val error: {:.6})", 
                            epoch, best_val_error);
                    break;
                }
            }
        }
    }

    let training_time = start_time.elapsed();
    println!("\n✅ Training completed in {:.2}s", training_time.as_secs_f64());

    // Save model if output path specified
    if let Some(output_path) = output_path {
        if output_path.extension().and_then(|s| s.to_str()) == Some("bin") {
            network.save_to_binary(&output_path)?;
            println!("💾 Model saved to: {} (binary format)", output_path.display());
        } else {
            network.save_to_file(&output_path)?;
            println!("💾 Model saved to: {} (JSON format)", output_path.display());
        }
    }

    // Final evaluation
    println!("\n📈 Final Evaluation:");
    if !val_inputs.is_empty() {
        let mut correct = 0;
        for i in 0..val_inputs.len() {
            let (output, _) = network.forward(&val_inputs[i]);
            let predicted = if output[0] > 0.5 { 1.0 } else { 0.0 };
            if (predicted - val_targets[i][0]).abs() < 0.1 {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / val_inputs.len() as f64;
        println!("   Validation Accuracy: {:.2}%", accuracy * 100.0);
    }

    Ok(())
}

pub fn run_prediction(
    config_path: Option<PathBuf>,
    input: String,
    model_path: Option<PathBuf>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔮 Neural Network Prediction");
    println!("===========================");

    // Create or load network
    let mut network = if let Some(model_path) = model_path {
        if model_path.extension().and_then(|s| s.to_str()) == Some("bin") {
            NeuralNetwork::load_from_binary(&model_path)?
        } else {
            NeuralNetwork::load_from_file(&model_path)?
        }
    } else if let Some(config_path) = config_path {
        // Load configuration
        let config = NetworkConfig::load_from_file(&config_path)?;
        println!("✅ Loaded configuration from: {}", config_path.display());
        config.create_network()?
    } else {
        return Err("Either config or model path must be provided".into());
    };

    // Parse input
    let input_values = parse_input_string(&input)?;
    println!("📥 Input: {:?}", input_values);

    // Run prediction
    let start_time = Instant::now();
    let (output, _) = network.forward(&input_values);
    let processing_time = start_time.elapsed();

    // Calculate confidence (simple heuristic)
    let confidence = output.iter().map(|&x| (x - 0.5).abs() + 0.5).fold(0.0, f64::max);

    let result = PredictionResult {
        timestamp: Utc::now(),
        input: input_values,
        output: output.clone(),
        confidence,
        processing_time_ms: processing_time.as_secs_f64() * 1000.0,
    };

    // Output result
    match format {
        OutputFormat::Json => {
            println!("📤 Result (JSON):");
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        OutputFormat::Csv => {
            println!("📤 Result (CSV):");
            println!("timestamp,input,output,confidence,processing_time_ms");
            println!("{},{:?},{:?},{:.4},{:.2}", 
                    result.timestamp, result.input, result.output, 
                    result.confidence, result.processing_time_ms);
        }
        OutputFormat::Plain => {
            println!("📤 Result:");
            println!("   Output: {:?}", output);
            println!("   Confidence: {:.2}%", confidence * 100.0);
            println!("   Processing time: {:.2}ms", result.processing_time_ms);
        }
    }

    Ok(())
}

pub fn create_sample_config(
    output_path: PathBuf,
    network_type: NetworkType,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Creating Sample Configuration");
    println!("===============================");

    let config = match network_type {
        NetworkType::Feedforward => NetworkConfig {
            architecture: vec![2, 4, 1],
            learning_rate: 0.1,
            hebbian_mode: HebbianLearningModeConfig::Classic,
            use_backprop: true,
            backprop_rate: 0.1,
            online_learning: false,
            ..Default::default()
        },
        NetworkType::Hebbian => NetworkConfig {
            architecture: vec![3, 6, 2],
            learning_rate: 0.05,
            hebbian_mode: HebbianLearningModeConfig::Oja,
            hebbian_rate: 0.05,
            use_backprop: false,
            online_learning: false,
            ..Default::default()
        },
        NetworkType::Online => NetworkConfig {
            architecture: vec![2, 4, 1],
            learning_rate: 0.1,
            hebbian_mode: HebbianLearningModeConfig::Classic,
            online_learning: true,
            ..Default::default()
        },
        NetworkType::Distributed => NetworkConfig {
            architecture: vec![4, 8, 4, 2],
            learning_rate: 0.05,
            hebbian_mode: HebbianLearningModeConfig::Hybrid,
            use_backprop: true,
            backprop_rate: 0.05,
            online_learning: false,
            ..Default::default()
        },
    };

    config.save_to_file(&output_path)?;
    println!("✅ Sample configuration saved to: {}", output_path.display());
    println!("   Network type: {:?}", network_type);
    println!("   Architecture: {:?}", config.architecture);

    Ok(())
}

pub fn run_interactive_mode(
    config_path: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 Interactive Neural Network Mode");
    println!("==================================");

    let config = if let Some(config_path) = config_path {
        NetworkConfig::load_from_file(&config_path)?
    } else {
        println!("No config provided, using default configuration");
        NetworkConfig::default()
    };

    let mut network = config.create_network()?;
    println!("✅ Network created: {}", network.info());
    println!("\nCommands:");
    println!("  predict <input>  - Run prediction (e.g., 'predict 0.5,0.8')");
    println!("  train <input> <target> - Train on single sample");
    println!("  info             - Show network information");
    println!("  save <file>      - Save network to file");
    println!("  load <file>      - Load network from file");
    println!("  quit             - Exit interactive mode");

    loop {
        print!("\nbenny> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts[0] {
            "predict" => {
                if parts.len() < 2 {
                    println!("❌ Usage: predict <input>");
                    continue;
                }
                match parse_input_string(parts[1]) {
                    Ok(input_values) => {
                        let (output, _) = network.forward(&input_values);
                        println!("📤 Input: {:?} -> Output: {:?}", input_values, output);
                    }
                    Err(e) => println!("❌ Error parsing input: {}", e),
                }
            }
            "train" => {
                if parts.len() < 3 {
                    println!("❌ Usage: train <input> <target>");
                    continue;
                }
                match (parse_input_string(parts[1]), parse_input_string(parts[2])) {
                    (Ok(input_values), Ok(target_values)) => {
                        let error = network.train(&input_values, &target_values);
                        println!("📈 Training error: {:.6}", error);
                    }
                    (Err(e), _) | (_, Err(e)) => println!("❌ Error parsing values: {}", e),
                }
            }
            "info" => {
                println!("🧠 Network Information:");
                println!("   {}", network.info());
                println!("   Parameters: {}", network.num_parameters());
                println!("   Hebbian rate: {}", network.get_hebbian_rate());
                println!("   Online learning: {}", network.is_online_learning());
            }
            "save" => {
                if parts.len() < 2 {
                    println!("❌ Usage: save <filename>");
                    continue;
                }
                match network.save_to_file(parts[1]) {
                    Ok(()) => println!("✅ Network saved to: {}", parts[1]),
                    Err(e) => println!("❌ Error saving: {}", e),
                }
            }
            "load" => {
                if parts.len() < 2 {
                    println!("❌ Usage: load <filename>");
                    continue;
                }
                match NeuralNetwork::load_from_file(parts[1]) {
                    Ok(loaded_network) => {
                        network = loaded_network;
                        println!("✅ Network loaded from: {}", parts[1]);
                    }
                    Err(e) => println!("❌ Error loading: {}", e),
                }
            }
            "quit" | "exit" => {
                println!("👋 Goodbye!");
                break;
            }
            _ => {
                println!("❌ Unknown command: {}", parts[0]);
            }
        }
    }

    Ok(())
}

pub fn run_benchmark(
    config_path: PathBuf,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Neural Network Benchmark");
    println!("==========================");

    let config = NetworkConfig::load_from_file(&config_path)?;
    let mut network = config.create_network()?;
    
    println!("✅ Network: {}", network.info());
    println!("   Parameters: {}", network.num_parameters());
    println!("   Iterations: {}", iterations);

    // Generate random test data
    let mut rng = rand::thread_rng();
    let test_inputs: Vec<Vec<f64>> = (0..iterations)
        .map(|_| (0..config.architecture[0])
            .map(|_| rng.gen_range(0.0..1.0))
            .collect())
        .collect();

    // Benchmark forward pass
    println!("\n🚀 Benchmarking forward pass...");
    let start_time = Instant::now();
    
    for input in &test_inputs {
        let _ = network.forward(input);
    }
    
    let forward_time = start_time.elapsed();
    let forward_per_sec = iterations as f64 / forward_time.as_secs_f64();
    
    println!("   Forward pass: {:.2}ms total, {:.0} ops/sec", 
            forward_time.as_secs_f64() * 1000.0, forward_per_sec);

    // Benchmark training
    println!("\n📈 Benchmarking training...");
    let targets: Vec<Vec<f64>> = (0..iterations)
        .map(|_| (0..*config.architecture.last().unwrap())
            .map(|_| rng.gen_range(0.0..1.0))
            .collect())
        .collect();

    let start_time = Instant::now();
    
    for (input, target) in test_inputs.iter().zip(targets.iter()) {
        let _ = network.train(input, target);
    }
    
    let train_time = start_time.elapsed();
    let train_per_sec = iterations as f64 / train_time.as_secs_f64();
    
    println!("   Training: {:.2}ms total, {:.0} ops/sec", 
            train_time.as_secs_f64() * 1000.0, train_per_sec);

    // Memory usage estimation
    let param_count = network.num_parameters();
    let memory_mb = (param_count * 8) as f64 / 1024.0 / 1024.0; // 8 bytes per f64
    
    println!("\n💾 Memory Usage:");
    println!("   Parameters: {}", param_count);
    println!("   Estimated memory: {:.2} MB", memory_mb);

    Ok(())
}

pub fn run_demo(demo_type: DemoType) -> Result<(), Box<dyn std::error::Error>> {
    match demo_type {
        DemoType::All => {
            run_demo(DemoType::Xor)?;
            println!("\n{}", "=".repeat(50));
            run_demo(DemoType::Hebbian)?;
            println!("\n{}", "=".repeat(50));
            run_demo(DemoType::Serialization)?;
        }
        DemoType::Xor => {
            println!("🔀 XOR Problem Demo");
            println!("==================");
            
            let mut nn = NeuralNetwork::new(2, 4, 1, 0.5);
            let training_data = vec![
                (vec![0.0, 0.0], vec![0.0]),
                (vec![0.0, 1.0], vec![1.0]),
                (vec![1.0, 0.0], vec![1.0]),
                (vec![1.0, 1.0], vec![0.0]),
            ];
            
            println!("Training XOR network...");
            for epoch in 0..1000 {
                let mut total_error = 0.0;
                for (inputs, targets) in &training_data {
                    total_error += nn.train(inputs, targets);
                }
                if epoch % 200 == 0 {
                    println!("  Epoch {}: Error = {:.6}", epoch, total_error);
                }
            }
            
            println!("\nResults:");
            for (inputs, expected) in &training_data {
                let (output, _) = nn.forward(inputs);
                let predicted = if output[0] > 0.5 { 1.0 } else { 0.0 };
                println!("  [{:.0}, {:.0}] -> Expected: {:.0}, Got: {:.3} ({:.0})", 
                        inputs[0], inputs[1], expected[0], output[0], predicted);
            }
        }
        DemoType::Hebbian => {
            println!("🧠 Hebbian Learning Demo");
            println!("=======================");
            
            let mut nn = NeuralNetwork::with_layers_and_mode(
                &[2, 3, 1], 
                0.05, 
                HebbianLearningMode::Oja
            );
            
            println!("Training with Hebbian learning...");
            for i in 0..50 {
                nn.train_unsupervised(&[1.0, 1.0]);
                if i % 10 == 0 {
                    let correlation = nn.get_neuron_correlation(0, 0, 0, 1);
                    println!("  Step {}: Correlation = {:.4}", i, correlation);
                }
            }
        }
        DemoType::Serialization => {
            println!("💾 Serialization Demo");
            println!("====================");
            
            let mut nn = NeuralNetwork::new(2, 3, 1, 0.1);
            
            // Train briefly
            for _ in 0..100 {
                nn.train(&[0.5, 0.8], &[0.7]);
            }
            
            let (original_output, _) = nn.forward(&[0.5, 0.8]);
            println!("Original output: {:?}", original_output);
            
            // Save and load
            nn.save_to_file("demo_network.json")?;
            let mut loaded_nn = NeuralNetwork::load_from_file("demo_network.json")?;
            let (loaded_output, _) = loaded_nn.forward(&[0.5, 0.8]);
            
            println!("Loaded output: {:?}", loaded_output);
            println!("Difference: {:.10}", (original_output[0] - loaded_output[0]).abs());
            
            // Cleanup
            let _ = fs::remove_file("demo_network.json");
        }
        _ => {
            println!("Demo type not implemented yet");
        }
    }
    
    Ok(())
}