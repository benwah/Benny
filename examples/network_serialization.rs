use neural_network::{NeuralNetwork, HebbianLearningMode};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Network Serialization Demo");
    println!("=====================================\n");

    // Create and train a neural network
    println!("1. Creating and training a neural network...");
    let mut nn = NeuralNetwork::with_layers_and_mode(
        &[2, 4, 3, 1], 
        0.05, 
        HebbianLearningMode::Oja
    );

    // Training data for XOR-like problem
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.1]),
        (vec![0.0, 1.0], vec![0.9]),
        (vec![1.0, 0.0], vec![0.9]),
        (vec![1.0, 1.0], vec![0.1]),
    ];

    println!("   Training on XOR-like data...");
    for epoch in 0..100 {
        let mut total_error = 0.0;
        for (inputs, targets) in &training_data {
            total_error += nn.train(inputs, targets);
        }
        if epoch % 20 == 0 {
            println!("   Epoch {}: Average Error = {:.6}", epoch, total_error / training_data.len() as f64);
        }
    }

    // Test the trained network
    println!("\n2. Testing trained network:");
    for (inputs, expected) in &training_data {
        let (output, _) = nn.forward(inputs);
        println!("   Input: {:?} -> Output: {:.3} (Expected: {:.1})", 
                 inputs, output[0], expected[0]);
    }

    // Display network metadata
    println!("\n3. Network Metadata:");
    println!("{}", nn.export_metadata());

    // Save to JSON file
    println!("\n4. Saving network to JSON file...");
    let json_filename = "trained_network.json";
    nn.save_to_file(json_filename)?;
    
    let json_size = fs::metadata(json_filename)?.len();
    println!("   ✅ Saved to '{}' ({} bytes)", json_filename, json_size);

    // Save to binary file
    println!("\n5. Saving network to binary file...");
    let binary_filename = "trained_network.bin";
    nn.save_to_binary(binary_filename)?;
    
    let binary_size = fs::metadata(binary_filename)?.len();
    println!("   ✅ Saved to '{}' ({} bytes)", binary_filename, binary_size);
    println!("   📊 Binary file is {:.1}% the size of JSON", 
             (binary_size as f64 / json_size as f64) * 100.0);

    // Load from JSON and test
    println!("\n6. Loading network from JSON file...");
    let mut loaded_nn_json = NeuralNetwork::load_from_file(json_filename)?;
    println!("   ✅ Successfully loaded from JSON");

    // Verify JSON-loaded network produces identical results
    println!("   Testing loaded network (JSON):");
    for (inputs, _expected) in &training_data {
        let (original_output, _) = nn.forward(inputs);
        let (loaded_output, _) = loaded_nn_json.forward(inputs);
        let difference = (original_output[0] - loaded_output[0]).abs();
        
        println!("   Input: {:?} -> Original: {:.6}, Loaded: {:.6}, Diff: {:.2e}", 
                 inputs, original_output[0], loaded_output[0], difference);
        
        if difference > 1e-10 {
            println!("   ⚠️  Warning: Outputs differ by more than expected!");
        }
    }

    // Load from binary and test
    println!("\n7. Loading network from binary file...");
    let mut loaded_nn_binary = NeuralNetwork::load_from_binary(binary_filename)?;
    println!("   ✅ Successfully loaded from binary");

    // Verify binary-loaded network produces identical results
    println!("   Testing loaded network (Binary):");
    for (inputs, _expected) in &training_data {
        let (original_output, _) = nn.forward(inputs);
        let (loaded_output, _) = loaded_nn_binary.forward(inputs);
        let difference = (original_output[0] - loaded_output[0]).abs();
        
        println!("   Input: {:?} -> Original: {:.6}, Loaded: {:.6}, Diff: {:.2e}", 
                 inputs, original_output[0], loaded_output[0], difference);
        
        if difference > 1e-15 {
            println!("   ⚠️  Warning: Outputs differ by more than expected!");
        }
    }

    // Demonstrate online learning preservation
    println!("\n8. Testing online learning network serialization...");
    let mut online_nn = NeuralNetwork::with_online_learning(
        &[2, 3, 1], 
        0.02, 
        HebbianLearningMode::Classic
    );

    // Do some forward passes to modify weights through online learning
    let test_input = vec![0.5, 0.7];
    let (original_online_output, _) = online_nn.forward(&test_input);
    println!("   Original online network output: {:.6}", original_online_output[0]);
    
    // Save and reload online network
    let online_filename = "online_network.json";
    online_nn.save_to_file(online_filename)?;
    let mut loaded_online_nn = NeuralNetwork::load_from_file(online_filename)?;
    
    let (loaded_online_output, _) = loaded_online_nn.forward(&test_input);
    println!("   Loaded online network output:   {:.6}", loaded_online_output[0]);
    println!("   Online learning preserved: {}", loaded_online_nn.is_online_learning());

    // Demonstrate different Hebbian learning modes
    println!("\n9. Testing different Hebbian learning modes...");
    let modes = vec![
        HebbianLearningMode::Classic,
        HebbianLearningMode::Competitive,
        HebbianLearningMode::Oja,
        HebbianLearningMode::BCM,
        HebbianLearningMode::AntiHebbian,
        HebbianLearningMode::Hybrid,
    ];

    for (i, mode) in modes.iter().enumerate() {
        let test_nn = NeuralNetwork::with_layers_and_mode(&[2, 3, 1], 0.1, mode.clone());
        let filename = format!("network_mode_{}.json", i);
        test_nn.save_to_file(&filename)?;
        
        let _loaded_test_nn = NeuralNetwork::load_from_file(&filename)?;
        println!("   Mode {:?}: ✅ Serialization successful", mode);
        
        // Clean up
        let _ = fs::remove_file(&filename);
    }

    // Clean up main files
    println!("\n10. Cleaning up files...");
    let files_to_remove = vec![json_filename, binary_filename, online_filename];
    for file in files_to_remove {
        match fs::remove_file(file) {
            Ok(_) => println!("   ✅ Removed {}", file),
            Err(_) => println!("   ⚠️  Could not remove {}", file),
        }
    }

    println!("\n🎉 Serialization demo completed successfully!");
    println!("\nKey Features Demonstrated:");
    println!("• JSON serialization (human-readable)");
    println!("• Binary serialization (compact)");
    println!("• Complete state preservation");
    println!("• Online learning network support");
    println!("• All Hebbian learning modes supported");
    println!("• Metadata export functionality");

    Ok(())
}