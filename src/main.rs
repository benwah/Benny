use neural_network::NeuralNetwork;

fn main() {
    println!("🧠 Simple Neural Network in Rust");
    println!("================================");
    
    // Demonstrate flexible architecture
    demonstrate_flexible_architecture();
    
    println!("\n{}", "=".repeat(50));
    
    // Demonstrate XOR problem
    solve_xor_problem();
    
    println!("\n{}", "=".repeat(50));
    
    // Demonstrate AND problem
    solve_and_problem();
    
    println!("\n{}", "=".repeat(50));
    
    // Demonstrate OR problem
    solve_or_problem();
    
    println!("\n{}", "=".repeat(50));
    
    // Demonstrate Hebbian learning
    demonstrate_hebbian_learning();
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
    test_network(&nn, &training_data);
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
    test_network(&nn, &training_data);
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
    test_network(&nn, &training_data);
}

fn demonstrate_hebbian_learning() {
    println!("\n🧠 Hebbian Learning Preview");
    println!("---------------------------");
    println!("\"Neurons that fire together, wire together\"");
    
    let mut hebbian_nn = NeuralNetwork::with_hebbian_learning(&[2, 3, 1], 0.1, 0.05, 5, 0.001);
    println!("{}", hebbian_nn.info());
    println!("Hebbian rate: {}, History size: {}", hebbian_nn.get_hebbian_rate(), hebbian_nn.get_history_size());
    
    // Show weight before Hebbian learning
    let initial_weight = hebbian_nn.get_weight(0, 0, 0);
    println!("Initial weight (input[0] -> hidden[0]): {:.4}", initial_weight);
    
    // Apply Hebbian learning with correlated inputs
    println!("Training with correlated inputs [1.0, 1.0]...");
    for _ in 0..20 {
        hebbian_nn.train_hebbian(&[1.0, 1.0]); // Both inputs high
    }
    
    let final_weight = hebbian_nn.get_weight(0, 0, 0);
    let correlation = hebbian_nn.get_neuron_correlation(0, 0, 0, 1);
    
    println!("After Hebbian training:");
    println!("  Final weight: {:.4}", final_weight);
    println!("  Input correlation: {:.4}", correlation);
    println!("  Weight change: {:.4}", final_weight - initial_weight);
    
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
    println!("💡 Run 'cargo run --example network_composition' for full demonstration!");
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

fn test_network(nn: &NeuralNetwork, test_data: &[(Vec<f64>, Vec<f64>)]) {
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
