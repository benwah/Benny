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
