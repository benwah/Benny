use neural_network::NeuralNetwork;

fn main() {
    println!("🧠 Hebbian Learning Demonstration");
    println!("==================================");
    println!("\"Neurons that fire together, wire together\"");
    println!();

    // Create a network with Hebbian learning capabilities
    let mut hebbian_nn = NeuralNetwork::with_hebbian_learning(
        &[2, 4, 1],  // 2 inputs, 4 hidden, 1 output
        0.1,         // Backpropagation learning rate
        0.05,        // Hebbian learning rate
        10,          // History size (remember last 10 activations)
        0.001        // Weight decay rate
    );

    println!("🏗️  Network Architecture:");
    println!("   {}", hebbian_nn.info());
    println!("   Hebbian rate: {}", hebbian_nn.get_hebbian_rate());
    println!("   History size: {}", hebbian_nn.get_history_size());
    println!("   Decay rate: {}", hebbian_nn.get_decay_rate());
    println!();

    // Demonstrate pure Hebbian learning (unsupervised)
    println!("🔬 Pure Hebbian Learning (Unsupervised)");
    println!("---------------------------------------");
    
    // Show initial weights
    println!("Initial weight (input[0] -> hidden[0]): {:.4}", hebbian_nn.get_weight(0, 0, 0));
    println!("Initial weight (input[1] -> hidden[0]): {:.4}", hebbian_nn.get_weight(0, 1, 0));
    println!();

    // Train with patterns where input[0] and input[1] are often active together
    println!("Training with correlated patterns...");
    let correlated_patterns = [
        [1.0, 1.0],  // Both high
        [1.0, 0.9],  // Both high
        [0.9, 1.0],  // Both high
        [0.8, 0.8],  // Both high
        [0.0, 0.0],  // Both low
        [0.1, 0.0],  // Both low
        [0.0, 0.1],  // Both low
    ];

    for epoch in 0..100 {
        for pattern in &correlated_patterns {
            hebbian_nn.train_hebbian(pattern);
        }
        
        if epoch % 20 == 0 {
            let avg_activation_0 = hebbian_nn.get_average_activation(0, 0);
            let avg_activation_1 = hebbian_nn.get_average_activation(0, 1);
            let correlation = hebbian_nn.get_neuron_correlation(0, 0, 0, 1);
            
            println!("  Epoch {}: Avg activations: [{:.3}, {:.3}], Correlation: {:.3}", 
                     epoch, avg_activation_0, avg_activation_1, correlation);
        }
    }

    // Show final weights
    println!();
    println!("Final weight (input[0] -> hidden[0]): {:.4}", hebbian_nn.get_weight(0, 0, 0));
    println!("Final weight (input[1] -> hidden[0]): {:.4}", hebbian_nn.get_weight(0, 1, 0));
    
    // Calculate final correlation
    let final_correlation = hebbian_nn.get_neuron_correlation(0, 0, 0, 1);
    println!("Final correlation between inputs: {:.4}", final_correlation);
    println!();

    // Demonstrate anti-correlated learning
    println!("🔄 Anti-Correlated Learning");
    println!("---------------------------");
    
    // Reset the network
    let mut anti_nn = NeuralNetwork::with_hebbian_learning(&[2, 4, 1], 0.1, 0.05, 10, 0.001);
    
    println!("Training with anti-correlated patterns...");
    let anti_correlated_patterns = [
        [1.0, 0.0],  // High-Low
        [0.9, 0.1],  // High-Low
        [0.8, 0.0],  // High-Low
        [0.0, 1.0],  // Low-High
        [0.1, 0.9],  // Low-High
        [0.0, 0.8],  // Low-High
    ];

    for epoch in 0..100 {
        for pattern in &anti_correlated_patterns {
            anti_nn.train_hebbian(pattern);
        }
        
        if epoch % 20 == 0 {
            let correlation = anti_nn.get_neuron_correlation(0, 0, 0, 1);
            println!("  Epoch {}: Correlation: {:.3}", epoch, correlation);
        }
    }

    let anti_correlation = anti_nn.get_neuron_correlation(0, 0, 0, 1);
    println!("Final anti-correlation: {:.4}", anti_correlation);
    println!();

    // Demonstrate hybrid learning (Backpropagation + Hebbian)
    println!("🔀 Hybrid Learning (Backpropagation + Hebbian)");
    println!("----------------------------------------------");
    
    let mut hybrid_nn = NeuralNetwork::with_hebbian_learning(&[2, 4, 1], 0.3, 0.02, 10, 0.001);
    
    // XOR problem with Hebbian enhancement
    let xor_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    println!("Training XOR with hybrid learning...");
    for epoch in 0..1000 {
        let mut total_error = 0.0;
        
        for (inputs, targets) in &xor_data {
            let error = hybrid_nn.train_hybrid(inputs, targets);
            total_error += error;
        }
        
        if epoch % 200 == 0 {
            println!("  Epoch {}: Average Error = {:.6}", epoch, total_error / 4.0);
        }
    }

    println!();
    println!("🧪 Testing Hybrid Network:");
    for (inputs, expected) in &xor_data {
        let prediction = hybrid_nn.predict(inputs);
        let predicted_class = if prediction[0] > 0.5 { 1 } else { 0 };
        let expected_class = if expected[0] > 0.5 { 1 } else { 0 };
        let status = if predicted_class == expected_class { "✅" } else { "❌" };
        
        println!("  Input: [{:.1}, {:.1}] -> Expected: {}, Predicted: {:.4} ({}) {}", 
                 inputs[0], inputs[1], expected_class, prediction[0], predicted_class, status);
    }

    println!();
    println!("🔍 Neuron Correlation Analysis:");
    println!("   Input correlation: {:.4}", hybrid_nn.get_neuron_correlation(0, 0, 0, 1));
    println!("   Input[0] -> Hidden[0] correlation: {:.4}", hybrid_nn.get_neuron_correlation(0, 0, 1, 0));
    println!("   Input[1] -> Hidden[0] correlation: {:.4}", hybrid_nn.get_neuron_correlation(0, 1, 1, 0));

    println!();
    println!("💡 Key Insights:");
    println!("   • Hebbian learning strengthens connections between co-active neurons");
    println!("   • Positive correlation emerges from synchronized firing patterns");
    println!("   • Negative correlation emerges from anti-synchronized patterns");
    println!("   • Hybrid learning combines supervised (backprop) and unsupervised (Hebbian) learning");
    println!("   • Weight decay prevents unbounded growth in Hebbian learning");
    println!();
    println!("🎉 Hebbian learning demonstration complete!");
}