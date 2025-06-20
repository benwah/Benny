use neural_network::NeuralNetwork;

fn main() {
    println!("🧠 Simple Hebbian Neural Network in Rust");
    println!("=========================================");
    println!();

    // Create a simple 2-layer network with Classic Hebbian learning
    let mut network = NeuralNetwork::new(2, 3, 1, 0.1);
    
    println!("📊 Network Info: {}", network.info());
    println!();

    // Train with some simple patterns (unsupervised)
    println!("🎯 Training with unsupervised Hebbian learning...");
    let patterns = [
        [1.0, 1.0],  // Both high
        [1.0, 0.8],  // High correlation
        [0.9, 1.0],  // High correlation
        [0.0, 0.0],  // Both low
    ];

    let test_pattern = [1.0, 1.0];
    
    for epoch in 0..50 {
        for pattern in &patterns {
            network.train_unsupervised(pattern);
        }
        
        if epoch % 10 == 0 {
            let (_, output) = network.forward(&test_pattern);
            println!("  Epoch {}: Output for [1.0, 1.0] = {:.4}", epoch, output[0]);
        }
    }

    println!();
    println!("🔗 Final neuron correlations:");
    let correlation = network.get_neuron_correlation(0, 0, 0, 1);
    println!("  Input[0] <-> Input[1]: {:.4}", correlation);

    println!();
    println!("✨ Hebbian learning complete! Neurons that fired together are now wired together.");
}