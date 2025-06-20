use neural_network::{HebbianLearningMode, NeuralNetwork};

fn main() {
    println!("🧠 Simple Neural Network in Rust");
    println!("=================================");

    // Create a simple neural network with 2 inputs, 3 hidden neurons, and 1 output
    let mut network = NeuralNetwork::new(2, 3, 1, 0.1);

    println!("📊 Network Architecture:");
    println!("   Input Layer:  2 neurons");
    println!("   Hidden Layer: 3 neurons");
    println!("   Output Layer: 1 neuron");
    println!("   Learning Rate: 0.1");
    println!("   Learning Mode: {:?}", network.get_learning_mode());

    // Training data for XOR problem
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    println!("\n🎯 Training on XOR problem...");

    // Train the network
    for epoch in 0..1000 {
        let mut total_error = 0.0;

        for (inputs, targets) in &training_data {
            let error = network.train(inputs, targets);
            total_error += error;
        }

        if epoch % 200 == 0 {
            println!(
                "   Epoch {}: Average Error = {:.6}",
                epoch,
                total_error / training_data.len() as f64
            );
        }
    }

    println!("\n✅ Training Complete!");

    // Test the trained network
    println!("\n🧪 Testing the network:");
    for (inputs, expected) in &training_data {
        let output = network.predict(inputs);
        println!(
            "   Input: {:?} → Output: {:.3} (Expected: {:.1})",
            inputs, output[0], expected[0]
        );
    }

    // Demonstrate Hebbian learning
    println!("\n🔬 Demonstrating Hebbian Learning:");
    let mut hebbian_network =
        NeuralNetwork::with_layers_and_mode(&[2, 3, 1], 0.1, HebbianLearningMode::Classic);

    println!("   Created network with Classic Hebbian learning");

    // Train with correlated inputs
    for i in 0..100 {
        hebbian_network.train_hebbian(&[1.0, 1.0]);
        if i % 25 == 0 {
            let correlation = hebbian_network.get_neuron_correlation(0, 0, 0, 1);
            println!("   Step {}: Input correlation = {:.4}", i, correlation);
        }
    }

    println!("\n🎉 Simple Neural Network Demo Complete!");
    println!("💡 The network learned to solve XOR and demonstrated Hebbian learning");
}
