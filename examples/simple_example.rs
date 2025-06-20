use neural_network::NeuralNetwork;

fn main() {
    println!("🚀 Simple Neural Network Example");
    println!("=================================");
    
    // Create a neural network for the XOR problem
    let mut nn = NeuralNetwork::new(2, 4, 1, 0.5);
    
    println!("Network Info: {}", nn.info());
    
    // XOR training data
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),  // 0 XOR 0 = 0
        (vec![0.0, 1.0], vec![1.0]),  // 0 XOR 1 = 1
        (vec![1.0, 0.0], vec![1.0]),  // 1 XOR 0 = 1
        (vec![1.0, 1.0], vec![0.0]),  // 1 XOR 1 = 0
    ];
    
    println!("\n📚 Training on XOR problem...");
    
    // Training loop
    for epoch in 0..10000 {
        let mut total_error = 0.0;
        
        for (inputs, targets) in &training_data {
            let error = nn.train(inputs, targets);
            total_error += error;
        }
        
        // Print progress every 2000 epochs
        if epoch % 2000 == 0 {
            println!("Epoch {}: Total Error = {:.6}", epoch, total_error);
        }
    }
    
    println!("\n🎯 Testing the trained network:");
    println!("Input -> Expected | Predicted");
    println!("-----------------------------");
    
    for (inputs, expected) in &training_data {
        let prediction = nn.predict(inputs);
        println!(
            "[{:.0}, {:.0}] -> {:.0}       | {:.4}",
            inputs[0], inputs[1], expected[0], prediction[0]
        );
    }
    
    println!("\n✅ Training complete! The network has learned the XOR function.");
    
    // Test with some intermediate values
    println!("\n🔍 Testing with intermediate values:");
    let test_cases = vec![
        vec![0.5, 0.5],
        vec![0.2, 0.8],
        vec![0.9, 0.1],
    ];
    
    for inputs in test_cases {
        let prediction = nn.predict(&inputs);
        println!("Input: [{:.1}, {:.1}] -> Prediction: {:.4}", inputs[0], inputs[1], prediction[0]);
    }
}