use neural_network::NeuralNetwork;

fn main() {
    println!("🧠 Flexible Neural Network Architecture Demo");
    println!("============================================\n");

    // Example 1: Simple network (2 inputs, 2 hidden, 1 output)
    println!("📊 Example 1: Simple Network");
    let nn1 = NeuralNetwork::with_layers(&[2, 2, 1], 0.3);
    println!("Architecture: {}", nn1.info());
    println!("Parameters: {}", nn1.num_parameters());
    println!("Hidden layers: {}\n", nn1.num_hidden_layers());

    // Example 2: Deep network (4 inputs, multiple hidden layers, 2 outputs)
    println!("🏗️  Example 2: Deep Network");
    let nn2 = NeuralNetwork::with_layers(&[4, 8, 6, 4, 2], 0.01);
    println!("Architecture: {}", nn2.info());
    println!("Parameters: {}", nn2.num_parameters());
    println!("Hidden layers: {}\n", nn2.num_hidden_layers());

    // Example 3: Wide network (many neurons per layer)
    println!("📏 Example 3: Wide Network");
    let nn3 = NeuralNetwork::with_layers(&[3, 20, 15, 1], 0.05);
    println!("Architecture: {}", nn3.info());
    println!("Parameters: {}", nn3.num_parameters());
    println!("Hidden layers: {}\n", nn3.num_hidden_layers());

    // Example 4: Direct input-output (no hidden layers)
    println!("⚡ Example 4: Direct Network (No Hidden Layers)");
    let nn4 = NeuralNetwork::with_layers(&[3, 2], 0.1);
    println!("Architecture: {}", nn4.info());
    println!("Parameters: {}", nn4.num_parameters());
    println!("Hidden layers: {}\n", nn4.num_hidden_layers());

    // Example 5: Multi-output classification network
    println!("🎯 Example 5: Multi-Output Classification");
    let mut nn5 = NeuralNetwork::with_layers(&[4, 6, 3], 0.2);
    println!("Architecture: {}", nn5.info());
    println!("Parameters: {}", nn5.num_parameters());

    // Train on some sample data
    println!("\nTraining on sample multi-class data...");
    let training_data = vec![
        (vec![1.0, 0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]), // Class 1
        (vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]), // Class 2
        (vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]), // Class 3
        (vec![0.0, 0.0, 1.0, 1.0], vec![1.0, 0.0, 0.0]), // Class 1
    ];

    for epoch in 0..1000 {
        let mut total_error = 0.0;
        for (inputs, targets) in &training_data {
            total_error += nn5.train(inputs, targets);
        }
        if epoch % 200 == 0 {
            println!(
                "  Epoch {}: Average Error = {:.6}",
                epoch,
                total_error / training_data.len() as f64
            );
        }
    }

    // Test predictions
    println!("\nTesting predictions:");
    for (i, (inputs, expected)) in training_data.iter().enumerate() {
        let prediction = nn5.predict(inputs);
        let predicted_class = prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let expected_class = expected
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        println!(
            "  Sample {}: Input {:?} -> Predicted class {} (expected {})",
            i + 1,
            inputs,
            predicted_class,
            expected_class
        );
    }

    println!("\n🎉 Demonstration complete!");
    println!("\n💡 Key Features:");
    println!("   • Flexible layer configuration with with_layers()");
    println!("   • Support for any number of hidden layers");
    println!("   • Backward compatible with original new() method");
    println!("   • Multi-input, multi-output capabilities");
    println!("   • Parameter counting and architecture inspection");
}
