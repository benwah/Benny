use neural_network::{HebbianLearningMode, NeuralNetwork};

fn main() {
    println!("🧠 Comprehensive Hebbian Learning Modes Demo");
    println!("============================================");
    println!("Demonstrating all 6 Hebbian learning modes");
    println!();

    // Test patterns for different learning scenarios
    let correlated_patterns = [
        [1.0, 1.0], // Both high
        [0.9, 0.8], // Both high
        [0.8, 0.9], // Both high
        [0.0, 0.0], // Both low
        [0.1, 0.0], // Both low
        [0.0, 0.1], // Both low
    ];

    let anti_correlated_patterns = [
        [1.0, 0.0], // High-Low
        [0.9, 0.1], // High-Low
        [0.0, 1.0], // Low-High
        [0.1, 0.9], // Low-High
    ];

    let competitive_patterns = [
        [1.0, 0.0], // Winner: input 0
        [0.8, 0.2], // Winner: input 0
        [0.0, 1.0], // Winner: input 1
        [0.2, 0.8], // Winner: input 1
    ];

    // 1. Classic Hebbian Learning
    println!("1️⃣  Classic Hebbian Learning");
    println!("   \"Neurons that fire together, wire together\"");
    println!("   ==========================================");

    let mut classic_nn =
        NeuralNetwork::with_layers_and_mode(&[2, 4, 1], 0.05, HebbianLearningMode::Classic);

    println!("   Network: {}", classic_nn.info());
    let initial_weight = classic_nn.get_weight(0, 0, 0);
    println!("   Initial weight: {:.4}", initial_weight);

    // Train with correlated patterns
    for _ in 0..50 {
        for pattern in &correlated_patterns {
            classic_nn.train_unsupervised(pattern);
        }
    }

    let final_weight = classic_nn.get_weight(0, 0, 0);
    let correlation = classic_nn.get_neuron_correlation(0, 0, 0, 1);
    println!(
        "   Final weight: {:.4} (change: {:.4})",
        final_weight,
        final_weight - initial_weight
    );
    println!("   Input correlation: {:.4}", correlation);
    println!();

    // 2. Competitive Learning
    println!("2️⃣  Competitive Learning");
    println!("   \"Winner takes all\"");
    println!("   ==================");

    let mut competitive_nn =
        NeuralNetwork::with_layers_and_mode(&[2, 4, 1], 0.1, HebbianLearningMode::Competitive);

    println!("   Network: {}", competitive_nn.info());

    // Train with competitive patterns
    for epoch in 0..100 {
        for pattern in &competitive_patterns {
            competitive_nn.train_unsupervised(pattern);
        }

        if epoch % 20 == 0 {
            let avg_act_0 = competitive_nn.get_average_activation(1, 0); // Hidden layer neuron 0
            let avg_act_1 = competitive_nn.get_average_activation(1, 1); // Hidden layer neuron 1
            println!(
                "   Epoch {}: Hidden neurons avg activation: [{:.3}, {:.3}]",
                epoch, avg_act_0, avg_act_1
            );
        }
    }
    println!();

    // 3. Oja's Rule (Normalized Hebbian)
    println!("3️⃣  Oja's Rule (Normalized Hebbian)");
    println!("   \"Prevents weight explosion\"");
    println!("   ===========================");

    let mut oja_nn =
        NeuralNetwork::with_layers_and_mode(&[2, 4, 1], 0.05, HebbianLearningMode::Oja);

    println!("   Network: {}", oja_nn.info());
    let initial_weight_oja = oja_nn.get_weight(0, 0, 0);

    // Train with strong correlated patterns
    for _ in 0..100 {
        for pattern in &correlated_patterns {
            oja_nn.train_unsupervised(pattern);
        }
    }

    let final_weight_oja = oja_nn.get_weight(0, 0, 0);
    println!("   Initial weight: {:.4}", initial_weight_oja);
    println!("   Final weight: {:.4} (normalized)", final_weight_oja);
    println!("   Weight magnitude: {:.4}", final_weight_oja.abs());
    println!();

    // 4. BCM Rule (Bienenstock-Cooper-Munro)
    println!("4️⃣  BCM Rule (Bienenstock-Cooper-Munro)");
    println!("   \"Sliding threshold for stability\"");
    println!("   =================================");

    let mut bcm_nn =
        NeuralNetwork::with_layers_and_mode(&[2, 4, 1], 0.02, HebbianLearningMode::BCM);

    println!("   Network: {}", bcm_nn.info());

    // Train with varying intensity patterns
    let bcm_patterns = [
        [0.2, 0.2], // Low activity
        [0.5, 0.5], // Medium activity
        [0.8, 0.8], // High activity
        [1.0, 1.0], // Very high activity
    ];

    for epoch in 0..50 {
        for pattern in &bcm_patterns {
            bcm_nn.train_unsupervised(pattern);
        }

        if epoch % 10 == 0 {
            let avg_activation = bcm_nn.get_average_activation(1, 0);
            println!(
                "   Epoch {}: Hidden neuron 0 avg activation: {:.3}",
                epoch, avg_activation
            );
        }
    }
    println!();

    // 5. Anti-Hebbian Learning
    println!("5️⃣  Anti-Hebbian Learning");
    println!("   \"Neurons that fire together, wire apart\"");
    println!("   =========================================");

    let mut anti_nn =
        NeuralNetwork::with_layers_and_mode(&[2, 4, 1], 0.05, HebbianLearningMode::AntiHebbian);

    println!("   Network: {}", anti_nn.info());

    // Train with anti-correlated patterns
    for _ in 0..50 {
        for pattern in &anti_correlated_patterns {
            anti_nn.train_unsupervised(pattern);
        }
    }

    let anti_correlation = anti_nn.get_neuron_correlation(0, 0, 0, 1);
    println!("   Final anti-correlation: {:.4}", anti_correlation);
    println!();

    // 6. Hybrid Learning (Hebbian + Backpropagation)
    println!("6️⃣  Hybrid Learning (Hebbian + Backpropagation)");
    println!("   \"Best of both worlds\"");
    println!("   =====================");

    let mut hybrid_nn = NeuralNetwork::with_hybrid_learning(&[2, 4, 1], 0.02, 0.3);

    println!("   Network: {}", hybrid_nn.info());

    // Train on XOR problem with hybrid learning
    let xor_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    println!("   Training XOR problem with hybrid learning...");
    for epoch in 0..500 {
        let mut total_error = 0.0;

        for (inputs, targets) in &xor_data {
            let error = hybrid_nn.train(inputs, targets);
            total_error += error;
        }

        if epoch % 100 == 0 {
            println!(
                "   Epoch {}: Average Error = {:.6}",
                epoch,
                total_error / 4.0
            );
        }
    }

    println!("   Testing XOR results:");
    for (inputs, expected) in &xor_data {
        let prediction = hybrid_nn.predict(inputs);
        let predicted_class = if prediction[0] > 0.5 { 1 } else { 0 };
        let expected_class = if expected[0] > 0.5 { 1 } else { 0 };
        let status = if predicted_class == expected_class {
            "✅"
        } else {
            "❌"
        };

        println!(
            "     [{:.1}, {:.1}] -> Expected: {}, Predicted: {:.4} {}",
            inputs[0], inputs[1], expected_class, prediction[0], status
        );
    }

    let hybrid_correlation = hybrid_nn.get_neuron_correlation(0, 0, 0, 1);
    println!(
        "   Input correlation after hybrid training: {:.4}",
        hybrid_correlation
    );
    println!();

    // Summary
    println!("📊 Summary of Hebbian Learning Modes");
    println!("====================================");
    println!("• Classic: Basic correlation strengthening");
    println!("• Competitive: Winner-take-all dynamics");
    println!("• Oja: Normalized weights prevent explosion");
    println!("• BCM: Adaptive threshold for stability");
    println!("• Anti-Hebbian: Decorrelation learning");
    println!("• Hybrid: Combines unsupervised + supervised learning");
    println!();
    println!("🎯 Each mode serves different learning objectives:");
    println!("   - Use Classic for basic pattern association");
    println!("   - Use Competitive for feature detection");
    println!("   - Use Oja for principal component analysis");
    println!("   - Use BCM for homeostatic plasticity");
    println!("   - Use Anti-Hebbian for decorrelation");
    println!("   - Use Hybrid for complex supervised tasks");
    println!();
    println!("🎉 Hebbian learning modes demonstration complete!");
}
