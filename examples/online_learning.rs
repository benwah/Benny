use neural_network::{NeuralNetwork, HebbianLearningMode};

fn main() {
    println!("🧠 Online Hebbian Learning Demonstration");
    println!("=========================================");
    println!("This demonstrates true biological neural plasticity where");
    println!("the network continuously adapts during every forward pass.\n");
    
    // Create a network with online learning enabled
    let mut online_network = NeuralNetwork::with_online_learning(
        &[2, 4, 1], 
        0.05, 
        HebbianLearningMode::Classic
    );
    
    println!("📊 Network Configuration:");
    println!("   Architecture: 2 → 4 → 1");
    println!("   Learning Mode: {:?}", online_network.get_learning_mode());
    println!("   Online Learning: {}", online_network.is_online_learning());
    println!("   Hebbian Rate: 0.05\n");
    
    // Test patterns that will create correlations
    let patterns = vec![
        (vec![1.0, 0.0], "Pattern A: [1.0, 0.0]"),
        (vec![0.0, 1.0], "Pattern B: [0.0, 1.0]"),
        (vec![1.0, 1.0], "Pattern C: [1.0, 1.0]"),
        (vec![0.0, 0.0], "Pattern D: [0.0, 0.0]"),
    ];
    
    println!("🔬 Demonstrating Continuous Adaptation:");
    println!("The network will adapt its weights during each forward pass...\n");
    
    // Show initial weights
    println!("📈 Weight Evolution During Online Learning:");
    
    for step in 0..20 {
        let pattern_idx = step % patterns.len();
        let (inputs, description) = &patterns[pattern_idx];
        
        // Get weight before forward pass
        let weight_before = online_network.get_weight(0, 0, 0);
        
        // Forward pass with online learning (weights will change!)
        let output = online_network.predict(inputs);
        
        // Get weight after forward pass
        let weight_after = online_network.get_weight(0, 0, 0);
        let weight_change = weight_after - weight_before;
        
        if step % 5 == 0 {
            println!("Step {:2}: {} → Output: {:.4}", 
                    step, description, output[0]);
            println!("         Weight[0][0][0]: {:.6} → {:.6} (Δ: {:+.6})", 
                    weight_before, weight_after, weight_change);
            
            // Show neuron correlations
            let correlation = online_network.get_neuron_correlation(0, 0, 0, 1);
            println!("         Input correlation: {:.4}\n", correlation);
        }
    }
    
    println!("🧪 Comparison: Online vs Traditional Learning");
    println!("==============================================\n");
    
    // Create a traditional network (no online learning)
    let traditional_network = NeuralNetwork::with_layers_and_mode(
        &[2, 4, 1], 
        0.05, 
        HebbianLearningMode::Classic
    );
    
    println!("Traditional Network (online_learning = false):");
    
    // Test with same pattern multiple times
    let test_input = vec![1.0, 0.5];
    
    for i in 0..3 {
        let weight_before = traditional_network.get_weight(0, 0, 0);
        let output = traditional_network.predict_static(&test_input);
        let weight_after = traditional_network.get_weight(0, 0, 0);
        
        println!("  Pass {}: Weight unchanged: {:.6} → {:.6}, Output: {:.4}", 
                i + 1, weight_before, weight_after, output[0]);
    }
    
    println!("\nOnline Learning Network (online_learning = true):");
    
    for i in 0..3 {
        let weight_before = online_network.get_weight(0, 0, 0);
        let output = online_network.predict(&test_input);
        let weight_after = online_network.get_weight(0, 0, 0);
        let change = weight_after - weight_before;
        
        println!("  Pass {}: Weight adapted: {:.6} → {:.6} (Δ: {:+.6}), Output: {:.4}", 
                i + 1, weight_before, weight_after, change, output[0]);
    }
    
    println!("\n🎯 Key Insights:");
    println!("• Traditional networks: Weights only change during explicit training");
    println!("• Online learning networks: Weights adapt during every forward pass");
    println!("• This mimics biological neurons that continuously adapt to input patterns");
    println!("• Online learning enables real-time adaptation without separate training phases");
    
    println!("\n🧬 Biological Relevance:");
    println!("• Real neurons don't have separate 'training' and 'inference' modes");
    println!("• Synaptic plasticity occurs continuously based on neural activity");
    println!("• This enables lifelong learning and adaptation to new environments");
    println!("• Online Hebbian learning captures this biological reality");
    
    println!("\n✨ Use Cases for Online Learning:");
    println!("• Real-time adaptation to changing environments");
    println!("• Continuous learning from streaming data");
    println!("• Adaptive control systems");
    println!("• Biological neural network simulation");
    println!("• Lifelong learning AI systems");
    
    println!("\n🎉 Online Learning Demo Complete!");
    println!("The network now continuously adapts to every input it processes!");
}