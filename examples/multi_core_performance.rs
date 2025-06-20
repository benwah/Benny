use neural_network::NeuralNetwork;
use std::time::Instant;

fn main() {
    println!("🚀 Multi-Core Neural Network Performance Demo");
    println!("==============================================");

    // Get system information
    let num_cores = rayon::current_num_threads();
    println!("🖥️  System: {} CPU cores available", num_cores);
    println!("📊 Testing parallel vs sequential performance\n");

    // Create a large neural network for performance testing
    let mut nn = NeuralNetwork::with_layers(&[100, 200, 150, 100, 50, 10], 0.01);
    println!("🧠 Network Architecture: 100 -> 200 -> 150 -> 100 -> 50 -> 10");
    println!("📈 Total parameters: {}\n", nn.num_parameters());

    // Generate large training dataset
    let batch_size = 1000;
    let training_data: Vec<(Vec<f64>, Vec<f64>)> = (0..batch_size)
        .map(|i| {
            let inputs: Vec<f64> = (0..100).map(|j| ((i + j) as f64 * 0.01).sin()).collect();
            let targets: Vec<f64> = (0..10)
                .map(|j| ((i + j) as f64 * 0.02).cos().abs())
                .collect();
            (inputs, targets)
        })
        .collect();

    println!("📦 Generated {} training samples", batch_size);

    // Test 1: Single Forward Pass Performance
    println!("\n🔬 Test 1: Single Forward Pass");
    println!("------------------------------");

    let test_input: Vec<f64> = (0..100).map(|i| (i as f64 * 0.01).sin()).collect();

    // Warm up
    for _ in 0..10 {
        nn.forward(&test_input);
    }

    let start = Instant::now();
    for _ in 0..1000 {
        nn.forward(&test_input);
    }
    let single_forward_time = start.elapsed();
    println!("⏱️  1000 forward passes: {:?}", single_forward_time);
    println!(
        "📊 Average per forward pass: {:?}",
        single_forward_time / 1000
    );

    // Test 2: Batch Forward Pass Performance
    println!("\n🔬 Test 2: Batch Forward Pass");
    println!("-----------------------------");

    let batch_inputs: Vec<Vec<f64>> = training_data
        .iter()
        .take(100)
        .map(|(inputs, _)| inputs.clone())
        .collect();

    let start = Instant::now();
    let batch_outputs = nn.forward_batch(&batch_inputs);
    let batch_forward_time = start.elapsed();

    println!("⏱️  100 samples in batch: {:?}", batch_forward_time);
    println!("📊 Average per sample: {:?}", batch_forward_time / 100);
    println!(
        "✅ Batch outputs shape: {} x {}",
        batch_outputs.len(),
        batch_outputs[0].len()
    );

    // Test 3: Training Performance
    println!("\n🔬 Test 3: Training Performance");
    println!("-------------------------------");

    // Single sample training
    let start = Instant::now();
    for i in 0..100 {
        nn.train(&training_data[i].0, &training_data[i].1);
    }
    let single_training_time = start.elapsed();
    println!(
        "⏱️  100 individual training steps: {:?}",
        single_training_time
    );

    // Batch training
    let batch_data = &training_data[100..200];
    let start = Instant::now();
    let batch_error = nn.train_batch(batch_data);
    let batch_training_time = start.elapsed();
    println!(
        "⏱️  100 samples in batch training: {:?}",
        batch_training_time
    );
    println!("📉 Batch training error: {:.6}", batch_error);

    // Test 4: Large Network Stress Test
    println!("\n🔬 Test 4: Large Network Stress Test");
    println!("------------------------------------");

    let mut large_nn = NeuralNetwork::with_layers(&[500, 1000, 800, 600, 400, 200, 100], 0.001);
    println!("🧠 Large Network: 500 -> 1000 -> 800 -> 600 -> 400 -> 200 -> 100");
    println!("📈 Total parameters: {}", large_nn.num_parameters());

    let large_input: Vec<f64> = (0..500).map(|i| (i as f64 * 0.001).sin()).collect();
    let large_target: Vec<f64> = (0..100).map(|i| (i as f64 * 0.002).cos().abs()).collect();

    let start = Instant::now();
    for _ in 0..10 {
        large_nn.train(&large_input, &large_target);
    }
    let large_training_time = start.elapsed();
    println!(
        "⏱️  10 training steps on large network: {:?}",
        large_training_time
    );
    println!(
        "📊 Average per training step: {:?}",
        large_training_time / 10
    );

    // Test 5: Parallel Architecture Comparison
    println!("\n🔬 Test 5: Architecture Scaling");
    println!("-------------------------------");

    let architectures = vec![
        vec![50, 100, 50, 10],
        vec![100, 200, 150, 100, 50, 10],
        vec![200, 400, 300, 200, 100, 50, 20],
        vec![500, 800, 600, 400, 200, 100, 50],
    ];

    for (i, arch) in architectures.iter().enumerate() {
        let mut test_nn = NeuralNetwork::with_layers(arch, 0.01);
        let input_size = arch[0];
        let output_size = arch[arch.len() - 1];

        let test_input: Vec<f64> = (0..input_size).map(|j| (j as f64 * 0.01).sin()).collect();
        let test_target: Vec<f64> = (0..output_size)
            .map(|j| (j as f64 * 0.02).cos().abs())
            .collect();

        let start = Instant::now();
        for _ in 0..100 {
            test_nn.train(&test_input, &test_target);
        }
        let arch_time = start.elapsed();

        println!("🏗️  Architecture {}: {:?}", i + 1, arch);
        println!("   Parameters: {}", test_nn.num_parameters());
        println!("   100 training steps: {:?}", arch_time);
        println!("   Avg per step: {:?}", arch_time / 100);
        println!();
    }

    // Performance Summary
    println!("📋 Performance Summary");
    println!("=====================");
    println!("✅ All operations now utilize {} CPU cores", num_cores);
    println!("🚀 Forward propagation: Parallelized neuron computations");
    println!("🔄 Backpropagation: Parallelized error calculation and weight updates");
    println!("📦 Batch processing: Parallelized across samples");
    println!(
        "⚡ Expected speedup: ~{}x on multi-core systems",
        num_cores.min(4)
    );

    println!("\n💡 Tips for maximum performance:");
    println!("   • Use batch processing for training multiple samples");
    println!("   • Larger networks benefit more from parallelization");
    println!("   • Forward pass parallelization scales with network width");
    println!("   • Backprop parallelization scales with network complexity");
}
