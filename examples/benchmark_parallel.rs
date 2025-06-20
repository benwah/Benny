use neural_network::NeuralNetwork;
use std::time::Instant;

fn main() {
    println!("🏁 Neural Network Parallel Performance Benchmark");
    println!("================================================");

    let num_cores = rayon::current_num_threads();
    println!("🖥️  Available CPU cores: {}", num_cores);
    println!("📊 Benchmarking parallel optimizations\n");

    // Test different network sizes
    let test_configs = vec![
        ("Small", vec![10, 20, 10], 1000),
        ("Medium", vec![50, 100, 50, 20], 500),
        ("Large", vec![100, 200, 150, 100, 50], 200),
        ("Very Large", vec![200, 400, 300, 200, 100], 100),
    ];

    for (name, architecture, iterations) in test_configs {
        println!("🧠 Testing {} Network: {:?}", name, architecture);
        println!("   Parameters: {}", count_parameters(&architecture));

        let mut nn = NeuralNetwork::with_layers(&architecture, 0.01);
        let input_size = architecture[0];
        let output_size = architecture[architecture.len() - 1];

        // Generate test data
        let test_input: Vec<f64> = (0..input_size).map(|i| (i as f64 * 0.01).sin()).collect();
        let test_target: Vec<f64> = (0..output_size)
            .map(|i| (i as f64 * 0.02).cos().abs())
            .collect();

        // Benchmark forward pass
        let start = Instant::now();
        for _ in 0..iterations {
            nn.forward(&test_input);
        }
        let forward_time = start.elapsed();

        // Benchmark training
        let start = Instant::now();
        for _ in 0..iterations {
            nn.train(&test_input, &test_target);
        }
        let training_time = start.elapsed();

        // Benchmark batch processing
        let batch_data: Vec<(Vec<f64>, Vec<f64>)> = (0..50)
            .map(|i| {
                let inputs: Vec<f64> = (0..input_size)
                    .map(|j| ((i + j) as f64 * 0.01).sin())
                    .collect();
                let targets: Vec<f64> = (0..output_size)
                    .map(|j| ((i + j) as f64 * 0.02).cos().abs())
                    .collect();
                (inputs, targets)
            })
            .collect();

        let start = Instant::now();
        nn.train_batch(&batch_data);
        let batch_time = start.elapsed();

        println!(
            "   ⏱️  {} forward passes: {:?} (avg: {:?})",
            iterations,
            forward_time,
            forward_time / iterations as u32
        );
        println!(
            "   🎯 {} training steps: {:?} (avg: {:?})",
            iterations,
            training_time,
            training_time / iterations as u32
        );
        println!(
            "   📦 50-sample batch: {:?} (avg: {:?})",
            batch_time,
            batch_time / 50
        );

        // Calculate theoretical speedup
        let theoretical_speedup = num_cores
            .min(architecture.iter().max().unwrap() / 10)
            .max(1);
        println!(
            "   ⚡ Expected speedup: ~{}x (limited by network width)",
            theoretical_speedup
        );
        println!();
    }

    // Memory efficiency test
    println!("🧪 Memory Efficiency Test");
    println!("-------------------------");

    let large_batch_sizes = vec![100, 500, 1000, 2000];
    let test_arch = vec![50, 100, 50, 10];
    let mut test_nn = NeuralNetwork::with_layers(&test_arch, 0.01);

    for batch_size in large_batch_sizes {
        let batch_data: Vec<(Vec<f64>, Vec<f64>)> = (0..batch_size)
            .map(|i| {
                let inputs: Vec<f64> = (0..50).map(|j| ((i + j) as f64 * 0.01).sin()).collect();
                let targets: Vec<f64> = (0..10)
                    .map(|j| ((i + j) as f64 * 0.02).cos().abs())
                    .collect();
                (inputs, targets)
            })
            .collect();

        let start = Instant::now();
        let error = test_nn.train_batch(&batch_data);
        let batch_time = start.elapsed();

        println!(
            "📊 Batch size {}: {:?} (avg: {:?}/sample, error: {:.6})",
            batch_size,
            batch_time,
            batch_time / batch_size as u32,
            error
        );
    }

    // Scalability test
    println!("\n🔬 Core Scalability Analysis");
    println!("----------------------------");

    let wide_networks = vec![
        vec![100, 500, 100],         // Very wide hidden layer
        vec![200, 1000, 200],        // Extremely wide
        vec![50, 100, 200, 100, 50], // Deep and wide
    ];

    for arch in wide_networks {
        let mut nn = NeuralNetwork::with_layers(&arch, 0.001);
        let input_size = arch[0];
        let output_size = arch[arch.len() - 1];

        let test_input: Vec<f64> = (0..input_size).map(|i| (i as f64 * 0.001).sin()).collect();
        let test_target: Vec<f64> = (0..output_size)
            .map(|i| (i as f64 * 0.002).cos().abs())
            .collect();

        let start = Instant::now();
        for _ in 0..10 {
            nn.train(&test_input, &test_target);
        }
        let time = start.elapsed();

        let max_width = arch.iter().max().unwrap();
        let parallelization_potential = (max_width / 50).max(1).min(num_cores);

        println!("🏗️  Architecture {:?}", arch);
        println!("   Max width: {} neurons", max_width);
        println!("   10 training steps: {:?}", time);
        println!(
            "   Parallelization potential: {}x",
            parallelization_potential
        );
        println!();
    }

    println!("📋 Summary");
    println!("==========");
    println!("✅ All operations now use parallel processing");
    println!("🚀 Forward pass: Parallel neuron computations within layers");
    println!("🔄 Backprop: Parallel error calculation and weight updates");
    println!("📦 Batch training: Parallel sample processing");
    println!("⚡ Best performance on wide networks with many neurons per layer");
    println!("🖥️  Utilizes all {} available CPU cores", num_cores);

    println!("\n💡 Performance Tips:");
    println!("   • Use batch training for multiple samples");
    println!("   • Wider networks (more neurons per layer) benefit most");
    println!("   • Deep networks benefit from parallel backpropagation");
    println!("   • Consider network architecture for optimal parallelization");
}

fn count_parameters(architecture: &[usize]) -> usize {
    let mut total = 0;
    for i in 0..architecture.len() - 1 {
        // Weights: from_layer * to_layer
        total += architecture[i] * architecture[i + 1];
        // Biases: to_layer
        total += architecture[i + 1];
    }
    total
}
