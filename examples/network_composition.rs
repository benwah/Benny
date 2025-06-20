use neural_network::{NeuralNetwork, NetworkComposer};
use std::collections::HashMap;

fn main() {
    println!("🔗 Neural Network Composition Demo");
    println!("==================================");
    println!("Connecting multiple neural networks together\n");

    // Example 1: Simple Pipeline (nn1 -> nn2)
    simple_pipeline_example();
    
    println!("\n{}", "=".repeat(50));
    
    // Example 2: Fan-out Architecture (nn1 -> nn2, nn3)
    fan_out_example();
    
    println!("\n{}", "=".repeat(50));
    
    // Example 3: Complex Multi-Stage Processing
    complex_processing_example();
    
    println!("\n{}", "=".repeat(50));
    
    // Example 4: Ensemble Learning
    ensemble_example();
}

fn simple_pipeline_example() {
    println!("📊 Example 1: Simple Pipeline");
    println!("-----------------------------");
    println!("Architecture: Input -> NN1 -> NN2 -> Output");
    
    let mut composer = NetworkComposer::new();
    
    // Create networks
    let feature_extractor = NeuralNetwork::new(4, 6, 3, 0.1); // 4 inputs -> 3 features
    let classifier = NeuralNetwork::new(3, 4, 2, 0.1);        // 3 features -> 2 classes
    
    // Add to composer
    composer.add_network("feature_extractor".to_string(), feature_extractor).unwrap();
    composer.add_network("classifier".to_string(), classifier).unwrap();
    
    // Connect: all outputs of feature_extractor -> all inputs of classifier
    composer.connect_networks(
        "feature_extractor", 
        "classifier", 
        vec![0, 1, 2], 
        vec![0, 1, 2]
    ).unwrap();
    
    println!("{}", composer.info());
    
    // Forward propagation
    let mut inputs = HashMap::new();
    inputs.insert("feature_extractor".to_string(), vec![0.8, 0.3, 0.9, 0.2]);
    
    let outputs = composer.forward(&inputs).unwrap();
    println!("Input: {:?}", inputs["feature_extractor"]);
    println!("Features: {:?}", outputs["feature_extractor"]);
    println!("Classification: {:?}", outputs["classifier"]);
    
    // Show execution order
    println!("Execution order: {:?}", composer.get_execution_order());
}

fn fan_out_example() {
    println!("🌟 Example 2: Fan-out Architecture");
    println!("----------------------------------");
    println!("Architecture: Input -> NN1 -> [NN2, NN3]");
    
    let mut composer = NetworkComposer::new();
    
    // Create networks
    let shared_processor = NeuralNetwork::new(3, 5, 4, 0.1);  // 3 inputs -> 4 outputs
    let specialist_a = NeuralNetwork::new(2, 3, 1, 0.1);      // 2 inputs -> 1 output
    let specialist_b = NeuralNetwork::new(2, 3, 1, 0.1);      // 2 inputs -> 1 output
    
    // Add to composer
    composer.add_network("shared".to_string(), shared_processor).unwrap();
    composer.add_network("specialist_a".to_string(), specialist_a).unwrap();
    composer.add_network("specialist_b".to_string(), specialist_b).unwrap();
    
    // Connect shared processor to both specialists
    composer.connect_networks("shared", "specialist_a", vec![0, 1], vec![0, 1]).unwrap();
    composer.connect_networks("shared", "specialist_b", vec![2, 3], vec![0, 1]).unwrap();
    
    println!("{}", composer.info());
    
    // Forward propagation
    let mut inputs = HashMap::new();
    inputs.insert("shared".to_string(), vec![0.7, 0.4, 0.9]);
    
    let outputs = composer.forward(&inputs).unwrap();
    println!("Input: {:?}", inputs["shared"]);
    println!("Shared features: {:?}", outputs["shared"]);
    println!("Specialist A output: {:?}", outputs["specialist_a"]);
    println!("Specialist B output: {:?}", outputs["specialist_b"]);
}

fn complex_processing_example() {
    println!("🧠 Example 3: Complex Multi-Stage Processing");
    println!("--------------------------------------------");
    println!("Architecture: Multiple inputs, multiple processing stages");
    
    let mut composer = NetworkComposer::new();
    
    // Create a complex processing pipeline
    let input_processor_a = NeuralNetwork::new(2, 3, 2, 0.1);
    let input_processor_b = NeuralNetwork::new(3, 4, 2, 0.1);
    let fusion_network = NeuralNetwork::new(4, 6, 3, 0.1);
    let output_network = NeuralNetwork::new(3, 4, 1, 0.1);
    
    // Add networks
    composer.add_network("input_a".to_string(), input_processor_a).unwrap();
    composer.add_network("input_b".to_string(), input_processor_b).unwrap();
    composer.add_network("fusion".to_string(), fusion_network).unwrap();
    composer.add_network("output".to_string(), output_network).unwrap();
    
    // Create connections
    composer.connect_networks("input_a", "fusion", vec![0, 1], vec![0, 1]).unwrap();
    composer.connect_networks("input_b", "fusion", vec![0, 1], vec![2, 3]).unwrap();
    composer.connect_networks("fusion", "output", vec![0, 1, 2], vec![0, 1, 2]).unwrap();
    
    println!("{}", composer.info());
    
    // Forward propagation with multiple inputs
    let mut inputs = HashMap::new();
    inputs.insert("input_a".to_string(), vec![0.6, 0.8]);
    inputs.insert("input_b".to_string(), vec![0.3, 0.7, 0.5]);
    
    let outputs = composer.forward(&inputs).unwrap();
    println!("Input A: {:?}", inputs["input_a"]);
    println!("Input B: {:?}", inputs["input_b"]);
    println!("Processed A: {:?}", outputs["input_a"]);
    println!("Processed B: {:?}", outputs["input_b"]);
    println!("Fused features: {:?}", outputs["fusion"]);
    println!("Final output: {:?}", outputs["output"]);
}

fn ensemble_example() {
    println!("🎯 Example 4: Ensemble Learning");
    println!("-------------------------------");
    println!("Multiple networks voting on the same input");
    
    let mut composer = NetworkComposer::new();
    
    // Create ensemble of classifiers
    let classifier_1 = NeuralNetwork::new(4, 5, 1, 0.1);
    let classifier_2 = NeuralNetwork::new(4, 6, 1, 0.1);
    let classifier_3 = NeuralNetwork::new(4, 4, 1, 0.1);
    let voting_network = NeuralNetwork::new(3, 2, 1, 0.1); // Combines 3 votes -> 1 decision
    
    // Add networks
    composer.add_network("classifier_1".to_string(), classifier_1).unwrap();
    composer.add_network("classifier_2".to_string(), classifier_2).unwrap();
    composer.add_network("classifier_3".to_string(), classifier_3).unwrap();
    composer.add_network("voter".to_string(), voting_network).unwrap();
    
    // Connect all classifiers to the voting network
    composer.connect_networks("classifier_1", "voter", vec![0], vec![0]).unwrap();
    composer.connect_networks("classifier_2", "voter", vec![0], vec![1]).unwrap();
    composer.connect_networks("classifier_3", "voter", vec![0], vec![2]).unwrap();
    
    println!("{}", composer.info());
    
    // Test with same input to all classifiers
    let mut inputs = HashMap::new();
    let test_input = vec![0.8, 0.3, 0.6, 0.9];
    inputs.insert("classifier_1".to_string(), test_input.clone());
    inputs.insert("classifier_2".to_string(), test_input.clone());
    inputs.insert("classifier_3".to_string(), test_input.clone());
    
    let outputs = composer.forward(&inputs).unwrap();
    println!("Input: {:?}", test_input);
    println!("Classifier 1 vote: {:.4}", outputs["classifier_1"][0]);
    println!("Classifier 2 vote: {:.4}", outputs["classifier_2"][0]);
    println!("Classifier 3 vote: {:.4}", outputs["classifier_3"][0]);
    println!("Final ensemble decision: {:.4}", outputs["voter"][0]);
    
    // Training example
    println!("\n🎓 Training Example:");
    println!("Training individual networks and the ensemble...");
    
    // Train each classifier individually (in practice, you'd use different training data)
    let target = vec![1.0];
    let error1 = composer.train_network("classifier_1", &test_input, &target).unwrap();
    let error2 = composer.train_network("classifier_2", &test_input, &target).unwrap();
    let error3 = composer.train_network("classifier_3", &test_input, &target).unwrap();
    
    println!("Training errors - C1: {:.4}, C2: {:.4}, C3: {:.4}", error1, error2, error3);
    
    // Test after training
    let outputs_after = composer.forward(&inputs).unwrap();
    println!("After training:");
    println!("Classifier 1 vote: {:.4}", outputs_after["classifier_1"][0]);
    println!("Classifier 2 vote: {:.4}", outputs_after["classifier_2"][0]);
    println!("Classifier 3 vote: {:.4}", outputs_after["classifier_3"][0]);
    println!("Final ensemble decision: {:.4}", outputs_after["voter"][0]);
}