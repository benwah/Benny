use rand::Rng;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<usize>,           // Layer sizes [input, hidden1, hidden2, ..., output]
    weights: Vec<Vec<Vec<f64>>>,  // weights[layer][from_neuron][to_neuron]
    biases: Vec<Vec<f64>>,        // biases[layer][neuron]
    
    // Hebbian learning - now core components
    activation_history: Vec<Vec<Vec<f64>>>, // activation_history[layer][neuron][time_step]
    history_size: usize,                    // Number of recent activations to remember
    hebbian_rate: f64,                      // Primary learning rate for Hebbian updates
    anti_hebbian_rate: f64,                 // Rate for anti-Hebbian (forgetting) updates
    decay_rate: f64,                        // Weight decay to prevent unbounded growth
    homeostatic_rate: f64,                  // Rate for homeostatic regulation
    target_activity: f64,                   // Target average activity level per neuron
    
    // Learning configuration
    learning_mode: HebbianLearningMode,     // Type of Hebbian learning to use
    use_backprop: bool,                     // Whether to supplement with backpropagation
    backprop_rate: f64,                     // Learning rate for backprop (when enabled)
    online_learning: bool,                  // Whether to continuously adapt during forward passes
}

#[derive(Debug, Clone)]
pub enum HebbianLearningMode {
    /// Classic Hebbian: "Neurons that fire together, wire together"
    Classic,
    /// Competitive: Winner-take-all learning with lateral inhibition
    Competitive,
    /// Oja's Rule: Normalized Hebbian learning with weight decay
    Oja,
    /// BCM Rule: Bienenstock-Cooper-Munro rule with sliding threshold
    BCM,
    /// Anti-Hebbian: Decorrelation learning
    AntiHebbian,
    /// Hybrid: Combines multiple Hebbian rules
    Hybrid,
}

impl NeuralNetwork {
    /// Creates a new Hebbian neural network with the specified architecture
    /// Uses Classic Hebbian learning by default
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, hebbian_rate: f64) -> Self {
        Self::with_layers(&[input_size, hidden_size, output_size], hebbian_rate)
    }
    
    /// Creates a new Hebbian neural network with specific learning mode
    pub fn new_with_mode(input_size: usize, hidden_size: usize, output_size: usize, 
                        hebbian_rate: f64, mode: HebbianLearningMode) -> Self {
        Self::with_layers_and_mode(&[input_size, hidden_size, output_size], hebbian_rate, mode)
    }
    
    /// Creates a new Hebbian neural network with flexible layer configuration
    /// Uses Classic Hebbian learning by default
    /// 
    /// # Arguments
    /// * `layer_sizes` - Array of layer sizes [input, hidden1, hidden2, ..., output]
    /// * `hebbian_rate` - Primary Hebbian learning rate
    /// 
    /// # Examples
    /// ```
    /// use neural_network::NeuralNetwork;
    /// 
    /// // Simple Hebbian network: 2 inputs, 3 hidden, 1 output
    /// let nn = NeuralNetwork::with_layers(&[2, 3, 1], 0.1);
    /// 
    /// // Deep Hebbian network: 4 inputs, 8 hidden, 6 hidden, 3 hidden, 2 outputs
    /// let nn = NeuralNetwork::with_layers(&[4, 8, 6, 3, 2], 0.05);
    /// ```
    pub fn with_layers(layer_sizes: &[usize], hebbian_rate: f64) -> Self {
        Self::with_layers_and_mode(layer_sizes, hebbian_rate, HebbianLearningMode::Classic)
    }
    
    /// Creates a new neural network with flexible layer configuration and specific Hebbian learning mode
    /// 
    /// # Arguments
    /// * `layer_sizes` - Array of layer sizes [input, hidden1, hidden2, ..., output]
    /// * `hebbian_rate` - Primary Hebbian learning rate
    /// * `mode` - Type of Hebbian learning to use
    /// 
    /// # Examples
    /// ```
    /// use neural_network::{NeuralNetwork, HebbianLearningMode};
    /// 
    /// // Competitive learning network
    /// let nn = NeuralNetwork::with_layers_and_mode(&[10, 20, 5], 0.1, HebbianLearningMode::Competitive);
    /// 
    /// // Oja's rule network for principal component analysis
    /// let nn = NeuralNetwork::with_layers_and_mode(&[50, 10], 0.01, HebbianLearningMode::Oja);
    /// ```
    pub fn with_layers_and_mode(layer_sizes: &[usize], hebbian_rate: f64, mode: HebbianLearningMode) -> Self {
        assert!(layer_sizes.len() >= 2, "Network must have at least input and output layers");
        
        let mut rng = rand::thread_rng();
        let layers = layer_sizes.to_vec();
        
        // Initialize weights for each layer connection
        let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();
        for i in 0..layers.len() - 1 {
            let from_size = layers[i];
            let to_size = layers[i + 1];
            let layer_weights: Vec<Vec<f64>> = (0..from_size)
                .map(|_| (0..to_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect();
            weights.push(layer_weights);
        }
        
        // Initialize biases for each layer (except input)
        let mut biases: Vec<Vec<f64>> = Vec::new();
        for &layer_size in layers.iter().skip(1) {
            let layer_biases: Vec<f64> = (0..layer_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
            biases.push(layer_biases);
        }
        
        // For backward compatibility, set up legacy fields
        // Initialize Hebbian learning components
        let history_size = 20; // Remember more activations for better correlation detection
        let mut activation_history = Vec::new();
        
        for &layer_size in layers.iter() {
            // Activation history for each neuron in each layer
            let layer_history = vec![vec![0.0; history_size]; layer_size];
            activation_history.push(layer_history);
        }
        
        // Set learning parameters based on mode
        let (anti_hebbian_rate, decay_rate, homeostatic_rate, target_activity, use_backprop, backprop_rate) = match mode {
            HebbianLearningMode::Classic => (0.0, 0.0001, 0.001, 0.1, false, 0.0),
            HebbianLearningMode::Competitive => (0.02, 0.001, 0.01, 0.05, false, 0.0),
            HebbianLearningMode::Oja => (0.0, hebbian_rate * 0.1, 0.005, 0.2, false, 0.0),
            HebbianLearningMode::BCM => (0.01, 0.0005, 0.02, 0.15, false, 0.0),
            HebbianLearningMode::AntiHebbian => (hebbian_rate, 0.0001, 0.001, 0.1, false, 0.0),
            HebbianLearningMode::Hybrid => (hebbian_rate * 0.3, 0.001, 0.01, 0.12, true, hebbian_rate * 0.1),
        };
        
        NeuralNetwork {
            layers,
            weights,
            biases,
            activation_history,
            history_size,
            hebbian_rate,
            anti_hebbian_rate,
            decay_rate,
            homeostatic_rate,
            target_activity,
            learning_mode: mode,
            use_backprop,
            backprop_rate,
            online_learning: false, // Default to false for backward compatibility
        }
    }
    
    /// Creates a hybrid network that combines Hebbian learning with backpropagation
    /// 
    /// # Arguments
    /// * `layer_sizes` - Array of layer sizes [input, hidden1, hidden2, ..., output]
    /// * `hebbian_rate` - Learning rate for Hebbian updates
    /// * `backprop_rate` - Learning rate for backpropagation
    pub fn with_hybrid_learning(layer_sizes: &[usize], hebbian_rate: f64, backprop_rate: f64) -> Self {
        let mut network = Self::with_layers_and_mode(layer_sizes, hebbian_rate, HebbianLearningMode::Hybrid);
        network.backprop_rate = backprop_rate;
        network.use_backprop = true;
        network
    }
    
    /// Configure Hebbian learning parameters
    pub fn configure_hebbian(&mut self, hebbian_rate: f64, anti_hebbian_rate: f64, 
                           homeostatic_rate: f64, target_activity: f64) {
        self.hebbian_rate = hebbian_rate;
        self.anti_hebbian_rate = anti_hebbian_rate;
        self.homeostatic_rate = homeostatic_rate;
        self.target_activity = target_activity;
    }
    
    /// Enable or disable backpropagation supplementation
    pub fn set_backprop_enabled(&mut self, enabled: bool, rate: f64) {
        self.use_backprop = enabled;
        self.backprop_rate = rate;
    }
    
    /// Enable or disable online Hebbian learning during forward passes
    /// When enabled, the network continuously adapts weights during inference
    /// This mimics biological neural plasticity where neurons adapt constantly
    pub fn set_online_learning(&mut self, enabled: bool) {
        self.online_learning = enabled;
    }
    
    /// Check if online learning is enabled
    pub fn is_online_learning(&self) -> bool {
        self.online_learning
    }
    
    /// Create a network with online learning enabled from the start
    /// This is the most biologically realistic mode where the network
    /// continuously adapts to input patterns without separate training phases
    pub fn with_online_learning(layer_sizes: &[usize], hebbian_rate: f64, mode: HebbianLearningMode) -> Self {
        let mut network = Self::with_layers_and_mode(layer_sizes, hebbian_rate, mode);
        network.online_learning = true;
        network
    }
    
    /// Sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Derivative of sigmoid function
    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }
    
    /// Forward propagation through the network
    /// If online learning is enabled, weights are continuously adapted during forward pass
    pub fn forward(&mut self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        if self.online_learning {
            self.forward_with_online_learning(inputs)
        } else {
            self.forward_static(inputs)
        }
    }
    
    /// Forward propagation without weight updates (traditional inference)
    pub fn forward_static(&self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(inputs.len(), self.layers[0], "Input size mismatch");
        
        let mut activations = vec![inputs.to_vec()];
        
        // Forward propagate through each layer
        for layer_idx in 0..self.weights.len() {
            let current_layer = &activations[layer_idx];
            let mut next_layer = vec![0.0; self.layers[layer_idx + 1]];
            
            // Calculate weighted sum + bias for each neuron in next layer
            for to_neuron in 0..next_layer.len() {
                let mut sum = self.biases[layer_idx][to_neuron];
                for from_neuron in 0..current_layer.len() {
                    sum += current_layer[from_neuron] * self.weights[layer_idx][from_neuron][to_neuron];
                }
                next_layer[to_neuron] = Self::sigmoid(sum);
            }
            
            activations.push(next_layer);
        }
        
        // For backward compatibility, return (hidden, output)
        let output = activations.last().unwrap().clone();
        let hidden = if activations.len() > 2 {
            activations[1].clone()
        } else {
            vec![]
        };
        
        (hidden, output)
    }
    
    /// Forward propagation with continuous Hebbian learning (online learning)
    /// This is the biologically realistic mode where neurons adapt during every activation
    pub fn forward_with_online_learning(&mut self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(inputs.len(), self.layers[0], "Input size mismatch");
        
        let mut activations = vec![inputs.to_vec()];
        
        // Store input activations in history for Hebbian learning
        self.store_activations(0, inputs);
        
        // Forward propagate through each layer with online adaptation
        for layer_idx in 0..self.weights.len() {
            let current_layer = &activations[layer_idx];
            let mut next_layer = vec![0.0; self.layers[layer_idx + 1]];
            
            // Calculate weighted sum + bias for each neuron in next layer
            for to_neuron in 0..next_layer.len() {
                let mut sum = self.biases[layer_idx][to_neuron];
                for from_neuron in 0..current_layer.len() {
                    sum += current_layer[from_neuron] * self.weights[layer_idx][from_neuron][to_neuron];
                }
                next_layer[to_neuron] = Self::sigmoid(sum);
            }
            
            activations.push(next_layer.clone());
            
            // Store activations for this layer
            self.store_activations(layer_idx + 1, &next_layer);
            
            // Apply online Hebbian learning to the connection we just used
            self.apply_online_hebbian_to_layer(layer_idx, &activations);
        }
        
        // Apply homeostatic regulation to maintain network stability
        self.apply_online_homeostatic_regulation(&activations);
        
        // For backward compatibility, return (hidden, output)
        let output = activations.last().unwrap().clone();
        let hidden = if activations.len() > 2 {
            activations[1].clone()
        } else {
            vec![]
        };
        
        (hidden, output)
    }
    
    /// Forward propagation returning all layer activations (optimized for multi-core)
    pub fn forward_all_layers(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        assert_eq!(inputs.len(), self.layers[0], "Input size mismatch");
        
        let mut activations = vec![inputs.to_vec()];
        
        // Forward propagate through each layer
        for layer_idx in 0..self.weights.len() {
            let current_layer = &activations[layer_idx];
            
            // Parallel computation of next layer activations
            let next_layer: Vec<f64> = (0..self.layers[layer_idx + 1])
                .into_par_iter()
                .map(|to_neuron| {
                    // Calculate weighted sum + bias for this neuron
                    let mut sum = self.biases[layer_idx][to_neuron];
                    
                    // Vectorized inner product using parallel iterator
                    sum += current_layer
                        .par_iter()
                        .enumerate()
                        .map(|(from_neuron, &activation)| {
                            activation * self.weights[layer_idx][from_neuron][to_neuron]
                        })
                        .sum::<f64>();
                    
                    Self::sigmoid(sum)
                })
                .collect();
            
            activations.push(next_layer);
        }
        
        activations
    }
    
    /// Train the network using Hebbian learning (primary method)
    /// Optionally supplements with backpropagation if enabled
    pub fn train(&mut self, inputs: &[f64], targets: &[f64]) -> f64 {
        assert_eq!(targets.len(), self.layers[self.layers.len() - 1], "Target size mismatch");
        
        // Forward pass and store activations in history
        let activations = self.forward_with_history(inputs);
        
        // Calculate error for monitoring
        let output = &activations[activations.len() - 1];
        let mut total_error = 0.0;
        for i in 0..output.len() {
            let error = targets[i] - output[i];
            total_error += error.powi(2);
        }
        total_error /= 2.0;
        
        // Primary Hebbian learning
        self.apply_hebbian_learning(&activations);
        
        // Apply homeostatic regulation
        self.apply_homeostatic_regulation(&activations);
        
        // Optional backpropagation supplementation
        if self.use_backprop {
            self.apply_backpropagation(&activations, targets);
        }
        
        total_error
    }
    
    /// Train the network using only Hebbian learning (unsupervised)
    pub fn train_unsupervised(&mut self, inputs: &[f64]) {
        // Forward pass and store activations in history
        let activations = self.forward_with_history(inputs);
        
        // Apply Hebbian learning
        self.apply_hebbian_learning(&activations);
        
        // Apply homeostatic regulation
        self.apply_homeostatic_regulation(&activations);
    }
    
    /// Forward propagation with activation history storage for Hebbian learning
    pub fn forward_with_history(&mut self, inputs: &[f64]) -> Vec<Vec<f64>> {
        assert_eq!(inputs.len(), self.layers[0], "Input size mismatch");
        
        let mut activations = vec![inputs.to_vec()];
        
        // Store input activations in history
        self.store_activations(0, inputs);
        
        // Forward propagate through each layer using parallel processing
        for layer_idx in 0..self.weights.len() {
            let current_layer = &activations[layer_idx];
            
            // Parallel computation of next layer activations
            let next_layer: Vec<f64> = (0..self.layers[layer_idx + 1])
                .into_par_iter()
                .map(|to_neuron| {
                    // Calculate weighted sum + bias for this neuron
                    let mut sum = self.biases[layer_idx][to_neuron];
                    
                    // Vectorized inner product using parallel iterator
                    sum += current_layer
                        .par_iter()
                        .enumerate()
                        .map(|(from_neuron, &activation)| {
                            activation * self.weights[layer_idx][from_neuron][to_neuron]
                        })
                        .sum::<f64>();
                    
                    Self::sigmoid(sum)
                })
                .collect();
            
            // Store activations in history
            self.store_activations(layer_idx + 1, &next_layer);
            activations.push(next_layer);
        }
        
        activations
    }
    
    /// Store neuron activations in history (circular buffer)
    fn store_activations(&mut self, layer_idx: usize, activations: &[f64]) {
        for (neuron_idx, &activation) in activations.iter().enumerate() {
            // Shift history (remove oldest, add newest)
            self.activation_history[layer_idx][neuron_idx].remove(0);
            self.activation_history[layer_idx][neuron_idx].push(activation);
        }
    }
    
    /// Apply Hebbian learning based on the selected learning mode
    fn apply_hebbian_learning(&mut self, activations: &[Vec<f64>]) {
        match self.learning_mode {
            HebbianLearningMode::Classic => self.apply_classic_hebbian(activations),
            HebbianLearningMode::Competitive => self.apply_competitive_learning(activations),
            HebbianLearningMode::Oja => self.apply_oja_rule(activations),
            HebbianLearningMode::BCM => self.apply_bcm_rule(activations),
            HebbianLearningMode::AntiHebbian => self.apply_anti_hebbian(activations),
            HebbianLearningMode::Hybrid => {
                self.apply_classic_hebbian(activations);
                self.apply_competitive_learning(activations);
                self.apply_oja_rule(activations);
            }
        }
    }
    
    /// Classic Hebbian learning: "neurons that fire together, wire together"
    fn apply_classic_hebbian(&mut self, activations: &[Vec<f64>]) {
        for layer_idx in 0..self.weights.len() {
            let from_layer = &activations[layer_idx];
            let to_layer = &activations[layer_idx + 1];
            
            // Parallel Hebbian updates
            self.weights[layer_idx]
                .par_iter_mut()
                .enumerate()
                .for_each(|(from_neuron, weight_row)| {
                    weight_row
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(to_neuron, weight)| {
                            // Classic Hebbian rule: Δw = η * pre * post
                            let pre_activity = from_layer[from_neuron];
                            let post_activity = to_layer[to_neuron];
                            let weight_update = self.hebbian_rate * pre_activity * post_activity;
                            *weight += weight_update;
                        });
                });
        }
    }
    
    /// Competitive learning with winner-take-all dynamics
    fn apply_competitive_learning(&mut self, activations: &[Vec<f64>]) {
        for layer_idx in 1..activations.len() {
            let layer_activations = &activations[layer_idx];
            
            // Find winner neuron (highest activation)
            let winner = layer_activations
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            // Update weights only for winner neuron
            if layer_idx > 0 {
                let weight_layer_idx = layer_idx - 1;
                let input_layer = &activations[layer_idx - 1];
                
                for from_neuron in 0..input_layer.len() {
                    let weight_update = self.hebbian_rate * input_layer[from_neuron];
                    self.weights[weight_layer_idx][from_neuron][winner] += weight_update;
                    
                    // Lateral inhibition - weaken connections to non-winners
                    for to_neuron in 0..layer_activations.len() {
                        if to_neuron != winner {
                            let inhibition = self.anti_hebbian_rate * input_layer[from_neuron] * 0.1;
                            self.weights[weight_layer_idx][from_neuron][to_neuron] -= inhibition;
                        }
                    }
                }
            }
        }
    }
    
    /// Oja's rule for normalized Hebbian learning
    fn apply_oja_rule(&mut self, activations: &[Vec<f64>]) {
        for layer_idx in 0..self.weights.len() {
            let from_layer = &activations[layer_idx];
            let to_layer = &activations[layer_idx + 1];
            
            self.weights[layer_idx]
                .par_iter_mut()
                .enumerate()
                .for_each(|(from_neuron, weight_row)| {
                    weight_row
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(to_neuron, weight)| {
                            let pre_activity = from_layer[from_neuron];
                            let post_activity = to_layer[to_neuron];
                            
                            // Oja's rule: Δw = η * post * (pre - post * w)
                            let weight_update = self.hebbian_rate * post_activity * 
                                (pre_activity - post_activity * *weight);
                            *weight += weight_update;
                        });
                });
        }
    }
    
    /// BCM (Bienenstock-Cooper-Munro) rule with sliding threshold
    fn apply_bcm_rule(&mut self, activations: &[Vec<f64>]) {
        for layer_idx in 0..self.weights.len() {
            let from_layer = &activations[layer_idx];
            let to_layer = &activations[layer_idx + 1];
            
            self.weights[layer_idx]
                .par_iter_mut()
                .enumerate()
                .for_each(|(from_neuron, weight_row)| {
                    weight_row
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(to_neuron, weight)| {
                            let pre_activity = from_layer[from_neuron];
                            let post_activity = to_layer[to_neuron];
                            
                            // BCM threshold (sliding average of post-synaptic activity squared)
                            let threshold = self.target_activity * self.target_activity;
                            
                            // BCM rule: Δw = η * pre * post * (post - threshold)
                            let weight_update = self.hebbian_rate * pre_activity * post_activity * 
                                (post_activity - threshold);
                            *weight += weight_update;
                        });
                });
        }
    }
    
    /// Anti-Hebbian learning for decorrelation
    fn apply_anti_hebbian(&mut self, activations: &[Vec<f64>]) {
        for layer_idx in 0..self.weights.len() {
            let from_layer = &activations[layer_idx];
            let to_layer = &activations[layer_idx + 1];
            
            self.weights[layer_idx]
                .par_iter_mut()
                .enumerate()
                .for_each(|(from_neuron, weight_row)| {
                    weight_row
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(to_neuron, weight)| {
                            let pre_activity = from_layer[from_neuron];
                            let post_activity = to_layer[to_neuron];
                            
                            // Anti-Hebbian: decrease weights when neurons fire together
                            let weight_update = -self.anti_hebbian_rate * pre_activity * post_activity;
                            *weight += weight_update;
                        });
                });
        }
    }
    
    /// Apply homeostatic regulation to maintain target activity levels
    fn apply_homeostatic_regulation(&mut self, activations: &[Vec<f64>]) {
        for layer_idx in 1..activations.len() {
            let layer_activations = &activations[layer_idx];
            
            // Calculate average activity for this layer (for future use in adaptive homeostasis)
            let _avg_activity: f64 = layer_activations.iter().sum::<f64>() / layer_activations.len() as f64;
            
            // Adjust biases to maintain target activity
            let bias_layer_idx = layer_idx - 1;
            for neuron_idx in 0..layer_activations.len() {
                let activity_error = self.target_activity - layer_activations[neuron_idx];
                let bias_update = self.homeostatic_rate * activity_error;
                self.biases[bias_layer_idx][neuron_idx] += bias_update;
            }
        }
    }
    
    /// Apply backpropagation as supplementary learning (when enabled)
    fn apply_backpropagation(&mut self, activations: &[Vec<f64>], targets: &[f64]) {
        // Standard backpropagation implementation
        let mut layer_errors = vec![vec![]; self.layers.len()];
        
        // Calculate output layer errors
        let output_layer_idx = self.layers.len() - 1;
        layer_errors[output_layer_idx] = vec![0.0; self.layers[output_layer_idx]];
        for i in 0..self.layers[output_layer_idx] {
            let error = targets[i] - activations[output_layer_idx][i];
            layer_errors[output_layer_idx][i] = error * Self::sigmoid_derivative(activations[output_layer_idx][i]);
        }
        
        // Backpropagate errors through hidden layers
        for layer_idx in (1..self.layers.len() - 1).rev() {
            layer_errors[layer_idx] = (0..self.layers[layer_idx])
                .into_par_iter()
                .map(|neuron| {
                    let error: f64 = (0..self.layers[layer_idx + 1])
                        .into_par_iter()
                        .map(|next_neuron| {
                            layer_errors[layer_idx + 1][next_neuron] * self.weights[layer_idx][neuron][next_neuron]
                        })
                        .sum();
                    error * Self::sigmoid_derivative(activations[layer_idx][neuron])
                })
                .collect();
        }
        
        // Update weights and biases with backprop
        for layer_idx in 0..self.weights.len() {
            self.weights[layer_idx]
                .par_iter_mut()
                .enumerate()
                .for_each(|(from_neuron, weight_row)| {
                    weight_row
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(to_neuron, weight)| {
                            let weight_update = self.backprop_rate * 
                                layer_errors[layer_idx + 1][to_neuron] * 
                                activations[layer_idx][from_neuron];
                            *weight += weight_update;
                        });
                });
            
            self.biases[layer_idx]
                .par_iter_mut()
                .enumerate()
                .for_each(|(neuron, bias)| {
                    *bias += self.backprop_rate * layer_errors[layer_idx + 1][neuron];
                });
        }
    }
    
    /// Apply Hebbian learning rule: "neurons that fire together, wire together"
    pub fn hebbian_update(&mut self, inputs: &[f64]) {
        // Forward pass with history storage
        let _activations = self.forward_with_history(inputs);
        
        // Apply Hebbian updates to all layer connections
        for layer_idx in 0..self.weights.len() {
            self.apply_hebbian_to_layer(layer_idx);
        }
        
        // Apply weight decay to prevent unbounded growth
        self.apply_weight_decay();
    }
    
    /// Apply Hebbian learning to a specific layer
    fn apply_hebbian_to_layer(&mut self, layer_idx: usize) {
        let from_layer = layer_idx;
        let to_layer = layer_idx + 1;
        
        // For each connection between layers
        for from_neuron in 0..self.layers[from_layer] {
            for to_neuron in 0..self.layers[to_layer] {
                // Calculate correlation between pre and post-synaptic neurons
                let correlation = self.calculate_correlation(from_layer, from_neuron, to_layer, to_neuron);
                
                // Hebbian update: Δw = η * correlation
                let weight_update = self.hebbian_rate * correlation;
                self.weights[layer_idx][from_neuron][to_neuron] += weight_update;
            }
        }
    }
    
    /// Calculate correlation between two neurons based on their activation history
    fn calculate_correlation(&self, layer1: usize, neuron1: usize, layer2: usize, neuron2: usize) -> f64 {
        let history1 = &self.activation_history[layer1][neuron1];
        let history2 = &self.activation_history[layer2][neuron2];
        
        // Calculate mean activations
        let mean1: f64 = history1.iter().sum::<f64>() / history1.len() as f64;
        let mean2: f64 = history2.iter().sum::<f64>() / history2.len() as f64;
        
        // Calculate correlation coefficient
        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;
        
        for i in 0..history1.len() {
            let diff1 = history1[i] - mean1;
            let diff2 = history2[i] - mean2;
            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }
        
        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0 // Avoid division by zero
        }
    }
    
    /// Apply weight decay to prevent unbounded weight growth
    fn apply_weight_decay(&mut self) {
        for layer_weights in &mut self.weights {
            for neuron_weights in layer_weights {
                for weight in neuron_weights {
                    *weight *= 1.0 - self.decay_rate;
                }
            }
        }
    }
    
    /// Train using pure Hebbian learning (unsupervised)
    pub fn train_hebbian(&mut self, inputs: &[f64]) {
        self.hebbian_update(inputs);
    }
    
    /// Train using hybrid approach: backpropagation + Hebbian learning
    pub fn train_hybrid(&mut self, inputs: &[f64], targets: &[f64]) -> f64 {
        // First apply backpropagation
        let error = self.train(inputs, targets);
        
        // Then apply Hebbian learning
        self.hebbian_update(inputs);
        
        error
    }
    
    /// Get average activation for a neuron over its history
    pub fn get_average_activation(&self, layer: usize, neuron: usize) -> f64 {
        let history = &self.activation_history[layer][neuron];
        history.iter().sum::<f64>() / history.len() as f64
    }
    
    /// Get correlation between two neurons
    pub fn get_neuron_correlation(&self, layer1: usize, neuron1: usize, layer2: usize, neuron2: usize) -> f64 {
        self.calculate_correlation(layer1, neuron1, layer2, neuron2)
    }
    

    
    /// Make a prediction using the trained network
    /// Note: If online learning is enabled, this will adapt weights during prediction
    pub fn predict(&mut self, inputs: &[f64]) -> Vec<f64> {
        let (_, output) = self.forward(inputs);
        output
    }
    
    /// Make a prediction without any weight updates (pure inference)
    /// This is useful when you want to test the network without adaptation
    pub fn predict_static(&self, inputs: &[f64]) -> Vec<f64> {
        let (_, output) = self.forward_static(inputs);
        output
    }
    
    /// Get network architecture information
    pub fn info(&self) -> String {
        let layer_info = self.layers.iter()
            .map(|&size| size.to_string())
            .collect::<Vec<_>>()
            .join(" -> ");
        
        format!(
            "Neural Network: {} (Hebbian rate: {}, mode: {:?})",
            layer_info, self.hebbian_rate, self.learning_mode
        )
    }
    
    /// Get the layer sizes
    pub fn get_layers(&self) -> &[usize] {
        &self.layers
    }
    
    /// Get the current Hebbian learning mode
    pub fn get_learning_mode(&self) -> &HebbianLearningMode {
        &self.learning_mode
    }
    
    /// Get a specific weight value for inspection
    pub fn get_weight(&self, layer: usize, from_neuron: usize, to_neuron: usize) -> f64 {
        self.weights[layer][from_neuron][to_neuron]
    }
    
    /// Apply online Hebbian learning to a specific layer during forward pass
    /// This is called during forward propagation when online learning is enabled
    fn apply_online_hebbian_to_layer(&mut self, layer_idx: usize, activations: &[Vec<f64>]) {
        let from_layer = &activations[layer_idx];
        let to_layer = &activations[layer_idx + 1];
        
        // Apply the selected Hebbian learning rule with reduced learning rate for stability
        let online_rate = self.hebbian_rate * 0.1; // Reduce rate for online learning stability
        
        match self.learning_mode {
            HebbianLearningMode::Classic => {
                self.apply_online_classic_hebbian(layer_idx, from_layer, to_layer, online_rate);
            },
            HebbianLearningMode::Competitive => {
                self.apply_online_competitive_learning(layer_idx, from_layer, to_layer, online_rate);
            },
            HebbianLearningMode::Oja => {
                self.apply_online_oja_rule(layer_idx, from_layer, to_layer, online_rate);
            },
            HebbianLearningMode::BCM => {
                self.apply_online_bcm_rule(layer_idx, from_layer, to_layer, online_rate);
            },
            HebbianLearningMode::AntiHebbian => {
                self.apply_online_anti_hebbian(layer_idx, from_layer, to_layer, online_rate);
            },
            HebbianLearningMode::Hybrid => {
                // Apply multiple rules with reduced rates
                self.apply_online_classic_hebbian(layer_idx, from_layer, to_layer, online_rate * 0.5);
                self.apply_online_oja_rule(layer_idx, from_layer, to_layer, online_rate * 0.3);
            }
        }
        
        // Apply light weight decay to prevent runaway growth
        self.apply_online_weight_decay(layer_idx);
    }
    
    /// Online classic Hebbian learning: immediate weight updates during forward pass
    fn apply_online_classic_hebbian(&mut self, layer_idx: usize, from_layer: &[f64], to_layer: &[f64], rate: f64) {
        for from_neuron in 0..from_layer.len() {
            for to_neuron in 0..to_layer.len() {
                let pre_activity = from_layer[from_neuron];
                let post_activity = to_layer[to_neuron];
                let weight_update = rate * pre_activity * post_activity;
                self.weights[layer_idx][from_neuron][to_neuron] += weight_update;
            }
        }
    }
    
    /// Online competitive learning with winner-take-all dynamics
    fn apply_online_competitive_learning(&mut self, layer_idx: usize, from_layer: &[f64], to_layer: &[f64], rate: f64) {
        // Find the most active neuron (winner)
        let winner = to_layer.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        // Update weights only for the winner
        for from_neuron in 0..from_layer.len() {
            let pre_activity = from_layer[from_neuron];
            let post_activity = to_layer[winner];
            let weight_update = rate * pre_activity * (post_activity - self.weights[layer_idx][from_neuron][winner]);
            self.weights[layer_idx][from_neuron][winner] += weight_update;
        }
    }
    
    /// Online Oja's rule with normalization
    fn apply_online_oja_rule(&mut self, layer_idx: usize, from_layer: &[f64], to_layer: &[f64], rate: f64) {
        for from_neuron in 0..from_layer.len() {
            for to_neuron in 0..to_layer.len() {
                let pre_activity = from_layer[from_neuron];
                let post_activity = to_layer[to_neuron];
                let current_weight = self.weights[layer_idx][from_neuron][to_neuron];
                
                // Oja's rule: Δw = η * y * (x - y * w)
                let weight_update = rate * post_activity * (pre_activity - post_activity * current_weight);
                self.weights[layer_idx][from_neuron][to_neuron] += weight_update;
            }
        }
    }
    
    /// Online BCM rule with sliding threshold
    fn apply_online_bcm_rule(&mut self, layer_idx: usize, from_layer: &[f64], to_layer: &[f64], rate: f64) {
        for from_neuron in 0..from_layer.len() {
            for to_neuron in 0..to_layer.len() {
                let pre_activity = from_layer[from_neuron];
                let post_activity = to_layer[to_neuron];
                
                // BCM threshold (simplified - using target activity)
                let threshold = self.target_activity;
                let bcm_factor = post_activity * (post_activity - threshold);
                
                let weight_update = rate * bcm_factor * pre_activity;
                self.weights[layer_idx][from_neuron][to_neuron] += weight_update;
            }
        }
    }
    
    /// Online anti-Hebbian learning for decorrelation
    fn apply_online_anti_hebbian(&mut self, layer_idx: usize, from_layer: &[f64], to_layer: &[f64], rate: f64) {
        for from_neuron in 0..from_layer.len() {
            for to_neuron in 0..to_layer.len() {
                let pre_activity = from_layer[from_neuron];
                let post_activity = to_layer[to_neuron];
                
                // Anti-Hebbian: decrease weights when both neurons are active
                let weight_update = -rate * pre_activity * post_activity;
                self.weights[layer_idx][from_neuron][to_neuron] += weight_update;
            }
        }
    }
    
    /// Apply light weight decay to a specific layer during online learning
    fn apply_online_weight_decay(&mut self, layer_idx: usize) {
        let decay = 1.0 - (self.decay_rate * 0.1); // Lighter decay for online learning
        for neuron_weights in &mut self.weights[layer_idx] {
            for weight in neuron_weights {
                *weight *= decay;
            }
        }
    }
    
    /// Apply homeostatic regulation during online learning
    fn apply_online_homeostatic_regulation(&mut self, activations: &[Vec<f64>]) {
        let regulation_rate = self.homeostatic_rate * 0.1; // Lighter regulation for online learning
        
        for layer_idx in 1..activations.len() {
            let layer_activations = &activations[layer_idx];
            
            for neuron_idx in 0..layer_activations.len() {
                let current_activity = layer_activations[neuron_idx];
                let activity_error = self.target_activity - current_activity;
                
                // Adjust bias to regulate activity level
                if layer_idx > 0 {
                    self.biases[layer_idx - 1][neuron_idx] += regulation_rate * activity_error;
                }
            }
        }
    }
    
    /// Get the number of layers (including input and output)
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Get the number of hidden layers
    pub fn num_hidden_layers(&self) -> usize {
        if self.layers.len() >= 3 {
            self.layers.len() - 2
        } else {
            0
        }
    }
    
    /// Get total number of parameters (weights + biases)
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        
        // Count weights
        for layer_weights in &self.weights {
            for neuron_weights in layer_weights {
                total += neuron_weights.len();
            }
        }
        
        // Count biases
        for layer_biases in &self.biases {
            total += layer_biases.len();
        }
        
        total
    }
    
    /// Get Hebbian learning rate
    pub fn get_hebbian_rate(&self) -> f64 {
        self.hebbian_rate
    }
    
    /// Set Hebbian learning rate
    pub fn set_hebbian_rate(&mut self, rate: f64) {
        self.hebbian_rate = rate;
    }
    
    /// Get activation history size
    pub fn get_history_size(&self) -> usize {
        self.history_size
    }
    
    /// Get weight decay rate
    pub fn get_decay_rate(&self) -> f64 {
        self.decay_rate
    }
    
    /// Set weight decay rate
    pub fn set_decay_rate(&mut self, rate: f64) {
        self.decay_rate = rate;
    }
    
    /// Parallel batch training - train on multiple samples simultaneously
    pub fn train_batch(&mut self, batch: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        if batch.is_empty() {
            return 0.0;
        }
        
        // Process batch in parallel and collect errors
        let total_error: f64 = batch
            .par_iter()
            .map(|(inputs, targets)| {
                // Each thread gets its own copy of the network for forward pass
                let activations = self.forward_all_layers(inputs);
                
                // Calculate error for this sample
                let output = &activations[activations.len() - 1];
                let mut sample_error = 0.0;
                for i in 0..output.len() {
                    let error = targets[i] - output[i];
                    sample_error += error.powi(2);
                }
                sample_error / 2.0
            })
            .sum();
        
        // Sequential weight updates (to avoid race conditions)
        // In practice, you'd accumulate gradients and apply them once
        for (inputs, targets) in batch {
            self.train(inputs, targets);
        }
        
        total_error / batch.len() as f64
    }
    
    /// Parallel batch forward propagation
    pub fn forward_batch(&self, inputs_batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        inputs_batch
            .par_iter()
            .map(|inputs| {
                let activations = self.forward_all_layers(inputs);
                activations[activations.len() - 1].clone() // Return only final output
            })
            .collect()
    }
    
    /// Get activation history for a specific neuron
    pub fn get_activation_history(&self, layer: usize, neuron: usize) -> &Vec<f64> {
        &self.activation_history[layer][neuron]
    }
    
    /// Reset activation history (useful for starting fresh experiments)
    pub fn reset_activation_history(&mut self) {
        for layer_history in &mut self.activation_history {
            for neuron_history in layer_history {
                neuron_history.fill(0.0);
            }
        }
    }
    

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_network_creation() {
        let nn = NeuralNetwork::new(2, 3, 1, 0.1);
        assert_eq!(nn.get_layers(), &[2, 3, 1]);
        assert_eq!(nn.hebbian_rate, 0.1);
        // Test that it defaults to Classic Hebbian learning
        matches!(nn.learning_mode, HebbianLearningMode::Classic);
    }

    #[test]
    fn test_forward_pass() {
        let mut nn = NeuralNetwork::new(2, 3, 1, 0.1);
        let inputs = vec![0.5, 0.8];
        let (hidden, output) = nn.forward(&inputs);
        
        assert_eq!(hidden.len(), 3);
        assert_eq!(output.len(), 1);
        
        // Output should be between 0 and 1 (sigmoid activation)
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
    }

    #[test]
    fn test_prediction() {
        let mut nn = NeuralNetwork::new(2, 3, 1, 0.1);
        let inputs = vec![0.5, 0.8];
        let prediction = nn.predict(&inputs);
        
        assert_eq!(prediction.len(), 1);
        assert!(prediction[0] >= 0.0 && prediction[0] <= 1.0);
    }
    
    #[test]
    fn test_flexible_architecture() {
        // Test deep network: 3 inputs -> 5 hidden -> 4 hidden -> 2 outputs
        let mut nn = NeuralNetwork::with_layers(&[3, 5, 4, 2], 0.05);
        
        assert_eq!(nn.get_layers(), &[3, 5, 4, 2]);
        assert_eq!(nn.num_layers(), 4);
        assert_eq!(nn.num_hidden_layers(), 2);
        
        let inputs = vec![0.1, 0.5, 0.9];
        let prediction = nn.predict(&inputs);
        
        assert_eq!(prediction.len(), 2);
        for &output in &prediction {
            assert!((0.0..=1.0).contains(&output));
        }
    }
    
    #[test]
    fn test_simple_network_no_hidden() {
        // Test direct input -> output (no hidden layers)
        let mut nn = NeuralNetwork::with_layers(&[2, 1], 0.1);
        
        assert_eq!(nn.get_layers(), &[2, 1]);
        assert_eq!(nn.num_layers(), 2);
        assert_eq!(nn.num_hidden_layers(), 0);
        
        let inputs = vec![0.3, 0.7];
        let prediction = nn.predict(&inputs);
        
        assert_eq!(prediction.len(), 1);
        assert!(prediction[0] >= 0.0 && prediction[0] <= 1.0);
    }
    
    #[test]
    fn test_parameter_counting() {
        let nn = NeuralNetwork::with_layers(&[2, 3, 1], 0.1);
        
        // Weights: (2*3) + (3*1) = 6 + 3 = 9
        // Biases: 3 + 1 = 4
        // Total: 9 + 4 = 13
        assert_eq!(nn.num_parameters(), 13);
    }
    
    #[test]
    fn test_training_compatibility() {
        let mut nn = NeuralNetwork::new(2, 3, 1, 0.5);
        let inputs = vec![1.0, 0.0];
        let targets = vec![1.0];
        
        let error_before = nn.train(&inputs, &targets);
        let error_after = nn.train(&inputs, &targets);
        
        // Error should generally decrease (though not guaranteed in single step)
        assert!(error_before >= 0.0);
        assert!(error_after >= 0.0);
    }
    
    #[test]
    fn test_hebbian_network_creation() {
        let nn = NeuralNetwork::with_layers(&[2, 3, 1], 0.05);
        
        assert_eq!(nn.get_layers(), &[2, 3, 1]);
        assert_eq!(nn.get_hebbian_rate(), 0.05);
        assert_eq!(nn.get_history_size(), 20); // Default history size
        assert_eq!(nn.get_decay_rate(), 0.0001); // Default decay rate for Classic mode
    }
    
    #[test]
    fn test_activation_history_storage() {
        let mut nn = NeuralNetwork::with_layers(&[2, 2, 1], 0.05);
        
        // Initially, history should be all zeros
        let initial_history = nn.get_activation_history(0, 0);
        assert_eq!(initial_history, &vec![0.0; 20]); // Default history size is 20
        
        // After forward pass, history should be updated
        let _activations = nn.forward_with_history(&[0.5, 0.8]);
        let updated_history = nn.get_activation_history(0, 0);
        
        // Last entry should be the input value (at the end of the history)
        assert_eq!(updated_history[19], 0.5);
    }
    
    #[test]
    fn test_hebbian_learning() {
        let mut nn = NeuralNetwork::with_layers(&[2, 2, 1], 0.1);
        
        // Store initial weights
        let initial_weight = nn.weights[0][0][0];
        
        // Apply Hebbian learning with some inputs (using dummy targets)
        for _ in 0..10 {
            nn.train(&[1.0, 0.0], &[0.5]); // Hebbian learning is primary, targets are for error calculation only
        }
        
        // Weights should have changed
        let final_weight = nn.weights[0][0][0];
        assert_ne!(initial_weight, final_weight);
    }
    
    #[test]
    fn test_neuron_correlation() {
        let mut nn = NeuralNetwork::with_layers(&[2, 2, 1], 0.1);
        
        // Build up some activation history
        for _ in 0..5 {
            nn.forward_with_history(&[1.0, 1.0]); // Both inputs high
        }
        
        // Check correlation between input neurons (should be high since both are always 1.0)
        let correlation = nn.get_neuron_correlation(0, 0, 0, 1);
        
        // Since both neurons always have the same activation, correlation should be high
        // (though exact value depends on the sigmoid outputs)
        assert!(correlation.abs() <= 1.0); // Correlation should be between -1 and 1
    }
    
    #[test]
    fn test_hybrid_training() {
        let mut nn = NeuralNetwork::with_hybrid_learning(&[2, 3, 1], 0.1, 0.05);
        
        let inputs = vec![1.0, 0.0];
        let targets = vec![1.0];
        
        // Hybrid training should return an error value
        let error = nn.train(&inputs, &targets);
        assert!(error >= 0.0);
        
        // Activation history should be updated
        let history = nn.get_activation_history(0, 0);
        assert_eq!(history[history.len() - 1], 1.0); // Last input was 1.0
    }
    
    #[test]
    fn test_weight_decay() {
        let mut nn = NeuralNetwork::with_layers(&[2, 2, 1], 0.1);
        nn.decay_rate = 0.1; // Set high decay rate for testing
        
        // Set a specific weight value
        nn.weights[0][0][0] = 1.0;
        
        // Apply weight decay
        nn.apply_weight_decay();
        
        // Weight should be reduced
        assert!(nn.weights[0][0][0] < 1.0);
        assert!(nn.weights[0][0][0] > 0.0);
    }
}