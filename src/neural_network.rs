use rand::Rng;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<usize>,           // Layer sizes [input, hidden1, hidden2, ..., output]
    weights: Vec<Vec<Vec<f64>>>,  // weights[layer][from_neuron][to_neuron]
    biases: Vec<Vec<f64>>,        // biases[layer][neuron]
    learning_rate: f64,
    
    // Keep legacy fields for backward compatibility
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Vec<Vec<f64>>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    bias_output: Vec<f64>,
}

impl NeuralNetwork {
    /// Creates a new neural network with the specified architecture (backward compatible)
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        // Use the new flexible constructor with a single hidden layer
        Self::with_layers(&[input_size, hidden_size, output_size], learning_rate)
    }
    
    /// Creates a new neural network with flexible layer configuration
    /// 
    /// # Arguments
    /// * `layer_sizes` - Array of layer sizes [input, hidden1, hidden2, ..., output]
    /// * `learning_rate` - Learning rate for training
    /// 
    /// # Examples
    /// ```
    /// use neural_network::NeuralNetwork;
    /// 
    /// // Simple network: 2 inputs, 3 hidden, 1 output
    /// let nn = NeuralNetwork::with_layers(&[2, 3, 1], 0.1);
    /// 
    /// // Deep network: 4 inputs, 8 hidden, 6 hidden, 3 hidden, 2 outputs
    /// let nn = NeuralNetwork::with_layers(&[4, 8, 6, 3, 2], 0.05);
    /// ```
    pub fn with_layers(layer_sizes: &[usize], learning_rate: f64) -> Self {
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
        for i in 1..layers.len() {
            let layer_biases: Vec<f64> = (0..layers[i]).map(|_| rng.gen_range(-1.0..1.0)).collect();
            biases.push(layer_biases);
        }
        
        // For backward compatibility, set up legacy fields
        let input_size = layers[0];
        let output_size = layers[layers.len() - 1];
        let hidden_size = if layers.len() > 2 { layers[1] } else { 0 };
        
        // Legacy weight matrices (for single hidden layer compatibility)
        let weights_input_hidden = if layers.len() > 2 {
            weights[0].clone()
        } else {
            vec![vec![0.0; output_size]; input_size]
        };
        
        let weights_hidden_output = if layers.len() > 2 {
            weights[weights.len() - 1].clone()
        } else {
            weights[0].clone()
        };
        
        let bias_hidden = if layers.len() > 2 {
            biases[0].clone()
        } else {
            vec![0.0; hidden_size]
        };
        
        let bias_output = biases[biases.len() - 1].clone();
        
        NeuralNetwork {
            layers,
            weights,
            biases,
            learning_rate,
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
        }
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
    pub fn forward(&self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>) {
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
    
    /// Forward propagation returning all layer activations (useful for deep networks)
    pub fn forward_all_layers(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
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
        
        activations
    }
    
    /// Train the network using backpropagation
    pub fn train(&mut self, inputs: &[f64], targets: &[f64]) -> f64 {
        assert_eq!(targets.len(), self.layers[self.layers.len() - 1], "Target size mismatch");
        
        // Forward pass - get all layer activations
        let activations = self.forward_all_layers(inputs);
        
        // Calculate total error
        let output = &activations[activations.len() - 1];
        let mut total_error = 0.0;
        for i in 0..output.len() {
            let error = targets[i] - output[i];
            total_error += error.powi(2);
        }
        total_error /= 2.0;
        
        // Backpropagation
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
            layer_errors[layer_idx] = vec![0.0; self.layers[layer_idx]];
            for neuron in 0..self.layers[layer_idx] {
                let mut error = 0.0;
                for next_neuron in 0..self.layers[layer_idx + 1] {
                    error += layer_errors[layer_idx + 1][next_neuron] * self.weights[layer_idx][neuron][next_neuron];
                }
                layer_errors[layer_idx][neuron] = error * Self::sigmoid_derivative(activations[layer_idx][neuron]);
            }
        }
        
        // Update weights and biases
        for layer_idx in 0..self.weights.len() {
            // Update weights
            for from_neuron in 0..self.layers[layer_idx] {
                for to_neuron in 0..self.layers[layer_idx + 1] {
                    let weight_update = self.learning_rate * 
                        layer_errors[layer_idx + 1][to_neuron] * 
                        activations[layer_idx][from_neuron];
                    self.weights[layer_idx][from_neuron][to_neuron] += weight_update;
                }
            }
            
            // Update biases
            for neuron in 0..self.layers[layer_idx + 1] {
                self.biases[layer_idx][neuron] += self.learning_rate * layer_errors[layer_idx + 1][neuron];
            }
        }
        
        // Update legacy fields for backward compatibility
        self.update_legacy_fields();
        
        total_error
    }
    
    /// Update legacy fields for backward compatibility
    fn update_legacy_fields(&mut self) {
        if self.layers.len() > 2 {
            self.weights_input_hidden = self.weights[0].clone();
            self.weights_hidden_output = self.weights[self.weights.len() - 1].clone();
            self.bias_hidden = self.biases[0].clone();
        }
        self.bias_output = self.biases[self.biases.len() - 1].clone();
    }
    
    /// Make a prediction using the trained network
    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let (_, output) = self.forward(inputs);
        output
    }
    
    /// Get network architecture information
    pub fn info(&self) -> String {
        let layer_info = self.layers.iter()
            .map(|&size| size.to_string())
            .collect::<Vec<_>>()
            .join(" -> ");
        
        format!(
            "Neural Network: {} (learning rate: {})",
            layer_info, self.learning_rate
        )
    }
    
    /// Get the layer sizes
    pub fn get_layers(&self) -> &[usize] {
        &self.layers
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_network_creation() {
        let nn = NeuralNetwork::new(2, 3, 1, 0.1);
        assert_eq!(nn.input_size, 2);
        assert_eq!(nn.hidden_size, 3);
        assert_eq!(nn.output_size, 1);
        assert_eq!(nn.learning_rate, 0.1);
        assert_eq!(nn.get_layers(), &[2, 3, 1]);
    }

    #[test]
    fn test_forward_pass() {
        let nn = NeuralNetwork::new(2, 3, 1, 0.1);
        let inputs = vec![0.5, 0.8];
        let (hidden, output) = nn.forward(&inputs);
        
        assert_eq!(hidden.len(), 3);
        assert_eq!(output.len(), 1);
        
        // Output should be between 0 and 1 (sigmoid activation)
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
    }

    #[test]
    fn test_prediction() {
        let nn = NeuralNetwork::new(2, 3, 1, 0.1);
        let inputs = vec![0.5, 0.8];
        let prediction = nn.predict(&inputs);
        
        assert_eq!(prediction.len(), 1);
        assert!(prediction[0] >= 0.0 && prediction[0] <= 1.0);
    }
    
    #[test]
    fn test_flexible_architecture() {
        // Test deep network: 3 inputs -> 5 hidden -> 4 hidden -> 2 outputs
        let nn = NeuralNetwork::with_layers(&[3, 5, 4, 2], 0.05);
        
        assert_eq!(nn.get_layers(), &[3, 5, 4, 2]);
        assert_eq!(nn.num_layers(), 4);
        assert_eq!(nn.num_hidden_layers(), 2);
        
        let inputs = vec![0.1, 0.5, 0.9];
        let prediction = nn.predict(&inputs);
        
        assert_eq!(prediction.len(), 2);
        for &output in &prediction {
            assert!(output >= 0.0 && output <= 1.0);
        }
    }
    
    #[test]
    fn test_simple_network_no_hidden() {
        // Test direct input -> output (no hidden layers)
        let nn = NeuralNetwork::with_layers(&[2, 1], 0.1);
        
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
}