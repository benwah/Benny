use rand::Rng;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Vec<Vec<f64>>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    bias_output: Vec<f64>,
    learning_rate: f64,
}

impl NeuralNetwork {
    /// Creates a new neural network with the specified architecture
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        
        // Initialize weights with random values between -1 and 1
        let weights_input_hidden = (0..input_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        let weights_hidden_output = (0..hidden_size)
            .map(|_| (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        // Initialize biases with random values
        let bias_hidden = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias_output = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
            learning_rate,
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
        // Calculate hidden layer
        let mut hidden = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut sum = self.bias_hidden[i];
            for j in 0..self.input_size {
                sum += inputs[j] * self.weights_input_hidden[j][i];
            }
            hidden[i] = Self::sigmoid(sum);
        }
        
        // Calculate output layer
        let mut output = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            let mut sum = self.bias_output[i];
            for j in 0..self.hidden_size {
                sum += hidden[j] * self.weights_hidden_output[j][i];
            }
            output[i] = Self::sigmoid(sum);
        }
        
        (hidden, output)
    }
    
    /// Train the network using backpropagation
    pub fn train(&mut self, inputs: &[f64], targets: &[f64]) -> f64 {
        // Forward pass
        let (hidden, output) = self.forward(inputs);
        
        // Calculate output layer errors
        let mut output_errors = vec![0.0; self.output_size];
        let mut total_error = 0.0;
        for i in 0..self.output_size {
            output_errors[i] = targets[i] - output[i];
            total_error += output_errors[i].powi(2);
        }
        total_error /= 2.0;
        
        // Calculate output layer gradients
        let mut output_gradients = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            output_gradients[i] = output_errors[i] * Self::sigmoid_derivative(output[i]);
        }
        
        // Calculate hidden layer errors
        let mut hidden_errors = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut error = 0.0;
            for j in 0..self.output_size {
                error += output_gradients[j] * self.weights_hidden_output[i][j];
            }
            hidden_errors[i] = error;
        }
        
        // Calculate hidden layer gradients
        let mut hidden_gradients = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            hidden_gradients[i] = hidden_errors[i] * Self::sigmoid_derivative(hidden[i]);
        }
        
        // Update weights and biases
        // Update weights between hidden and output layers
        for i in 0..self.hidden_size {
            for j in 0..self.output_size {
                self.weights_hidden_output[i][j] += self.learning_rate * output_gradients[j] * hidden[i];
            }
        }
        
        // Update output biases
        for i in 0..self.output_size {
            self.bias_output[i] += self.learning_rate * output_gradients[i];
        }
        
        // Update weights between input and hidden layers
        for i in 0..self.input_size {
            for j in 0..self.hidden_size {
                self.weights_input_hidden[i][j] += self.learning_rate * hidden_gradients[j] * inputs[i];
            }
        }
        
        // Update hidden biases
        for i in 0..self.hidden_size {
            self.bias_hidden[i] += self.learning_rate * hidden_gradients[i];
        }
        
        total_error
    }
    
    /// Make a prediction using the trained network
    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let (_, output) = self.forward(inputs);
        output
    }
    
    /// Get network architecture information
    pub fn info(&self) -> String {
        format!(
            "Neural Network: {} inputs -> {} hidden -> {} outputs (learning rate: {})",
            self.input_size, self.hidden_size, self.output_size, self.learning_rate
        )
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
}