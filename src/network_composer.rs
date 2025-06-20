use crate::neural_network::NeuralNetwork;
use std::collections::HashMap;

/// Represents a connection between neural networks
#[derive(Debug, Clone)]
pub struct NetworkConnection {
    pub source_network: String,
    pub target_network: String,
    pub source_outputs: Vec<usize>, // Which outputs from source
    pub target_inputs: Vec<usize>,  // Which inputs to target
}

/// A composer that manages multiple neural networks and their connections
#[derive(Debug)]
pub struct NetworkComposer {
    networks: HashMap<String, NeuralNetwork>,
    connections: Vec<NetworkConnection>,
    execution_order: Vec<String>,
}

impl NetworkComposer {
    /// Create a new network composer
    pub fn new() -> Self {
        Self {
            networks: HashMap::new(),
            connections: Vec::new(),
            execution_order: Vec::new(),
        }
    }

    /// Add a neural network to the composer
    pub fn add_network(&mut self, name: String, network: NeuralNetwork) -> Result<(), String> {
        if self.networks.contains_key(&name) {
            return Err(format!("Network '{}' already exists", name));
        }

        self.networks.insert(name.clone(), network);
        self.update_execution_order();
        Ok(())
    }

    /// Remove a neural network from the composer
    pub fn remove_network(&mut self, name: &str) -> Result<NeuralNetwork, String> {
        // Remove all connections involving this network
        self.connections
            .retain(|conn| conn.source_network != name && conn.target_network != name);

        self.update_execution_order();

        self.networks
            .remove(name)
            .ok_or_else(|| format!("Network '{}' not found", name))
    }

    /// Connect outputs of one network to inputs of another
    pub fn connect_networks(
        &mut self,
        source_name: &str,
        target_name: &str,
        source_outputs: Vec<usize>,
        target_inputs: Vec<usize>,
    ) -> Result<(), String> {
        // Validate networks exist
        let source_net = self
            .networks
            .get(source_name)
            .ok_or_else(|| format!("Source network '{}' not found", source_name))?;
        let target_net = self
            .networks
            .get(target_name)
            .ok_or_else(|| format!("Target network '{}' not found", target_name))?;

        // Validate output indices
        let source_output_size = source_net.get_layers().last().unwrap();
        for &output_idx in &source_outputs {
            if output_idx >= *source_output_size {
                return Err(format!(
                    "Source output index {} out of range (max: {})",
                    output_idx,
                    source_output_size - 1
                ));
            }
        }

        // Validate input indices
        let target_input_size = target_net.get_layers().first().unwrap();
        for &input_idx in &target_inputs {
            if input_idx >= *target_input_size {
                return Err(format!(
                    "Target input index {} out of range (max: {})",
                    input_idx,
                    target_input_size - 1
                ));
            }
        }

        // Validate connection sizes match
        if source_outputs.len() != target_inputs.len() {
            return Err(format!(
                "Connection size mismatch: {} outputs -> {} inputs",
                source_outputs.len(),
                target_inputs.len()
            ));
        }

        // Check for cycles (simple check - could be more sophisticated)
        if self.would_create_cycle(source_name, target_name) {
            return Err("Connection would create a cycle".to_string());
        }

        // Add the connection
        let connection = NetworkConnection {
            source_network: source_name.to_string(),
            target_network: target_name.to_string(),
            source_outputs,
            target_inputs,
        };

        self.connections.push(connection);
        self.update_execution_order();
        Ok(())
    }

    /// Forward propagation through the entire network composition
    pub fn forward(
        &mut self,
        inputs: &HashMap<String, Vec<f64>>,
    ) -> Result<HashMap<String, Vec<f64>>, String> {
        let mut network_outputs: HashMap<String, Vec<f64>> = HashMap::new();

        // Initialize with external inputs
        for (network_name, input_values) in inputs {
            if !self.networks.contains_key(network_name) {
                return Err(format!("Input network '{}' not found", network_name));
            }
            network_outputs.insert(network_name.clone(), input_values.clone());
        }

        // Execute networks in topological order
        for network_name in &self.execution_order {
            let network = self.networks.get_mut(network_name).unwrap();

            // Prepare inputs for this network
            let mut network_inputs = vec![0.0; network.get_layers()[0]];

            // Use external inputs if provided
            if let Some(external_inputs) = inputs.get(network_name) {
                if external_inputs.len() != network_inputs.len() {
                    return Err(format!(
                        "Input size mismatch for network '{}': expected {}, got {}",
                        network_name,
                        network_inputs.len(),
                        external_inputs.len()
                    ));
                }
                network_inputs = external_inputs.clone();
            }

            // Apply connections from other networks
            for connection in &self.connections {
                if connection.target_network == *network_name {
                    if let Some(source_outputs) = network_outputs.get(&connection.source_network) {
                        for (&source_idx, &target_idx) in connection
                            .source_outputs
                            .iter()
                            .zip(connection.target_inputs.iter())
                        {
                            if source_idx < source_outputs.len()
                                && target_idx < network_inputs.len()
                            {
                                network_inputs[target_idx] = source_outputs[source_idx];
                            }
                        }
                    }
                }
            }

            // Forward propagation through this network
            let outputs = network.predict(&network_inputs);
            network_outputs.insert(network_name.clone(), outputs);
        }

        Ok(network_outputs)
    }

    /// Train a specific network in the composition
    pub fn train_network(
        &mut self,
        network_name: &str,
        inputs: &[f64],
        targets: &[f64],
    ) -> Result<f64, String> {
        let network = self
            .networks
            .get_mut(network_name)
            .ok_or_else(|| format!("Network '{}' not found", network_name))?;

        Ok(network.train(inputs, targets))
    }

    /// Train the entire composition with end-to-end backpropagation
    pub fn train_composition(
        &mut self,
        inputs: &HashMap<String, Vec<f64>>,
        targets: &HashMap<String, Vec<f64>>,
    ) -> Result<f64, String> {
        // For now, implement simple individual network training
        // TODO: Implement true end-to-end backpropagation through connections
        let mut total_error = 0.0;
        let mut error_count = 0;

        for (network_name, target_values) in targets {
            if let Some(network_inputs) = inputs.get(network_name) {
                let error = self.train_network(network_name, network_inputs, target_values)?;
                total_error += error;
                error_count += 1;
            }
        }

        if error_count > 0 {
            Ok(total_error / error_count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Get a reference to a specific network
    pub fn get_network(&self, name: &str) -> Option<&NeuralNetwork> {
        self.networks.get(name)
    }

    /// Get a mutable reference to a specific network
    pub fn get_network_mut(&mut self, name: &str) -> Option<&mut NeuralNetwork> {
        self.networks.get_mut(name)
    }

    /// Get all network names
    pub fn get_network_names(&self) -> Vec<&String> {
        self.networks.keys().collect()
    }

    /// Get all connections
    pub fn get_connections(&self) -> &[NetworkConnection] {
        &self.connections
    }

    /// Get execution order
    pub fn get_execution_order(&self) -> &[String] {
        &self.execution_order
    }

    /// Get information about the composition
    pub fn info(&self) -> String {
        let mut info = format!(
            "Network Composition ({} networks, {} connections):\n",
            self.networks.len(),
            self.connections.len()
        );

        for network_name in &self.execution_order {
            if let Some(network) = self.networks.get(network_name) {
                info.push_str(&format!("  {}: {}\n", network_name, network.info()));
            }
        }

        if !self.connections.is_empty() {
            info.push_str("\nConnections:\n");
            for connection in &self.connections {
                info.push_str(&format!(
                    "  {} -> {}: outputs{:?} -> inputs{:?}\n",
                    connection.source_network,
                    connection.target_network,
                    connection.source_outputs,
                    connection.target_inputs
                ));
            }
        }

        info
    }

    /// Simple cycle detection (could be improved with proper graph algorithms)
    fn would_create_cycle(&self, source: &str, target: &str) -> bool {
        // Simple check: if target can reach source through existing connections,
        // adding source->target would create a cycle
        self.can_reach(target, source)
    }

    /// Check if one network can reach another through connections
    fn can_reach(&self, from: &str, to: &str) -> bool {
        if from == to {
            return true;
        }

        for connection in &self.connections {
            if connection.source_network == from
                && (connection.target_network == to
                    || self.can_reach(&connection.target_network, to))
            {
                return true;
            }
        }

        false
    }

    /// Update execution order based on connections (topological sort)
    fn update_execution_order(&mut self) {
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut temp_visited = std::collections::HashSet::new();

        for network_name in self.networks.keys() {
            if !visited.contains(network_name) {
                self.topological_sort(network_name, &mut visited, &mut temp_visited, &mut order);
            }
        }

        self.execution_order = order;
    }

    /// Topological sort helper for determining execution order
    fn topological_sort(
        &self,
        network_name: &str,
        visited: &mut std::collections::HashSet<String>,
        temp_visited: &mut std::collections::HashSet<String>,
        order: &mut Vec<String>,
    ) {
        if temp_visited.contains(network_name) {
            // Cycle detected - for now, just continue
            return;
        }

        if visited.contains(network_name) {
            return;
        }

        temp_visited.insert(network_name.to_string());

        // Visit all networks that this network depends on (sources)
        for connection in &self.connections {
            if connection.target_network == network_name {
                self.topological_sort(&connection.source_network, visited, temp_visited, order);
            }
        }

        temp_visited.remove(network_name);
        visited.insert(network_name.to_string());
        order.push(network_name.to_string());
    }
}

impl Default for NetworkComposer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composer_creation() {
        let composer = NetworkComposer::new();
        assert_eq!(composer.get_network_names().len(), 0);
        assert_eq!(composer.get_connections().len(), 0);
    }

    #[test]
    fn test_add_remove_networks() {
        let mut composer = NetworkComposer::new();
        let nn1 = NeuralNetwork::new(2, 3, 1, 0.1);
        let nn2 = NeuralNetwork::new(1, 2, 1, 0.1);

        // Add networks
        assert!(composer.add_network("net1".to_string(), nn1).is_ok());
        assert!(composer.add_network("net2".to_string(), nn2).is_ok());
        assert_eq!(composer.get_network_names().len(), 2);

        // Try to add duplicate
        let nn3 = NeuralNetwork::new(1, 1, 1, 0.1);
        assert!(composer.add_network("net1".to_string(), nn3).is_err());

        // Remove network
        assert!(composer.remove_network("net1").is_ok());
        assert_eq!(composer.get_network_names().len(), 1);
        assert!(composer.remove_network("nonexistent").is_err());
    }

    #[test]
    fn test_network_connections() {
        let mut composer = NetworkComposer::new();
        let nn1 = NeuralNetwork::new(2, 3, 2, 0.1); // 2 outputs
        let nn2 = NeuralNetwork::new(2, 3, 1, 0.1); // 2 inputs

        composer.add_network("source".to_string(), nn1).unwrap();
        composer.add_network("target".to_string(), nn2).unwrap();

        // Valid connection
        assert!(
            composer
                .connect_networks("source", "target", vec![0, 1], vec![0, 1])
                .is_ok()
        );
        assert_eq!(composer.get_connections().len(), 1);

        // Invalid connections
        assert!(
            composer
                .connect_networks("source", "target", vec![2], vec![0])
                .is_err()
        ); // Out of range output
        assert!(
            composer
                .connect_networks("source", "target", vec![0], vec![2])
                .is_err()
        ); // Out of range input
        assert!(
            composer
                .connect_networks("source", "target", vec![0, 1], vec![0])
                .is_err()
        ); // Size mismatch
    }

    #[test]
    fn test_forward_propagation() {
        let mut composer = NetworkComposer::new();
        let nn1 = NeuralNetwork::new(2, 3, 1, 0.1);
        let nn2 = NeuralNetwork::new(1, 2, 1, 0.1);

        composer.add_network("net1".to_string(), nn1).unwrap();
        composer.add_network("net2".to_string(), nn2).unwrap();
        composer
            .connect_networks("net1", "net2", vec![0], vec![0])
            .unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("net1".to_string(), vec![0.5, 0.8]);

        let outputs = composer.forward(&inputs).unwrap();
        assert!(outputs.contains_key("net1"));
        assert!(outputs.contains_key("net2"));
        assert_eq!(outputs["net1"].len(), 1);
        assert_eq!(outputs["net2"].len(), 1);
    }
}
