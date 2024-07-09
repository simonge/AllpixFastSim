# Hyperparameters
node_feature_dim = 4  # For pixel x, pixel y, Time of Arrival, Time over Threshold
hidden_dim = 128
latent_dim = 64
output_dim = 4  # For pixel x, pixel y, Time of Arrival, Time over Threshold
condition_dim = 5  # For x, y, px, py, start_time
max_nodes = 100

# Model components
encoder = GraphEncoder(node_feature_dim, hidden_dim, latent_dim, condition_dim)
node_count_predictor = NodeCountPredictor(latent_dim)
decoder = Decoder(latent_dim, hidden_dim, output_dim, condition_dim, max_nodes)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 50

for epoch in range(epochs):
    for data in data_loader:
        with tf.GradientTape() as tape:
            # Encode
            z_mean, z_logvar = encoder(data.graph_tensor, data.conditions)
            std = tf.exp(0.5 * z_logvar)
            eps = tf.random.normal(shape=std.shape)
            z = z_mean + eps * std
            
            # Predict number of nodes
            num_nodes = tf.nn.relu(node_count_predictor(z))
            num_nodes = tf.minimum(tf.cast(num_nodes, tf.int32), max_nodes)
            
            # Decode
            reconstructed_nodes = decoder(z, data.conditions, num_nodes)
            
            # Compute loss
            loss = vae_loss(reconstructed_nodes, data.y[:num_nodes], z_mean, z_logvar)
        
        gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables + node_count_predictor.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables + node_count_predictor.trainable_variables))

    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')
