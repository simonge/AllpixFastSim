import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_gnn as tfgnn

class GraphEncoder(Model):
    def __init__(self, node_feature_dim, hidden_dim, latent_dim, condition_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = tfgnn.keras.layers.GraphConv(hidden_dim)
        self.conv2 = tfgnn.keras.layers.GraphConv(hidden_dim)
        self.fc1 = layers.Dense(latent_dim)
        self.fc2_mean = layers.Dense(latent_dim)
        self.fc2_logvar = layers.Dense(latent_dim)
        self.condition_dim = condition_dim

    def call(self, graph_tensor, conditions):
        node_features = graph_tensor.node_features['node']
        x = tf.nn.relu(self.conv1(node_features, graph_tensor.edge_sets['edges']))
        x = tf.nn.relu(self.conv2(x, graph_tensor.edge_sets['edges']))
        x = tf.reduce_mean(x, axis=0)
        x = tf.concat([x, conditions], axis=-1)
        h = tf.nn.relu(self.fc1(x))
        z_mean = self.fc2_mean(h)
        z_logvar = self.fc2_logvar(h)
        return z_mean, z_logvar
    
class NodeCountPredictor(Model):
    def __init__(self, latent_dim):
        super(NodeCountPredictor, self).__init__()
        self.fc = layers.Dense(1)

    def call(self, z):
        num_nodes = tf.nn.relu(self.fc(z))
        return num_nodes
    
class Decoder(Model):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim, max_nodes):
        super(Decoder, self).__init__()
        self.fc1 = layers.Dense(hidden_dim)
        self.fc2 = layers.Dense(hidden_dim)
        self.fc3 = layers.Dense(output_dim * max_nodes)
        self.output_dim = output_dim
        self.max_nodes = max_nodes

    def call(self, z, conditions, num_nodes):
        z = tf.concat([z, conditions], axis=-1)
        h = tf.nn.relu(self.fc1(z))
        h = tf.nn.relu(self.fc2(h))
        h = self.fc3(h)
        h = tf.reshape(h, (-1, self.max_nodes, self.output_dim))
        return h[:, :num_nodes, :]

def vae_loss(reconstructed_nodes, original_nodes, z_mean, z_logvar):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(original_nodes, reconstructed_nodes))
    kl_loss = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
    return reconstruction_loss + kl_loss

def generate_sample(conditions):
    z = tf.random.normal(shape=(1, latent_dim))
    num_nodes = tf.nn.relu(node_count_predictor(z))
    num_nodes = tf.minimum(tf.cast(num_nodes, tf.int32), max_nodes)
    generated_nodes = decoder(z, conditions, num_nodes)
    return generated_nodes

# Generate a new sample
conditions = tf.constant([x, y, px, py, start_time], shape=(1, condition_dim), dtype=tf.float32)
generated_nodes = generate_sample(conditions)
