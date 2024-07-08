import tensorflow as tf
#import tensorflow_probability as tfp

tfkl = tf.keras.layers
#tfpl = tfp.layers
#tfd = tfp.distributions

        
######################################################
# Define the Encoder
######################################################
class Encoder(tf.keras.Model):
    def __init__(self, nconditions, latent_dim, flat_shape=200):
        super(Encoder, self).__init__()
        self.encode = tf.keras.Sequential([
            tfkl.InputLayer(shape=(flat_shape+nconditions,)),            
            tfkl.Dense(1024, activation='relu'),
            tfkl.Dense(1024, activation='relu'),
            tfkl.Dense(256, activation='relu'),
            tfkl.Dense(256, activation='relu'),
            tfkl.Dense(2*latent_dim),
        ])

    def call(self, inputs):
        encoded = self.encode(inputs)
        mean, logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
        return mean, logvar
    
######################################################
# Define the Decoder
######################################################
class Decoder(tf.keras.Model):
    def __init__(self, nconditions, latent_dim, grid_size=9):
        super(Decoder, self).__init__()
        self.flat_shape = 2*grid_size*grid_size
        self.grid_size  = grid_size
        self.decode = tf.keras.Sequential([            
            tfkl.InputLayer(shape=(latent_dim+nconditions,),name='input_layer'),
            tfkl.Dense(256, activation='relu'),
            tfkl.Dense(256, activation='relu'),
            tfkl.Dense(1024, activation='relu'),
            tfkl.Dense(1024, activation='relu'),
            tfkl.Dense(self.flat_shape, name='output_layer')
        ])

    def call(self, normalized_conditions, z):
        inputs = tf.concat([normalized_conditions, z], axis=-1)
        outputs = self.decode(inputs)
        reshaped_outputs = tf.reshape(outputs,(tf.shape(outputs)[0], self.grid_size, self.grid_size, 2))
        return reshaped_outputs
    
######################################################
# Define the Discriminator
######################################################
class Adversarial(tf.keras.Model):
    def __init__(self, latent_dim, nconditions):
        super(Adversarial, self).__init__()
        self.reconstruct_conditions = tf.keras.Sequential([
            tfkl.InputLayer(shape=(latent_dim,)),
            tfkl.Dense(1024, activation='relu'),
            tfkl.Dense(256, activation='relu'),
            tfkl.Dense(64, activation='relu'),
            tfkl.Dense(16, activation='relu'),
            tfkl.Dense(nconditions, activation='linear')
        ])

    def call(self, inputs):
        x = self.reconstruct_conditions(inputs)
        return x

######################################################
# Define the Conditional, Adviserial Variational Autoencoder
######################################################
class VAE(tf.keras.Model):
    def __init__(self, latent_dim=50, nconditions=5, grid_size=9, nSamples=100 ):
        super(VAE, self).__init__()
        self.flat_shape  = grid_size*grid_size*2
        self.latent_dim  = latent_dim
        self.nconditions = nconditions        
        self.nSamples    = nSamples

        self.encoder     = Encoder(nconditions, latent_dim, self.flat_shape)
        self.decoder     = Decoder(nconditions, latent_dim, grid_size)
        self.adversarial = Adversarial(latent_dim, nconditions)
    
    # Function to compile the model
    def compile(self, r_optimizer, a_optimizer):
        super().compile()
        self.optimizer_encoder_decoder = r_optimizer
        self.optimizer_adversarial     = a_optimizer

    # Function to reparameterize the latent space
    def reparameterize(self, mean, logvar):
        #eps = tf.random.normal(shape=(self.nSamples, tf.shape(mean)[0], tf.shape(mean)[1]))
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean
    
    
    def train_step(self, input):
        conditions, image = input  # Unpack the data from the input tuple

        # Flatten the image other than the batch dimension and concatenate with the conditions
        data = tf.concat([conditions, tf.reshape(image, (tf.shape(image)[0], -1))], axis=-1)

        with tf.GradientTape() as tape:
            # Forward pass
            mean, logvar = self.encoder(data)
            z = self.reparameterize(mean, logvar)
            reconstruction = self.decoder(conditions, z)            
            adversarial_predicted_conditions = self.adversarial(z)

            # Compute losses
            reconstruction_loss = tf.reduce_sum(tf.math.squared_difference(image, reconstruction), axis=[-1,-2,-3])
            kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
            adversarial_lossA = tf.reduce_sum(tf.math.squared_difference(conditions, adversarial_predicted_conditions), axis=-1)

            # Compute total loss
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss - adversarial_lossA)

        # Compute gradients with respect to the weights of the encoder and decoder
        grads_encoder_decoder = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)

        # Apply gradients
        self.optimizer_encoder_decoder.apply_gradients(zip(grads_encoder_decoder, self.encoder.trainable_variables + self.decoder.trainable_variables))

        with tf.GradientTape() as tape:
            # Compute adversarial loss
            adversarial_conditions = self.adversarial(z)
            adversarial_loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(conditions, adversarial_conditions), axis=-1))

        # Compute gradients with respect to the weights of the adversarial network
        grads_adversarial = tape.gradient(adversarial_loss, self.adversarial.trainable_variables)

        # Apply gradients
        self.optimizer_adversarial.apply_gradients(zip(grads_adversarial, self.adversarial.trainable_variables))

        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss, "adversarial_loss": adversarial_loss}
    
    def test_step(self, input):
        print("test_step")
        conditions, image = input  # Unpack the data from the input tuple
        data = tf.concat([conditions, tf.reshape(image, (tf.shape(image)[0], -1))], axis=-1)
        mean, logvar = self.encoder(data)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(conditions,z)
        advisarial_predicted_conditions = self.adversarial(z)

        # Reconstruction loss
        reconstruction_loss = tf.reduce_sum(tf.math.squared_difference(image, reconstruction), axis=[-1,-2,-3])
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
        adversarial_loss = tf.reduce_sum(tf.math.squared_difference(conditions, advisarial_predicted_conditions), axis=-1)
        
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss - adversarial_loss)
        
        # Return a dictionary containing the loss and the metrics
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss, "adversarial_loss": adversarial_loss}
        #return reconstruction

######################################################
# Define denomalisation layer inverting a tfkl nomorlisation layer
######################################################
class Denormalization(tf.keras.layers.Layer):
    def __init__(self, normalizer, name=None):
        super(Denormalization, self).__init__(name=name)
        self.normalizer = normalizer

    def call(self, inputs):
        mean = self.normalizer.mean
        variance = self.normalizer.variance
        return inputs * tf.sqrt(variance) + mean

######################################################
# Define the Preprocessor
######################################################
class Preprocessor(tf.keras.Model):
    def __init__(self, isSquare=True):
        super(Preprocessor, self).__init__()
        self.conditions_normalizer = tfkl.Normalization(name='conditions_normalizer')
        self.image_normalizer      = tfkl.Normalization(name='image_normalizer')
        self.conditions_denormalizer = Denormalization(self.conditions_normalizer, name='conditions_denormalizer')
        self.image_denormalizer      = Denormalization(self.image_normalizer, name='image_denormalizer')
        self.isSquare = isSquare

    def adapt(self, conditions, image):
        # Transform conditions and image before adapting
        transformed_conditions, transformed_image = self.transform_input(conditions, image)
        # Normalise the transformed conditions and image
        self.conditions_normalizer.adapt(transformed_conditions)        
        self.image_normalizer.adapt(transformed_image)
    
    # Function to transform the input grid and conditions into 1/8th of a square or 1/4 of a rectangle
    def transform_input(self, conditions, image):

        # Determine the transformations    
        flip_horizontal = tf.expand_dims(conditions[:,0] < 0, axis=-1)
        flip_vertical   = tf.expand_dims(conditions[:,1] < 0, axis=-1)
        transpose_image = tf.expand_dims(tf.math.abs(conditions[:,0]) < tf.math.abs(conditions[:,1]), axis=-1)
        
        # Use tf.gather to select indices 0 and 2 from conditions tensor
        x_conditions_indices = tf.gather(conditions, [0, 2], axis=1)
        y_conditions_indices = tf.gather(conditions, [1, 3], axis=1)

        # Negate the 0 and 2 components of the conditions when filp_horizontal is True
        x_conditions = tf.where(flip_horizontal, -x_conditions_indices, x_conditions_indices)
        # Negate the 1 and 3 components of the conditions when filp_vertical is True
        y_conditions = tf.where(flip_vertical, -y_conditions_indices, y_conditions_indices)
        # Transpose 0,1 and 2,3 indices of the conditions when transpose_image is True
        transformed_conditions = tf.where(transpose_image, tf.stack([y_conditions[:,0], x_conditions[:,0], y_conditions[:,1], x_conditions[:,1], conditions[:,4]], axis=1), tf.concat([x_conditions, y_conditions, conditions[:,4:]], axis=1))
       
        flip_horizontal_image = flip_horizontal[:, None, None]
        flip_vertical_image   = flip_vertical[:, None, None]
        transpose_image_image = transpose_image[:, None, None]

        # Transform the image
        transformed_image = tf.where(flip_horizontal_image, tf.reverse(image, axis=[2]), image)
        transformed_image = tf.where(flip_vertical_image,   tf.reverse(transformed_image, axis=[1]), transformed_image)
        transformed_image = tf.where(transpose_image_image, tf.transpose(image, perm=[0, 2, 1, 3]), transformed_image)

        return transformed_conditions, transformed_image
    
    def transform_output(self, conditions, image):
        # Determine the transformations    
        flip_horizontal = conditions[:,0] < 0
        flip_vertical   = conditions[:,1] < 0
        transpose_image = tf.math.abs(conditions[:,0]) < tf.math.abs(conditions[:,1])
        
        flip_horizontal_image = flip_horizontal[:, None, None, None]
        flip_vertical_image   = flip_vertical[:, None, None, None]
        transpose_image_image = transpose_image[:, None, None, None]

        # Transform the image
        transformed_image = tf.where(flip_horizontal_image, tf.reverse(image, axis=[2]), image)
        transformed_image = tf.where(flip_vertical_image,   tf.reverse(transformed_image, axis=[1]), transformed_image)
        if self.isSquare:
            transformed_image = tf.where(transpose_image_image, tf.transpose(image, perm=[0, 2, 1, 3]), transformed_image)
        
        return transformed_image
    
    def transform_conditions(self, conditions):
        # Determine the transformations    
        # Determine the transformations    
        flip_horizontal = tf.expand_dims(conditions[:,0] < 0, axis=-1)
        flip_vertical   = tf.expand_dims(conditions[:,1] < 0, axis=-1)
        transpose_image = tf.expand_dims(tf.math.abs(conditions[:,0]) < tf.math.abs(conditions[:,1]), axis=-1)
        
        
        # Use tf.gather to select indices 0 and 2 from conditions tensor
        x_conditions_indices = tf.gather(conditions, [0, 2], axis=1)
        y_conditions_indices = tf.gather(conditions, [1, 3], axis=1)

        # Negate the 0 and 2 components of the conditions when filp_horizontal is True
        x_conditions = tf.where(flip_horizontal, -x_conditions_indices, x_conditions_indices)
        # Negate the 1 and 3 components of the conditions when filp_vertical is True
        y_conditions = tf.where(flip_vertical, -y_conditions_indices, y_conditions_indices)
        # Transpose 0,1 and 2,3 indices of the conditions when transpose_image is True
        transformed_conditions = tf.where(transpose_image, tf.stack([y_conditions[:,0], x_conditions[:,0], y_conditions[:,1], x_conditions[:,1], conditions[:,4]], axis=1), tf.concat([x_conditions, y_conditions, conditions[:,4:]], axis=1))
        
        return transformed_conditions

    def call(self, input):
        conditions, image = input
        # Folding input into 1/8th of a square or 1/4 of a rectangle
        transformed_conditions, transformed_image = self.transform_input(conditions, image)        
        self.adapt(transformed_conditions, transformed_image)
        normalized_conditions = self.conditions_normalizer(transformed_conditions)
        normalized_image      = self.image_normalizer(transformed_image)
        
        return normalized_conditions, normalized_image


######################################################
# Define the Generator
######################################################
class Generator(tf.keras.Model):
    def __init__(self, vae_model, preprocessor):
        super(Generator, self).__init__()
        self.decoder     = vae_model.decoder
        self.latent_dim  = vae_model.latent_dim
        self.nconditions = vae_model.nconditions
        self.conditions_normalizer = preprocessor.conditions_normalizer
        self.image_denormalizer    = preprocessor.image_denormalizer
        self.transform_output      = preprocessor.transform_output
        self.transform_conditions  = preprocessor.transform_conditions
        self.output_names = ['output']

        # Pre and Post processor


    def call(self, conditions):
        # Concatenate conditions and z along the last axis
        z = tf.random.normal(shape=(tf.shape(conditions)[0], self.latent_dim))
        #input = tf.concat([conditions,z], axis=-1)
        #concat = self.concatLatentSpace(conditions,z)
        transformed_conditions    = self.transform_conditions(conditions)
        normalized_conditions     = self.conditions_normalizer(transformed_conditions)
        reconstructed             = self.decoder(normalized_conditions, z)
        denorm_reconstructed      = self.image_denormalizer(reconstructed)
        transformed_reconstructed = self.transform_output(conditions, denorm_reconstructed)
        return transformed_reconstructed
    
######################################################
# Define latent space encoder
######################################################
class LatentSpace(tf.keras.Model):
    def __init__(self, original_model, preprocessor):
        super(LatentSpace, self).__init__()
        self.flat_shape   = original_model.flat_shape
        self.nconditions  = original_model.nconditions
        self.input_layer  = tfkl.InputLayer(shape=(self.flat_shape+self.nconditions,))
        self.encoder      = original_model.encoder
        self.conditions_normalizer = preprocessor.conditions_normalizer
        self.image_normalizer = preprocessor.image_normalizer
        self.transform_input = preprocessor.transform_input
        self.output_names = ['output']

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean
    
    def call(self, input):
        conditions, image = input  # Unpack the data from the input tuple
        transformed_conditions, transformed_image = self.transform_input(conditions, image)
        normalized_conditions = self.conditions_normalizer(transformed_conditions)
        normalized_image      = self.image_normalizer(transformed_image)

        # Flatten the image other than the batch dimension and concatenate with the conditions
        data = tf.concat([normalized_conditions, tf.reshape(normalized_image, (tf.shape(normalized_image)[0], -1))], axis=-1)

        mean, logvar = self.encoder(data)
        z = self.reparameterize(mean, logvar)
        return z