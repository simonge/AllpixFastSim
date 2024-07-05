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
    def __init__(self, latent_dim=50, nconditions=5, grid_size=9 ):
        super(VAE, self).__init__()
        self.flat_shape  = grid_size*grid_size*2
        self.latent_dim  = latent_dim
        self.nconditions = nconditions        
        self.nSamples    = 1000

        self.encoder     = Encoder(nconditions, latent_dim, self.flat_shape)
        self.decoder     = Decoder(nconditions, latent_dim, grid_size)
        self.adversarial = Adversarial(latent_dim, nconditions)
    
    # Function to compile the model
    def compile(self, r_optimizer, a_optimizer):
        super().compile()
        self.optimizer_encoder_decoder = r_optimizer
        self.optimizer_adversarial     = a_optimizer

    # Function to reparameterize the latent space
    def reparameterize(self, mean, logvar, nSamples=1000):
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
# Define the Preprocessor
######################################################
class Preprocessor(tf.keras.Model):
    def __init__(self, isSquare=True):
        super(Preprocessor, self).__init__()
        self.conditions_normalizer = tfkl.Normalization(name='conditions_normalizer')
        self.image_normalizer      = tfkl.Normalization(name='image_normalizer')
        self.isSquare = isSquare

    def adapt(self, conditions, image):
        # Transform conditions and image before adapting
        transformed_conditions, transformed_image = self.transform_input(conditions, image)
        # Normalise the transformed conditions and image
        self.conditions_normalizer.adapt(transformed_conditions)        
        self.image_normalizer.adapt(transformed_image)

    def determine_transformations(self, conditions, isSquare):
        def make_transformation(position):
            # Initialize transformation parameters
            horizontal_transform = 1.0
            vertical_transform   = 1.0
            transpose_image      = False
            
            # Determine transformations based on conditions
            if position[0] < 0:
                horizontal_transform = -1.0
            if position[1] < 0:
                vertical_transform = -1.0
            if isSquare and abs(position[0]) < abs(position[1]):
                transpose_image = True
            
            return horizontal_transform, vertical_transform, transpose_image 
        
        positions = conditions[:,0:2]
        transformations = tf.map_fn(make_transformation, positions, fn_output_signature=(tf.float32, tf.float32, tf.bool)) 
        return transformations
    
    # Function to transform the position and momentum conditions
    def transform_conditions(self, input):
        conditions, transformations = input
        # Transform the conditions
        flip_horizontal, flip_vertical, transpose_image = transformations

        # Position transformation
        transformed_x = conditions[0] * flip_horizontal
        transformed_y = conditions[1] * flip_vertical
        # Momentum transformation
        transformed_px = conditions[2] * flip_horizontal
        transformed_py = conditions[3] * flip_vertical

        # If the image is transposed, swap the x and y components
        if transpose_image:
            transformed_x, transformed_y = transformed_y, transformed_x
            transformed_px, transformed_py = transformed_py, transformed_px
        
        return tf.stack([transformed_x, transformed_y, transformed_px, transformed_py, conditions[4]], axis=0)

    # Function to transform the image
    def transform_image(self, input):
        image, transformations = input
        # Transform the image
        transformed_image = image
        flip_horizontal, flip_vertical, transpose_image = transformations

        # If the image is flipped horizontally, reverse the order of the columns
        if flip_horizontal < 0:
            transformed_image = tf.reverse(transformed_image, axis=[1])
        # If the image is flipped vertically, reverse the order of the rows
        if flip_vertical < 0:
            transformed_image = tf.reverse(transformed_image, axis=[0])
        # If the image is transposed, swap the x and y components
        if transpose_image:
            transformed_image = tf.transpose(transformed_image, perm=[1, 0, 2])

        return transformed_image

    # Function to transform the input grid and conditions into 1/8th of a square or 1/4 of a rectangle
    def transform_input(self, conditions, image):
        
        transformations = self.determine_transformations(conditions, self.isSquare)
                    
        # Transform the conditions
        transformed_conditions = tf.map_fn(self.transform_conditions, (conditions,transformations), fn_output_signature=tf.float32)
       
        # Transform the image
        transformed_image = tf.map_fn(self.transform_image, (image,transformations), fn_output_signature=tf.float32)

        return transformed_conditions, transformed_image

    def call(self, input):
        conditions, image = input
        transformed_conditions, transformed_image = self.transform_input(conditions, image)        
        self.adapt(transformed_conditions, transformed_image)
        normalized_conditions = self.conditions_normalizer(transformed_conditions)
        normalized_image      = self.image_normalizer(transformed_image)
        
        return normalized_conditions, normalized_image


######################################################
# Define the Generator
######################################################
class Generator(tf.keras.Model):
    def __init__(self, original_model):
        super(Generator, self).__init__()
        # self.conditions_encoder = original_model.conditions_encoder
        self.decoder    = original_model.decoder
        self.latent_dim = original_model.latent_dim
        self.nconditions = original_model.nconditions
        self.output_names = ['output']

    def call(self, conditions):
        # Concatenate conditions and z along the last axis
        z = tf.random.normal(shape=(tf.shape(conditions)[0], self.latent_dim))
        #input = tf.concat([conditions,z], axis=-1)
        #concat = self.concatLatentSpace(conditions,z)
        reconstructed = self.decoder(conditions, z)
        return reconstructed
    
######################################################
# Define latent space encoder
######################################################
class LatentSpace(tf.keras.Model):
    def __init__(self, original_model):
        super(LatentSpace, self).__init__()
        self.flat_shape  = original_model.flat_shape
        self.nconditions = original_model.nconditions
        self.input_layer = tfkl.InputLayer(shape=(self.flat_shape+self.nconditions,))
        self.encoder    = original_model.encoder
        self.output_names = ['output']

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean
    
    def call(self, input):
        mean, logvar = self.encoder(input)
        z = self.reparameterize(mean, logvar)
        return z