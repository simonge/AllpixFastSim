
global_layers_list = {}

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


#from caloGraphNN import *
#from caloGraphNN_keras import GlobalExchange
#global_layers_list['GlobalExchange']=GlobalExchange  

class RaggedMeanPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RaggedMeanPooling, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute the mean across the feature dimension (axis=2)
        mean = tf.reduce_mean(inputs, axis=1)
        return mean
    
class RaggedMinPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RaggedMinPooling, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute the mean across the feature dimension (axis=2)
        mean = tf.reduce_min(inputs, axis=1)
        return mean
    
class RaggedMaxPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RaggedMaxPooling, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute the mean across the feature dimension (axis=2)
        mean = tf.reduce_max(inputs, axis=1)
        return mean
    
# Define custom layers here and add them to the global_layers_list dict (important!)
class RaggedGlobalAverage(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(RaggedGlobalAverage, self).__init__(**kwargs)
        

    def build(self, input_shape):
        # tf.ragged FIXME?
        self.num_vertices = input_shape[1]
        super(RaggedGlobalAverage, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1)
        mean_expanded = tf.expand_dims(mean, axis=1)
        zeros_like_x = tf.zeros_like(x)
        broadcasted_mean = zeros_like_x + mean_expanded
    
        # repeat_counts = x.row_lengths()
        # expanded_mean_flat = tf.repeat(mean, repeats=repeat_counts, axis=0)
        # expanded_mean = tf.RaggedTensor.from_row_lengths(expanded_mean_flat, row_lengths=repeat_counts)
        # result =  broadcasted_mean
        # tf.ragged FIXME?
        # maybe just use tf.shape(x)[1] instead?
        # rl     = x.row_lengths()
        # max_rl = tf.math.reduce_max(rl)
        # mean_t = tf.repeat(mean, repeats=max_rl, axis=1)
        # msk    = tf.sequence_mask(rl)
        # mean_r = tf.ragged.boolean_mask(mean_t, msk)
        
        #mean = tf.tile(mean, [1, self.num_vertices, 1])
        # result = tf.concat([x, mean_r], axis=-1)
        result = tf.concat([x, broadcasted_mean], axis=-1)
        
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], input_shape[2],)
    
# Define custom layers here and add them to the global_layers_list dict (important!)
class RaggedBatchNormalization(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(RaggedBatchNormalization, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(RaggedBatchNormalization, self).build(input_shape)

    def call(self, x):

        mean, variance = tf.nn.moments(x.flat_values,axes=[0,1])
        normalizedx    = tf.nn.batch_normalization(x, mean, variance,None,None,0.00000001) #Replace with eta
        
        return normalizedx

    def compute_output_shape(self, input_shape):
        return input_shape[:2]

class Conv2DGlobalExchange(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(Conv2DGlobalExchange, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]+input_shape[3])
    
    def call(self, inputs):
        average = tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
        average = tf.tile(average, [1,tf.shape(inputs)[1],tf.shape(inputs)[2],1])
        return tf.concat([inputs,average],axis=-1)
        
    
    def get_config(self):
        base_config = super(Conv2DGlobalExchange, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['Conv2DGlobalExchange']=Conv2DGlobalExchange 

class GaussActivation(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(GaussActivation, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.exp(- inputs**2 )
        
    
    def get_config(self):
        base_config = super(GaussActivation, self).get_config()
        return dict(list(base_config.items()) )
        

global_layers_list['GaussActivation']=GaussActivation 


class GravNet_ragged(tf.keras.layers.Layer):
    def __init__(self, 
                 n_neighbours, 
                 n_dimensions, 
                 n_filters, 
                 n_propagate,
                 subname,**kwargs):
        super(GravNet_ragged, self).__init__(**kwargs)
        
        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        
        #for naming
        self.subname = subname
        #name = self.name+subname
        
        self.input_feature_transform  = tf.keras.layers.Dense(n_propagate,name = subname+'_FLR')
        self.input_spatial_transform  = tf.keras.layers.Dense(n_dimensions,name = subname+'_S')
        self.output_feature_transform = tf.keras.layers.Dense(n_filters, activation='tanh', name = subname+'_Fout')

    def build(self, input_shape):
        
        self.input_feature_transform.build(input_shape)
        self.input_spatial_transform.build(input_shape)
        
        self.output_feature_transform.build((input_shape[0], input_shape[1], 
                                             input_shape[2] + self.input_feature_transform.units * 2))
 
        super(GravNet_ragged, self).build(input_shape)
        
    def call(self, x):
      
        coordinates = self.input_spatial_transform(x)
        features    = self.input_feature_transform(x)

        radius = coordinates[:,:,0]
        coords = coordinates[:,:,1:]
                    
        collected_neighbours = self.collect_neighbours_k(coordinates, features)
        #collected_neighbours = tf.map_fn(self.collect_neighbours_range,[radius,coords,features],fn_output_signature=tf.RaggedTensorSpec(shape=[None,2*self.n_propagate],dtype=tf.float32,ragged_rank=0),name=self.subname+'rank_k')
        #collected_neighbours = self.collect_neighbours_range2(coordinates,features)
         
        updated_features = tf.concat([x, collected_neighbours], axis=-1,name=self.subname+'update_feat')
        #tf.print(updated_features)
        return self.output_feature_transform(updated_features)
    

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)


    def collect_neighbours_range(self, coordinates):
        radius             = coordinates[0]
        
        coordsA            = tf.expand_dims(coordinates[1],0)
        coordsB            = tf.expand_dims(coordinates[1],1)
        dist_components    = (coordsA-coordsB)**2
        distance_matrix    = tf.reduce_sum(dist_components,axis=-1,name=self.subname+'reduce_sum1')

        weight_matrix      = tf.nn.relu(1-distance_matrix/radius)
        #weight_matrix      = tf.where(distance_matrix==0.,0.,tf.nn.relu(1-distance_matrix))
        #weight_matrix      = tf.linalg.band_part(weight_matrix, -1, -1)

        pass_boolean       = weight_matrix>0.0
        pass_cut           = tf.where(pass_boolean)
        passTensor         = tf.RaggedTensor.from_value_rowids(pass_cut[:,1],pass_cut[:,0])

        features           = coordinates[2]

        neighbours_features = tf.gather(features,passTensor,axis=0)
        gathered_weights    = tf.expand_dims(tf.gather(weight_matrix,passTensor,batch_dims=1),-1)
        gathered_weights    = tf.where(gathered_weights==1,0.0,gathered_weights)
        weighted_features   = neighbours_features*gathered_weights

        neighbours_max  = tf.reduce_max(weighted_features,  axis=-2, name=self.subname+'feat_max')
        neighbours_mean = tf.reduce_mean(weighted_features, axis=-2, name=self.subname+'feat_mean')

        return tf.concat([neighbours_max, neighbours_mean], axis=-1, name=self.subname+'feat_concat')

    def collect_neighbours_range2(self, coordinates,features): 
        coordsA            = tf.expand_dims(coordinates,1)
        coordsB            = tf.expand_dims(coordinates,2)
        dist_components    = (coordsA-coordsB)**2
        distance_matrix    = tf.reduce_sum(dist_components,axis=-1,name=self.subname+'reduce_sum1')
        
        weight_matrix      = tf.where(distance_matrix==0.,0.,tf.nn.relu(1-(1-distance_matrix)**3))
        weight_matrix      = tf.expand_dims(weight_matrix,-1)
    
        features =  tf.expand_dims(features,-2)
        weighted_features   = features*weight_matrix
               
        
        neighbours_max  = tf.reduce_max(weighted_features,  axis=-2, name=self.subname+'feat_max')
        neighbours_mean = tf.reduce_mean(weighted_features, axis=-2, name=self.subname+'feat_mean')

        
        return tf.concat([neighbours_max, neighbours_mean], axis=-1, name=self.subname+'feat_concat')
            
    def neighbours_indices(self, coordinates): 
        coordsA         = tf.expand_dims(coordinates,0,name=self.subname+'expand_coords1')
        coordsB         = tf.expand_dims(coordinates,1,name=self.subname+'expand_coords1')
        distances       = (coordsA-coordsB)**2
        distance_matrix = tf.reduce_sum(distances,axis=-1,name=self.subname+'reduce_sum1')
        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours,name=self.subname+'rank_k')

        stacked = tf.stack([-ranked_distances[:,1:], tf.cast(ranked_indices[:,1:],tf.float32)],axis=1,name=self.subname+'stack_dist_index')

        return stacked

    def collect_neighbours_k(self, coordinates, features):
        
        #neighbour_indices = tf.map_fn(self.neighbours_indeces_range,coordinates,fn_output_signature=tf.RaggedTensorSpec(shape=[None,None],dtype=tf.float32,ragged_rank=1))
        
        ranks = tf.map_fn(self.neighbours_indices,coordinates,fn_output_signature=tf.RaggedTensorSpec(shape=[None,2,self.n_neighbours-1],dtype=tf.float32,ragged_rank=0),name=self.subname+'rank_k')
        
        distance          = ranks[:,:,0]
        neighbour_indices = tf.cast(ranks[:,:,1],tf.int32,name=self.subname+'cast_indices')
               
        ################
        weights = tf.math.exp(-distance,name=self.subname+'weight_dist')
        weights = tf.expand_dims(weights, axis=-1,name=self.subname+'expand_weight')
        ################
        

        neighbour_features = tf.gather(features, neighbour_indices,batch_dims=1,name=self.subname+'gather_feat')
        #tf.print("neighbour_features")
        #tf.print(tf.reduce_any(tf.raw_ops.IsNan(x=neighbour_features)))
        #neighbour_features = tf.gather_nd(features, neighbour_indices)
               
        neighbour_features *= weights
        
        neighbours_max  = tf.reduce_max(neighbour_features,  axis=2, name=self.subname+'feat_max')
        neighbours_mean = tf.reduce_mean(neighbour_features, axis=2, name=self.subname+'feat_mean')

        return tf.concat([neighbours_max, neighbours_mean], axis=-1, name=self.subname+'feat_concat')
    
    def get_config(self):
        config = {'n_neighbours': self.n_neighbours, 
                  'n_dimensions': self.n_dimensions, 
                  'n_filters': self.n_filters, 
                  'n_propagate': self.n_propagate,
                  'subname': self.subname}
        base_config = super(GravNet_ragged, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

global_layers_list['GravNet_ragged']=GravNet_ragged   


def gauss_of_lin(x):
    return tf.exp(-1.0*x)
