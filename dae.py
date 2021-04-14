#################### Autoencoder Class ##################################
#import tensorflow.compat.v1 as tf
import os
import numpy as np
#tf.compat.v1.disable_v2_behavior()
import tensorflow as tf
class Autoencoder:
    """
    Autoencoder.
    """
    def __init__(self, ndims=253, hidden_width=[200,125, 75, 20, 75, 125, 200], nsamples=800):
        """Initializes a RAE.

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
            nsamples(int): Number of samples from each frame of MD simulation
        """
        tf.reset_default_graph()
        self._ndims = ndims
        self._hidden_width = hidden_width
        nlatent = hidden_width[3]
        self._nlatent = nlatent
        self._nsamples = nsamples

        # Create session
        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None ,ndims])
        self.y_placeholder = tf.placeholder(tf.float32,[None, ndims])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        # Build graph.
        self.z = self._encoder(self.x_placeholder, self.keep_prob)
        self.outputs_tensor = self._decoder(self.z, self.keep_prob)

        # Setup loss tensor, predict_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.outputs_tensor, self.y_placeholder)

        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)
        # Saving the model
        self.saver = tf.train.Saver(max_to_keep=None)
        
        # Load the model
        self.loader = tf.train

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
    def _encoder(self, x, keep_prob):
        """Encoder block of the network.

        Builds a three layer network of fully connected layers reducing
        input to tensor of (batch_size, latent) dimension:

        Input --> h1 --> h2 --> h3 --> _nlatent( layer_3)

        Args:
            x (tf.Tensor): The input tensor of dimension (None, _ndims).
        Returns:
            layer_3(tf.Tensor): The , tensor of dimension
                (None, _nlatent).
        """
        layer_1= tf.contrib.layers.fully_connected(x, self._hidden_width[0], activation_fn=tf.nn.tanh)
        layer_2 = tf.contrib.layers.fully_connected(layer_1, self._hidden_width[1], activation_fn=tf.nn.tanh)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        layer_3 = tf.contrib.layers.fully_connected(layer_2, self._hidden_width[2], activation_fn=tf.nn.tanh)
        layer_3 = tf.nn.dropout(layer_3, keep_prob)
        layer_4 = tf.contrib.layers.fully_connected(layer_3, self._nlatent, activation_fn=tf.nn.tanh)

        return layer_4

    def _decoder(self, z, keep_prob):
        """From a latent space, decode back into the mean quantities.

        Builds a three layer network of fully connected layers,
        with 50, 100, _ndims nodes.

        z (_nlatent) --> h4 -->  h5 --> h6 --> _ndims.

        Use activation of tf.nn.softplus for hidden layers.

        Args:
            z(tf.Tensor): z from _encoder with dimension (None, _nlatent).
        Returns:
            f(tf.Tensor): Decoded features, tensor of dimension (None, _ndims).
        """
        layer_1 = tf.contrib.layers.fully_connected(z, self._hidden_width[4], activation_fn=tf.nn.tanh)
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = tf.contrib.layers.fully_connected(layer_1, self._hidden_width[5], activation_fn=tf.nn.tanh)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        layer_3 = tf.contrib.layers.fully_connected(layer_2, self._hidden_width[6], activation_fn=tf.nn.tanh)
        f = tf.contrib.layers.fully_connected(layer_3, self._ndims, activation_fn=None)
        
        return f

    def _reconstruction_loss(self, f, x_gt):
        """Constructs the reconstruction loss.

        Args:
            f(tf.Tensor): Predicted score for each example, dimension (None,
                _ndims).
            x_gt(tf.Tensor): Ground truth for each example, dimension (None,
                _ndims).
        Returns:
            recon_loss(tf.Tensor): A scalar Tensor for dimension ()
                containing the reconstruction loss.
        """
        recon_loss =  tf.reduce_sum(tf.abs(tf.subtract(f, x_gt)), 1)

        return tf.reduce_mean(recon_loss)
    
    def _reg_loss(self):
        """Constructs the contractive loss.
         Not Implemented Yet!
        Args:
        Returns:
            recon_loss(tf.Tensor): A scalar Tensor for dimension ()
                containing the reconstruction loss.
        """
        L1 = 0.0#tf.reduce_sum([tf.reduce_sum(tf.abs(i)) for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        L2 = tf.reduce_sum([tf.reduce_sum(tf.square(i)) for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

        
        return tf.add(L1, L2)
    
    def loss(self, f, x_gt, lam=0.001):
        """Computes the total loss.

        Computes the sum of latent and reconstruction loss.
        For AE no latent_loss!

        Args:
            f (tf.Tensor): Decoded image for each example, dimension (None,
                _ndims).
            x_gt (tf.Tensor): Ground truth for each example, dimension (None,
                _ndims)
        Returns:
            total_loss: Tensor for dimension (). Sum of
                latent_loss and reconstruction loss.
                for AE no latent_loss
        """
        total_loss =  0.5*self._reconstruction_loss(f, x_gt) + lam*self._reg_loss() 
        return total_loss

    def update_op(self, loss, learning_rate):
        """Creates the update optimizer.

        Use tf.train.AdamOptimizer to obtain the update op.

        Args:
            loss(tf.Tensor): Tensor of shape () containing the loss function.
            learning_rate(tf.Tensor): Tensor of shape (). Learning rate for
                gradient descent.
        Returns:
            train_op(tf.Operation): Update opt tensorflow operation.
        """
        train_op = tf.train.AdamOptimizer(learning_rate,name='Adam').minimize(loss)
        return train_op

    def generate_samples(self, x):
        """Generates random samples from the provided z_np.

        Args:
            x(numpy.ndarray): Numpy array of dimension
                (batch_size, _ndims) i.e. the single MD frame.

        Returns:
            out(numpy.ndarray): The sampled images (numpy.ndarray) of
                dimension (batch_size, _ndims).
        """
        out = self.session.run(self.outputs_tensor, feed_dict={self.x_placeholder: x, self.keep_prob: 1.0})
        return np.array(out)

    def save(self, save_dir, step=0):
        self.saver.save(self.session, save_dir, step)

    def load(self,  load_dir,ct = 'checkpoint.ckpt-0.meta', step=0):
        os.chdir(load_dir)
        ct =os.path.join(load_dir, ct )
        saver = self.loader.import_meta_graph(ct)
        if step==0:
            saver.restore(self.session, self.loader.latest_checkpoint(load_dir))
        else:
            saver.restore(self.session, os.path.join(load_dir,'checkpoint.ckpt-'+str(step)))