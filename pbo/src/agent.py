# Generic imports
import os
import warnings

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow                    as     tf
import tf_keras                      as     tk
import tensorflow_probability        as     tfp
from   pbo.src.gpu                   import configure_gpus
from   tf_keras                      import Model
from   tf_keras.layers               import Dense
from   tf_keras.initializers         import LecunNormal

# Cap the GPU footprint before the first kernel initializes the devices
configure_gpus()

# Define alias
tk.backend.set_floatx('float64')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tfd = tfp.distributions

###############################################
### Neural network for mu and sg prediction
class nn(Model):
    def __init__(self, arch, dim, act, last, lr):
        super(nn, self).__init__()

        # Initialize network as empty list
        self.net = []

        # Define hidden layers
        for layer in range(len(arch)):
            self.net.append(Dense(arch[layer],
                                  kernel_initializer=LecunNormal(),
                                  activation=act,
                                  dtype='float64'))

        # Define last layer
        self.net.append(Dense(dim,
                              kernel_initializer=LecunNormal(),
                              activation=last,
                              dtype='float64'))

        # Define optimizer
        self.opt = tk.optimizers.Adam(learning_rate = lr)

    # Network forward pass
    @tf.function
    def call(self, var):

        # Copy input
        x = var

        # Compute output
        for layer in range(len(self.net)):
            x = self.net[layer](x)

        return x
