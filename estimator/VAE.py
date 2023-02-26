from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


from tensorflow.keras.layers import BatchNormalization,Conv2D,Conv2DTranspose,Dense,Input,Activation,Flatten,Lambda,Reshape,Lambda 
from tensorflow.keras.models import Model
import tensorflow.keras.backend as k
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from conf.settings import Settings

settings = Settings()

kl_multiplier = settings.kl_multiplier
rmse_multiplier = settings.rmse_multiplier
latent_space = settings.latent_space

img_height = settings.img_height
img_width = settings.img_width

class VariationalAutoEncoder():
    def __init__(self):
        self._build()
        
    def _build(self):
        
        #Encoder
        
        def sampling(args):
            mu,log_var = args
            epsilon = k.random_normal(shape = k.shape(mu),mean = 0,stddev = 1)
            return mu + k.exp(log_var/2) * epsilon
        
        image_input = Input(shape = (img_height,img_width,1), name = "vae_input")

        x_enc = Conv2D(784,(3,3),(2,2),padding = "same", name = "conv_784encoder")(image_input)
        x_enc = BatchNormalization(name = "batch_norm_784_encoder")(x_enc)
        x_enc = Activation("relu",name = "activation_784_encoder")(x_enc)

        x_enc = Conv2D(256,(3,3),(2,2),padding = "same", name = "conv_256_encoder")(x_enc)
        x_enc = BatchNormalization(name = "batch_norm_256_encoder")(x_enc)
        x_enc = Activation("relu",name = "activation_256_encoder")(x_enc)
        
        x_enc = Conv2D(32,(3,3),(1,1),padding = "same", name = "conv_32_encoder")(image_input)
        x_enc = BatchNormalization(name = "batch_norm_32_encoder")(x_enc)
        x_enc = Activation("relu",name = "activation_32_encoder")(x_enc)
        
        x_enc = Conv2D(4,(3,3),(1,1),padding = "same", name = "conv_4_encoder")(image_input)
        x_enc = BatchNormalization(name = "batch_norm_4_encoder")(x_enc)
        x_enc = Activation("relu",name = "activati132on_4_encoder")(x_enc)

        x_enc = Flatten(name = "flatten_encoder")(x_enc)
        self.mu = Dense(latent_space, activation = "linear",name = "mu")(x_enc)
        self.log_var = Dense(latent_space, activation = "linear", name = "log_var")(x_enc)

        encoder_output = Lambda(sampling, name = "encoder_output")([self.mu,self.log_var])        

        self.encoder = Model(inputs = [image_input], outputs = [encoder_output])
        
        #Decoder
        
        latent_space_input = Input(shape=(latent_space,), name = "decoder_input")

        x_dec = Dense(3136, activation = "linear",name = "decoder_dense")(latent_space_input)
        x_dec = Reshape(target_shape = (7,7,64)) (x_dec)

        x_dec = Conv2DTranspose(784,(3,3),(1,1),padding = "same", name = "conv_784_decoder")(x_dec)
        x_dec = BatchNormalization(name = "batch_norm_784_decoder")(x_dec)
        x_dec = Activation("relu",name = "activation_784_decoder")(x_dec)
        
        x_dec = Conv2DTranspose(256,(3,3),(1,1),padding = "same", name = "conv_256_decoder")(x_dec)
        x_dec = BatchNormalization(name = "batch_norm_256_decoder")(x_dec)
        x_dec = Activation("relu",name = "activation_256_decoder")(x_dec)
                       
        x_dec = Conv2DTranspose(32,(3,3),(2,2),padding = "same", name = "conv_13_31decoder")(x_dec)
        x_dec = BatchNormalization(name = "batch_norm_32_decoder")(x_dec)
        x_dec = Activation("relu",name = "activation_32_decoder")(x_dec)
                       
        
        x_dec = Conv2DTranspose(4,(3,3),(2,2),padding = "same", name = "conv_4_decoder")(x_dec)
        x_dec = BatchNormalization(name = "batch_norm_4_decoder")(x_dec)
        x_dec = Activation("relu",name = "activation_4_decoder")(x_dec)

        x_dec = Conv2DTranspose(1,(3,3),(1,1),padding = "same", name = "conv_1_decoder")(x_dec)
        x_dec = BatchNormalization(name = "batch_norm_1_decoder")(x_dec)
        decoder_output = Activation("sigmoid",name = "activation_1_decoder")(x_dec)       
 
        self.decoder = Model(inputs = [latent_space_input], outputs = [decoder_output])
        
        
        #Variational AutoEncoder
        
        model_input = image_input
        model_output = self.decoder(encoder_output)

        self.full_model = Model(model_input, model_output)
        
    def custom_compile(self, learning_rate, rmse_multiplier):
        self.learning_rate = learning_rate
       
        def vae_r_loss(y_true, y_pred):
            r_loss = k.mean(k.square(y_true - y_pred), axis = [1])
            return rmse_multiplier * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  - 0.5 * k.sum(1 + self.log_var - k.square(self.mu) - k.exp(self.log_var), axis = 1)
            return kl_multiplier * kl_loss 

        def vae_loss(y_true, y_pred):
            reconstruction_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  reconstruction_loss + kl_loss 

        optimizer = Adam(learning_rate=learning_rate)
        self.full_model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss,vae_kl_loss]) 