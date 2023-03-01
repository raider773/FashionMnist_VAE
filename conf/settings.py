
class Settings():
    
    def __init__(self):
        
        self.metrics_folder = "tmp/metrics"
        self.model_folder = "tmp/model"
        
        
        self.img_height = 28
        self.img_width = 28
        self.latent_space = 3
        self.rmse_multiplier = 5000
        self.kl_multiplier = 1

        self.epochs = 100
        self.patience = 10        
        self.batch_size = 256
        self.learning_rate = 0.001

        self.amount_to_generate = 50
        self.amount_of_vector_multipliers = 20
  