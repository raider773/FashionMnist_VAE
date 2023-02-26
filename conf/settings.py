
class Settings():
    
    def __init__(self):
        
        self.metrics_folder = "tmp/metrics"
        self.model_folder = "tmp/model"
        
        
        self.img_height = 28
        self.img_width = 28
        self.latent_space = 3
        self.rmse_multiplier = 7000
        self.kl_multiplier = 1

        self.epochs = 3
        self.patience = 10        
        self.batch_size = 256
        self.learning_rate = 0.001
