class Config():
    def __init__(self):
        # Parameters related to the problem
        self.num_classes = 1 # 1 class for regression
        
        # Parameters related to network
        self.original_checkpoint_path = "C:/Users/kevin/.cache/kagglehub/models/metaresearch/llama-3/pyTorch/8b/1"
        self.original_checkpoint_file_path = "/kaggle/input/llama-3/pytorch/8b/1/consolidated.00.pth"
        self.network_type = "llama3"
        self.architecture = {"backbone": "/kaggle/input/llama-3/transformers/8b-hf/1", # this is only used for tokenizers (from HF)
                             "params": {}}

        self.remove_layers = 16 # number of layer to remove to make the model smaller
        self.freeze_layers = None # number of layers to freeze to reduce number of training parameters
        self.lora_config = {"r": 8, # rank of the decomposed matrix (higher means less memory saving)
                            "lora_alpha": 16, # scaling factor, should be 2xr according to https://www.entrypointai.com/blog/lora-fine-tuning/
                            "lora_dropout": 0.05,
                            # make sure that you name correctly your modules according to your backbone
                            # you should spot the linear layers in the attention blocks
                            "target_modules": ['q_proj', 'v_proj', 'k_proj', 'output_proj'],
                            "quantize" : True, # True for QLora, False for LORA
                           }
        self.attn_dropout = 0.05
        self.computation_type = "bfloat16" 
        self.token_info = {"padding" :"longest", # batch are going to be the length of longest sequence
                           "max_length" : 1024, # max training sample length
                           "truncation": True,
                           "pad_to_multiple_of" : 512 # I heard that modern GPUs are fastest with multiple of 512? is that True?
                          }

        # Parameters related to training
        self.max_epochs = 2 # number of epochs

        self.initial_lr =3e-4
        self.optimizer_name = "AdamW" #"AdamW" # try 8 bit adam AdamW8bit     
        self.optimizer_params = {"lr": self.initial_lr, 
                                 "weight_decay":1e-2
                                }
        self.loss_config = {"loss_name" : "MSELoss",
                            "reduction":"mean",
                           }
        
        self.scheduler_name = "OneCycleLR"
        self.steps_per_epochs = -1 # this is automatically overwritten
        self.scheduler_params={
                              "max_lr":self.optimizer_params["lr"] if type(self.optimizer_params)==dict else self.optimizer_params[-1]["lr"],
                               "div_factor":10,
                              "steps_per_epoch": self.steps_per_epochs,
                              "final_div_factor":1e2, #1e2
                               "anneal_strategy":"cos", #"cos"
                               "three_phase" : False,
                              "pct_start":0.1, #0.3
                              "epochs": self.max_epochs}
        
        
        self.eval_on_train = False # You might want to compute the exact metric on training set to monitor overfitting
        self.batch_size = 1 #2 # Let's start small
        self.gradient_accumulation = 16 // self.batch_size # this allows you to train with low batch size but compute gradients on more that a few samples
        self.mixed_precision = True
        self.num_workers = 2 # I think num_workers for kaggle environment should be kept low
        self.pin_memory = True
        self.clip_value = 10.0

        # parameters related to logs
        self.verbose = 1 # how often do you want to compute the competition metric?
        self.save_path = "./torch-tune-logs"