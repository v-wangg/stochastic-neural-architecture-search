class BaseOptions():
    def initialize(self):
        self.dataset = './dataset' # path to the dir of the dataset
        self.name = 'experiment' # Name of the experiment

class TrainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.initialize(self)
        self.init_channels = 16
        self.layers = 16
        self.train_portion = 0.5
        self.report_freq = 50
        self.batch_size = 64 # batch size from paper
        self.learning_rate = 0.025 # lr frmo paper
        self.learning_rate_min = 0.001 # lr_min from paper
        self.momentum = 0.9 # momentum from paper
        self.weight_decay = 3e-4 # weight decay from papay
        self.epochs = 150 # num epochs from pape
        self.cutout = True # use cutout
        self.cutout_length = 16 # cutout length
        self.arch_learning_rate = 3e-4
        self.arch_weight_decay = 1e-3 # architecture weight decay from paper
        self.arch_betas = (0.5, 0.999)
        self.initial_temp = 2.5 # initial softmax temperature from paper
        self.anneal_rate = 0.00003 # annealation rate of softmax temperature from paper