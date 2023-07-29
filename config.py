from typing import Union

class TrainConfig(object):
    def __init__(self) -> None:
        
        # data parameters
        self.data: list[str] = []    # training data path
        self.out: str = ''        # location for saving results
        
        
        # training parameters
        self.run: int = 1           # run number to distinguish different runs
        self.max_proj: int = 1      # 1 if train network on max projection of 3D stacks else 0
        self.arch: str = 'hourglass_wres'   # CNN architechture to use for training (default is hourglass_wres, choices are unet, unet_fixed, hourglass_wres, hourglass_wores)
        self.mode: str = '2D'       # training mode (default is 2D, choices are 2D, 2.5D, 3D)
        self.depth: int = 1         # stack depth to use for training (must be odd number, default is 1)
        self.loss: int = 'l1'       # L2 or L1 loss for training (choices are l1 or l2)
        self.epochs: int = 150      # number of epochs to train the model for (150-200 is good choice, default is 150)
        self.lr: float = 0.001      # learning rate (default is 0.001)
        self.bs: int = 6            # batch size of training (default is 6)
        self.tsize: int = 0         # data size (number of images) to use for training in case training data is huge


class InferenceConfig(object):
    def __init__(self) -> None:
        self.data: list[str] = []   # paths of datasets to be denoised
        self.run: str = ''          # path of saved trained model