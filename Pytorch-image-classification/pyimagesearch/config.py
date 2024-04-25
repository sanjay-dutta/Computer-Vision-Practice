# import the necessary packages
import torch

# specify image dimension
IMAGE_SIZE = 224

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# determine the device we will be using for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# specify path to the ImageNet labels
IN_LABELS = "ilsvrc2012_wordnet_lemmas.txt"