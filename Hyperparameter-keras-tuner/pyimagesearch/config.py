# define the path to our output directory
OUTPUT_PATH = "output"

# initialize the input shape and number of classes
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

# define the total number of epochs to train, batch size, and the
# early stopping patience
EPOCHS = 50
BS = 32
EARLY_STOPPING_PATIENCE = 5