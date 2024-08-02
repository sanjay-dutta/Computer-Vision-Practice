# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras import layers
# # from tensorflow.keras.utils import to_categorical, plot_model
# # from tensorflow.keras.optimizers import Adam
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import random

# # # Load data
# # labels = np.load('labels.npy')
# # features = np.load('features.npy')

# # # Define constants
# # INPUT_SHAPE = (64, 64, 64, 3)
# # NUM_CLASSES = len(np.unique(labels))
# # LEARNING_RATE = 1e-3
# # WEIGHT_DECAY = 1e-5
# # EPOCHS = 100
# # PATCH_SIZE = (8, 8, 8)
# # NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2
# # LAYER_NORM_EPS = 1e-6
# # PROJECTION_DIM = 128
# # NUM_HEADS = 8
# # NUM_LAYERS = 8

# # # Set seeds for reproducibility
# # seed_constant = 50
# # np.random.seed(seed_constant)
# # random.seed(seed_constant)
# # tf.random.set_seed(seed_constant)

# # # Preprocess labels
# # labels = to_categorical(labels, num_classes=NUM_CLASSES)

# # # Split data
# # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# # class TubeletEmbedding(layers.Layer):
# #     def __init__(self, embed_dim, patch_size, **kwargs):
# #         super().__init__(**kwargs)
# #         self.projection = layers.Conv3D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding="VALID")
# #         self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

# #     def call(self, videos):
# #         projected_patches = self.projection(videos)
# #         flattened_patches = self.flatten(projected_patches)
# #         return flattened_patches

# # class PositionalEncoder(layers.Layer):
# #     def __init__(self, embed_dim, **kwargs):
# #         super().__init__(**kwargs)
# #         self.embed_dim = embed_dim

# #     def build(self, input_shape):
# #         _, num_tokens, _ = input_shape
# #         self.position_embedding = layers.Embedding(input_dim=num_tokens, output_dim=self.embed_dim)
# #         self.positions = tf.range(start=0, limit=num_tokens, delta=1)

# #     def call(self, encoded_tokens):
# #         encoded_positions = self.position_embedding(self.positions)
# #         encoded_tokens += encoded_positions
# #         return encoded_tokens

# # def create_vivit_classifier(input_shape=INPUT_SHAPE, transformer_layers=NUM_LAYERS, num_heads=NUM_HEADS, embed_dim=PROJECTION_DIM, layer_norm_eps=LAYER_NORM_EPS, num_classes=NUM_CLASSES):
# #     inputs = layers.Input(shape=input_shape)
# #     tubelet_embedder = TubeletEmbedding(embed_dim=embed_dim, patch_size=PATCH_SIZE)
# #     positional_encoder = PositionalEncoder(embed_dim=embed_dim)
    
# #     patches = tubelet_embedder(inputs)
# #     encoded_patches = positional_encoder(patches)

# #     for _ in range(transformer_layers):
# #         x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
# #         attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1)(x1, x1)
# #         x2 = layers.Add()([attention_output, encoded_patches])
# #         x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
# #         x3 = tf.keras.Sequential([
# #             layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
# #             layers.Dense(units=embed_dim, activation=tf.nn.gelu),
# #         ])(x3)
# #         encoded_patches = layers.Add()([x3, x2])

# #     representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
# #     representation = layers.GlobalAvgPool1D()(representation)
# #     outputs = layers.Dense(units=num_classes, activation="softmax")(representation)
    
# #     model = tf.keras.Model(inputs=inputs, outputs=outputs)
# #     model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
# #     return model

# # vvit_model = create_vivit_classifier()
# # vvit_model.summary()

# # checkpoint = tf.keras.callbacks.ModelCheckpoint('vvit_saved_model', monitor='val_accuracy', save_weights_only=True, save_best_only=False, verbose=1)
# # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)

# # history = vvit_model.fit(x=X_train, y=y_train, validation_split=0.2, callbacks=[checkpoint, early_stopping], shuffle=True, epochs=EPOCHS, batch_size=8)

# # # Evaluate model
# # test_loss, test_accuracy = vvit_model.evaluate(x=X_test, y=y_test, verbose=1)
# # print(f"Test Accuracy: {test_accuracy}")

# # # Predictions and metrics calculation
# # y_pred = np.argmax(vvit_model.predict(X_test), axis=1)
# # y_true = np.argmax(y_test, axis=1)

# # accuracy = accuracy_score(y_true, y_pred)
# # precision = precision_score(y_true, y_pred, average='weighted')
# # f1 = f1_score(y_true, y_pred, average='weighted')
# # conf_matrix = confusion_matrix(y_true, y_pred)

# # print(f"Accuracy: {accuracy}")
# # print(f"Precision: {precision}")
# # print(f"F1 Score: {f1}")

# # # Plot confusion matrix
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(NUM_CLASSES), yticklabels=np.arange(NUM_CLASSES))
# # plt.xlabel('Predicted Labels')
# # plt.ylabel('True Labels')
# # plt.title('Confusion Matrix')
# # plt.show()

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import random

# GPU_LIMIT = 32 * 1024

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_LIMIT)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

# # Load data
# labels = np.load('labels.npy')
# features = np.load('features.npy')

# # Define constants
# INPUT_SHAPE = (64, 64, 64, 3)
# NUM_CLASSES = len(np.unique(labels))
# LEARNING_RATE = 1e-3
# WEIGHT_DECAY = 1e-4
# EPOCHS = 100
# PATCH_SIZE = (8, 8, 8)
# NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2
# LAYER_NORM_EPS = 1e-6
# PROJECTION_DIM = 128
# NUM_HEADS = 8
# NUM_LAYERS = 8

# # Define class labels
# CLASSES_LIST = [
#     'BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk',
#     'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop',
#     'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lunges', 'MilitaryParade',
#     'Mixing', 'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin',
#     'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing',
#     'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing',
#     'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo'
# ]

# # Set seeds for reproducibility
# seed_constant = 50
# np.random.seed(seed_constant)
# random.seed(seed_constant)
# tf.random.set_seed(seed_constant)

# # Preprocess labels
# labels = to_categorical(labels, num_classes=NUM_CLASSES)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# class TubeletEmbedding(layers.Layer):
#     def __init__(self, embed_dim, patch_size, **kwargs):
#         super().__init__(**kwargs)
#         self.projection = layers.Conv3D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding="VALID")
#         self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

#     def call(self, videos):
#         projected_patches = self.projection(videos)
#         flattened_patches = self.flatten(projected_patches)
#         return flattened_patches

# class PositionalEncoder(layers.Layer):
#     def __init__(self, embed_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim

#     def build(self, input_shape):
#         _, num_tokens, _ = input_shape
#         self.position_embedding = layers.Embedding(input_dim=num_tokens, output_dim=self.embed_dim)
#         self.positions = tf.range(start=0, limit=num_tokens, delta=1)

#     def call(self, encoded_tokens):
#         encoded_positions = self.position_embedding(self.positions)
#         encoded_tokens += encoded_positions
#         return encoded_tokens

# def create_vivit_classifier(input_shape=INPUT_SHAPE, transformer_layers=NUM_LAYERS, num_heads=NUM_HEADS, embed_dim=PROJECTION_DIM, layer_norm_eps=LAYER_NORM_EPS, num_classes=NUM_CLASSES):
#     inputs = layers.Input(shape=input_shape)
#     tubelet_embedder = TubeletEmbedding(embed_dim=embed_dim, patch_size=PATCH_SIZE)
#     positional_encoder = PositionalEncoder(embed_dim=embed_dim)
    
#     patches = tubelet_embedder(inputs)
#     encoded_patches = positional_encoder(patches)

#     for _ in range(transformer_layers):
#         x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#         attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1)(x1, x1)
#         x2 = layers.Add()([attention_output, encoded_patches])
#         x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
#         x3 = tf.keras.Sequential([
#             layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
#             layers.Dense(units=embed_dim, activation=tf.nn.gelu),
#         ])(x3)
#         encoded_patches = layers.Add()([x3, x2])

#     representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
#     representation = layers.GlobalAvgPool1D()(representation)
#     outputs = layers.Dense(units=num_classes, activation="softmax")(representation)
    
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# vvit_model = create_vivit_classifier()
# vvit_model.summary()

# checkpoint = tf.keras.callbacks.ModelCheckpoint('vvit_saved_model', monitor='val_accuracy', save_weights_only=True, save_best_only=False, verbose=1)
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)

# history = vvit_model.fit(x=X_train, y=y_train, validation_split=0.2, callbacks=[checkpoint, early_stopping], shuffle=True, epochs=EPOCHS, batch_size=8)

# # Evaluate model
# test_loss, test_accuracy = vvit_model.evaluate(x=X_test, y=y_test, verbose=1)
# print(f"Test Accuracy: {test_accuracy}")

# # Predictions and metrics calculation
# y_pred = np.argmax(vvit_model.predict(X_test), axis=1)
# y_true = np.argmax(y_test, axis=1)

# accuracy = accuracy_score(y_true, y_pred)
# precision = precision_score(y_true, y_pred, average='weighted')
# f1 = f1_score(y_true, y_pred, average='weighted')
# conf_matrix = confusion_matrix(y_true, y_pred)

# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"F1 Score: {f1}")

# # Plot confusion matrix
# plt.figure(figsize=(12, 10))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import random

GPU_LIMIT = 32 * 1024

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_LIMIT)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Load data
labels = np.load('labels.npy')
features = np.load('features.npy')

# Define constants
INPUT_SHAPE = (64, 64, 64, 3)
NUM_CLASSES = len(np.unique(labels))
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

# Define class labels
CLASSES_LIST = [
    'BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk',
    'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop',
    'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lunges', 'MilitaryParade',
    'Mixing', 'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin',
    'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing',
    'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing',
    'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo'
]

# Set seeds for reproducibility
seed_constant = 50
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Preprocess labels
labels = to_categorical(labels, num_classes=NUM_CLASSES)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding="VALID")
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(input_dim=num_tokens, output_dim=self.embed_dim)
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens += encoded_positions
        return encoded_tokens

def create_vivit_classifier(input_shape=INPUT_SHAPE, transformer_layers=NUM_LAYERS, num_heads=NUM_HEADS, embed_dim=PROJECTION_DIM, layer_norm_eps=LAYER_NORM_EPS, num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    tubelet_embedder = TubeletEmbedding(embed_dim=embed_dim, patch_size=PATCH_SIZE)
    positional_encoder = PositionalEncoder(embed_dim=embed_dim)
    
    patches = tubelet_embedder(inputs)
    encoded_patches = positional_encoder(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = tf.keras.Sequential([
            layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
            layers.Dense(units=embed_dim, activation=tf.nn.gelu),
        ])(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

vvit_model = create_vivit_classifier()
vvit_model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('vvit_saved_model', monitor='val_accuracy', save_weights_only=True, save_best_only=False, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)

history = vvit_model.fit(x=X_train, y=y_train, validation_split=0.2, callbacks=[checkpoint, early_stopping], shuffle=True, epochs=EPOCHS, batch_size=4)

# Evaluate model
test_loss, test_accuracy = vvit_model.evaluate(x=X_test, y=y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy}")

# Predictions and metrics calculation
y_pred = np.argmax(vvit_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")

# Plot and save confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
