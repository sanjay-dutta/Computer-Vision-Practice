import os
from glob import glob
# import tensorflow as tf
import cv2 as cv
import numpy as np
import random

# Set seed for reproducibility
seed_constant = 50
np.random.seed(seed_constant)
random.seed(seed_constant)
# tf.random.set_seed(seed_constant)

# Directory and class definitions
data_set_dir = '/impacs/sad64/SLURM/dataset/UCF50'
CLASSES_LIST = [
    'BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk',
    'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop',
    'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lunges', 'MilitaryParade',
    'Mixing', 'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin',
    'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing',
    'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing',
    'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo'
]
seq_len = 64
resized_height, resized_width = 64, 64

def extract_frames(video_path):
    extracted_frames = []
    video_reader = cv.VideoCapture(video_path)
    total_frames = int(video_reader.get(cv.CAP_PROP_FRAME_COUNT))
    skip_frames = max(int(total_frames / seq_len), 1)
    
    for frame_counter in range(0, total_frames, skip_frames):
        video_reader.set(cv.CAP_PROP_POS_FRAMES, frame_counter)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv.resize(frame, (resized_width, resized_height))
        normalized_frame = resized_frame / 255.0
        extracted_frames.append(normalized_frame)
        if len(extracted_frames) == seq_len:
            break
            
    video_reader.release()
    extracted_frames = np.asarray(extracted_frames)
    return extracted_frames

def create_dataset():
    features, labels = [], []
    
    for idx, class_name in enumerate(CLASSES_LIST):
        print(f"Processing class: {class_name}")
        video_paths = glob(os.path.join(data_set_dir, class_name, "*"))
        
        for video_path in video_paths:
            extracted_frames = extract_frames(video_path)
            if len(extracted_frames) == seq_len:
                features.append(extracted_frames)
                labels.append(idx)
    
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

# Create dataset
# features, labels = create_dataset()

features,labels=create_dataset()
features=np.array(features)
np.save('features.npy',features)
labels=np.array(labels)
np.save('labels.npy',labels)
