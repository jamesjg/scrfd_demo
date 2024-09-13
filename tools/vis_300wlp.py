import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from collections import Counter

# Path to the directory containing MATLAB pose annotation files
directory_path = '/mnt/data/lanxing/300W_LP/AFW'

# Function to process a single MATLAB file and return Euler angles
def process_mat_file(file_path):
    try:
        # Load the MATLAB file
        mat = scipy.io.loadmat(file_path)

        # Extract pose parameters
        pose_params = mat['Pose_Para'][0]

        # Extract the rotation matrix and translation vector
        if pose_params.size >= 3:
            pitch_param, yaw_param, roll_param = pose_params[:3]
        else:
            raise ValueError(f"Unexpected shape for Pose_Para in {file_path}. Expected at least 3 elements, got {pose_params.size}")

        # Convert the parameters to degrees
        roll = roll_param * 180 / np.pi
        pitch = pitch_param * 180 / np.pi
        yaw = yaw_param * 180 / np.pi

        return roll, pitch, yaw
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

# Initialize counters for each Euler angle
rolls = []
pitches = []
yaws = []

# Iterate through all files in the directory
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.mat'):
            file_path = os.path.join(root, file)
            roll, pitch, yaw = process_mat_file(file_path)
            if roll is not None:
                rolls.append(round(roll))
                pitches.append(round(pitch))
                yaws.append(round(yaw))

# Count occurrences of each angle
roll_counter = Counter(rolls)
pitch_counter = Counter(pitches)
yaw_counter = Counter(yaws)

# Plot the histograms
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Roll
axes[0].bar(roll_counter.keys(), roll_counter.values())
axes[0].set_title('Roll Angle Distribution')
axes[0].set_xlabel('Roll Angle (degrees)')
axes[0].set_ylabel('Count')

# Pitch
axes[1].bar(pitch_counter.keys(), pitch_counter.values())
axes[1].set_title('Pitch Angle Distribution')
axes[1].set_xlabel('Pitch Angle (degrees)')
axes[1].set_ylabel('Count')

# Yaw
axes[2].bar(yaw_counter.keys(), yaw_counter.values())
axes[2].set_title('Yaw Angle Distribution')
axes[2].set_xlabel('Yaw Angle (degrees)')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.show()

