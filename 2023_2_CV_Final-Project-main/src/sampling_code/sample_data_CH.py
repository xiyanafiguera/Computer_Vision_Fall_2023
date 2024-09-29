import os
import shutil
import random


def shuffle_and_save_every_nth_file(
    input_folder, train_folder, val_folder, interval=21, total_size=8064
):
    # Ensure the output folder exists
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # shuffle the files
    # random.seed(42)
    # random.Random(4).shuffle(files)
    files = sorted(files)

    # pick one file every 21 files for validation and the rest for training
    # do this for 384 times
    for i in range(total_size):
        # Copy the file to the val folder
        if i % interval == 0:
            input_path = os.path.join(input_folder, files[i])
            val_path = os.path.join(val_folder, files[i])
            shutil.copy(input_path, val_path)

        else:
            # Copy the file to the train folder
            input_path = os.path.join(input_folder, files[i])
            train_path = os.path.join(train_folder, files[i])
            shutil.copy(input_path, train_path)


# Example usage
input_folder = "data/CrowdHuman/teacher_preds/"
train_folder = "data/combined_dataset/teacher_preds/CH/train/"
val_folder = "data/combined_dataset/teacher_preds/CH/val/"
interval = 21
total_size = 384 + 7680
shuffle_and_save_every_nth_file(
    input_folder, train_folder, val_folder, interval, total_size
)
