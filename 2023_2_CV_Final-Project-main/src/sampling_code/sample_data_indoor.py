import os
import shutil


def save_every_nth_file(input_folder, train_folder, val_folder, interval=20):
    # Ensure the output folder exists
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Sort the files to ensure they are processed in order
    files.sort()

    # Iterate through the files and save every nth text file
    for i, file in enumerate(files):
        if i % interval == 14:
            # Copy the file to the val folder
            input_path = os.path.join(input_folder, file)
            val_path = os.path.join(val_folder, f"classroom_{file}")
            shutil.copy(input_path, val_path)

        else:
            # Copy the file to the train folder
            input_path = os.path.join(input_folder, file)
            train_path = os.path.join(train_folder, f"classroom_{file}")
            shutil.copy(input_path, train_path)


# Example usage
input_folder = "data/Indoor/images/1_classroom/"
train_folder = "data/combined_dataset/images/train/"
val_folder = "data/combined_dataset/images/val/"
save_every_nth_file(input_folder, train_folder, val_folder, interval=20)
