import os
import random
import shutil

def move_files_percentage(source_folder, destination_folder, percentage):
    """Moves a specified percentage of files from a source folder to a destination folder.

    Args:
        source_folder (str): Path to the source folder.
        destination_folder (str): Path to the destination folder.
        percentage (float): Percentage of files to move (0.0 to 1.0).
    """
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    files = os.listdir(source_folder)
    num_files_to_move = int(len(files) * percentage)
    
    if num_files_to_move > len(files):
        num_files_to_move = len(files)

    files_to_move = random.sample(files, num_files_to_move)

    for file_name in files_to_move:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)
        print(f"Moved '{file_name}' to '{destination_folder}'")

if __name__ == "__main__":
    source_folder = "PlantVillage/val"  # Replace with your source folder path
    destination_folder = "PlantVillage/test"  # Replace with your destination folder path
    percentage_to_move = 0.33  # Adjust the percentage as needed

    # Create dummy files for testing
    # print(os.listdir(source_folder))
    for i in os.listdir(source_folder):
        move_files_percentage(source_folder + "/" + i, destination_folder + "/" + i, percentage_to_move)