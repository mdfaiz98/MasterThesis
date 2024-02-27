

import os

# Deleting 'final_combined_data.csv' file from each participant's folder
for participant_id in range(5, 20):  # Folders 5 to 19
    folder_path = f'C:\\Thesis-script\\biobss\\data\\{participant_id}'  # Change this to the folder path of your choice
    file_path = os.path.join(folder_path, 'final_combined_data.csv')

    # Check if the file exists before attempting to delete
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted 'final_combined_data.csv' from {folder_path}")
    else:
        print(f"No file found: 'final_combined_data.csv' in {folder_path}")
