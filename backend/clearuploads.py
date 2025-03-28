import os

directory = r'C://Users//wenji//OneDrive//Desktop//Y3S2//ATAP//Image Enhancement Project//image-enhancement-app//backend//uploads'

if os.path.exists(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
else:
    print(f"Directory does not exist: {directory}")
