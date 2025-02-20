import os

directory = r'C:\Users\wenji\OneDrive\Desktop\Y3S2\ATAP\Image Enhancement Project\image-enhancement-app\backend\uploads'
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('jpeg'):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)
