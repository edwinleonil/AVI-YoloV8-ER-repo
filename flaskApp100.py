from flask import Flask, render_template, request
from PIL import Image, ImageDraw
import os
import numpy as np
from ultralytics import YOLO
import os
import glob

app = Flask(__name__)

class App:
    def __init__(self, model_path, folder_path=None, image_path=None):
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.folder_path = folder_path
        self.image_path = image_path
        self.file_list = []
        self.image_index = 0

        if self.folder_path is not None:
            self.file_list = sorted(glob.glob(os.path.join(self.folder_path, '*.png')))
            if len(self.file_list) == 0:
                raise ValueError(f"No images found in folder: {self.folder_path}")
            self.image_path = self.file_list[self.image_index]

    def display_image(self, image_index):
        
        # joint the folder path and the image name
        # image_path = os.path.join(self.folder_path, image_index)
        image_path = self.file_list[image_index]
        
        # Run inference on a single image
        results = self.model(image_path)

        # get the label, confidence and xyxy from the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

        # save the image to a temporary file
        temp_file = f"static/temp/{image_index}.png"
        im.save(temp_file)

        return temp_file

# Define the default model path
default_model_path = 'runs/detect/train/weights/best.pt'

# default folder path
default_folder_path = 'dataset/images/val'

app_instance = App(default_model_path, folder_path=default_folder_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    global app_instance
    if request.method == 'POST':
        if 'image' in request.files:
            image_file = request.files['image']
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            app_instance = App(default_model_path, image_path)
    
    # Get a list of all the files in the folder
    folder_path = app_instance.folder_path
    files = os.listdir(folder_path)

    image_path = app_instance.display_image(app_instance.image_index)

    return render_template('index.html', image_path=image_path, files=files)

if __name__ == '__main__':
    app.run(debug=True)