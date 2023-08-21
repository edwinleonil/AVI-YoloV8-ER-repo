# create a flask server to load images from a folder and display them on a webpage
# use a html template to display the image file list and the images

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

import os

app = Flask(__name__)

# define the path to the image folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')
APP_IMAGES = os.path.join(APP_STATIC, 'images')

# define the path to the template folder
APP_TEMPLATES = os.path.join(APP_STATIC, 'templates')


# define the route to the index page
@app.route('/')
def index():
    # get the list of image files
    image_names = os.listdir(APP_IMAGES)
    # render the index.html template with the image list
    return render_template("index1.html", image_names=image_names)


# define the route to the image file
@app.route('/images/<image_name>')
def images(image_name):
    # return the image file
    return send_from_directory(APP_IMAGES, image_name)


# define the route to the upload page
@app.route('/upload')
def upload():
    # render the upload.html template
    return render_template("upload.html")


# define the route to the upload action
@app.route('/uploadAction', methods=['POST'])
def uploadAction():
    # get the file from the request
    file = request.files['file']
    # save the file to the image folder
    file.save(os.path.join(APP_IMAGES, file.filename))
    # redirect to the index page
    return redirect(url_for('index'))


# run the app
if __name__ == '__main__':
    app.run(debug=True)

