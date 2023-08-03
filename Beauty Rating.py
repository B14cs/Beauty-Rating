import os
import datetime
import cv2
import numpy as np
import glob
from PIL import Image
from keras.models import load_model
from tkinter import filedialog
import dlib
from time import time

# Load the beauty rating model
print('Beauty Rating Model is loading...')
model = load_model('models/Beauty Rating Model.h5')
print('model has loaded')

# Load the HOG Face Detector
print('HOG Face Detector is loading...')
hog_face_detector = dlib.get_frontal_face_detector()
print('HOG Face Detector has loaded')


def beautyRate(img_path):
    # Read an image
    image = img_path

    # Face detection
    print('Face detecting...')
    try:
        return hogDetectFaces(image, hog_face_detector)
    except Exception as e:
        print('Error while face detecting')
        print(e)
        exit()


# Face Detection Using the DLIB Face Detector Model
def hogDetectFaces(image, hog_face_detector, display=True):

    output_image = image.copy()

    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hog_face_detector(imgRGB, 0)
    print('Number of detected faces:', len(results))

    i = 1
    for bbox in results:
        ''' top = Distance from top
            right = Distance from right
            bottom = Distance from bottom
            left = Distance from left '''
        left = bbox.left()
        top = bbox.top()
        right = bbox.right()
        bottom = bbox.bottom()

        # Define image dimensions
        img_height, img_width, _ = image.shape

        # Face selection
        face_image = image[top:bottom, left:right]

        # Define face dimensions
        height, width, _ = face_image.shape

        # Make face dimensions square
        if width > height:
            new_height = height + (width - height)
            new_width = width
        elif width < height:
            new_width = width + (height - width)
            new_height = height
        else:
            new_height = height
            new_width = width

        # Padding
        new_top = top - int(new_height * 0.5)
        new_right = right + int(new_width * 0.5)
        new_bottom = bottom + int(new_height * 0.5)
        new_left = left - int(new_width * 0.5)
        if new_top < 0:
            new_top = 0
        if new_right > img_width:
            new_right = img_width
        if new_bottom > img_height:
            new_bottom = img_height
        if new_left < 0:
            new_left = 0

        # Time determination
        time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # Face cropping
        face_image = image[new_top:new_bottom, new_left:new_right]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_image)

        # Save cropped face image
        try:
            pil_img.save(
                f'img/Cropped images/image ({i}) cropped in {time}.jpg')
            print(f'image ({i}) has Saved')
        except:
            print(f'Error while saving {time}.jpg')

        i += 1

    # Import saved images
    img_test_list = glob.glob('img/Cropped images/*.jpg')

    # Resize them to fit model-trained images
    imgs_test_resized = []
    for i, img_path in enumerate(img_test_list):
        imag = cv2.imread(img_path)
        img_resized = cv2.resize(imag, (350, 350))
        img_resized = img_resized.astype(np.float32) / 255.
        imgs_test_resized.append(img_resized)
    imgs_test_resized = np.array(imgs_test_resized, dtype=np.float32)

    # Predict
    preds = model.predict(imgs_test_resized)
    print('Predicts: ')
    print(preds)

    # Final result
    k = 0
    for bbox in results:
        left = bbox.left()
        top = bbox.top()
        right = bbox.right()
        bottom = bbox.bottom()

        # Percentages of face for drawing
        rectangle_percentage = int(height * 0.03)
        font_percentage = int(rectangle_percentage * 0.67)
        thkness_percentage = int(rectangle_percentage * 0.79)

        if font_percentage < 3:
            font_percentage = 3
        if thkness_percentage < 4:
            thkness_percentage = 4

        if font_percentage > 8:
            font_percentage = 8
        if thkness_percentage > 9:
            thkness_percentage = 9

        if rectangle_percentage > 10:
            rectangle_percentage = 10

        y_text = top - 10
        if y_text < 50:
            y_text = bottom + (font_percentage * 12)
            if y_text > (img_height - 50):
                y_text = bottom

        # Draw a rectangle around the face
        cv2.rectangle(output_image, (left, top),
                      (right, bottom), (0, 255, 0), rectangle_percentage)
        # Write a rate above the rectangle
        cv2.putText(output_image, '%.2f' % (
            preds[k]), (left, y_text), cv2.FONT_HERSHEY_PLAIN, font_percentage, (0, 255, 0), thkness_percentage, cv2.LINE_AA)
        k += 1

    # Remove cropped photos
    fileList = glob.glob('img/Cropped images/*.jpg')
    for i, filePath in enumerate(fileList):
        try:
            os.remove(filePath)
            print(f'face{i}.jpg is removed')
        except:
            print('Error while deleting file : ', filePath)

    return output_image


# Open file dialog to select an image
img_path = filedialog.askopenfilename(initialdir='img/Test/')
image = cv2.imread(img_path)

# Call the function
output_image = beautyRate(image)

# Convert the image to RGB back
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Save the result
time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
pil_img = Image.fromarray(output_image[:, :, ::-1])
try:
    pil_img.save(f'img/Results/Image rated in {time}.jpg')
    print(f'Image rated in {time} has saved')
except:
    print(f'Error while saving {time}.jpg')
# Show the result
pil_img.show()


# # More than one image
# # Read all images in the Test folder
# img_list = glob.glob('img/Test/*.jpg')

# for i, img_path in enumerate(img_list):
#     image = cv2.imread(img_path)
#     # Call the function
#     beautyRate(image)
#     # Convert the image to RGB back
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Save the results
#     time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
#     pil_img = Image.fromarray(image)
#     pil_img.save(f'img/Results/Image rated in {time}.jpg')
