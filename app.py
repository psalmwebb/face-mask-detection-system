from flask import Flask,render_template,request,jsonify
import cv2
import base64
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import requests
from pwcnn import PWCNN



app = Flask(__name__)


def plot_detected_images(cropped_imgs):
    fig = plt.figure(figsize = (5,5),dpi = 80)
    for i,blob in enumerate(cropped_imgs):
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(blob)

    plt.show()

## variable for removing the useless part of the images

TRUNCS = 22

## Loading the trained Convnet
checkpoint = torch.load('maskVSNoMask_checkpoints.pth',map_location=torch.device('cpu'))
model = PWCNN()
model.load_state_dict(checkpoint['cnn'])

def convertArr2byte(img_array):
    # i converted the numpy array (image) to an Image object
        img_to_send = Image.fromarray(img_array.astype("uint8"))

        # i allocated  a memory in the RAM
        rawBytes = io.BytesIO()
        # Now, i save the image to that memory (RAM)
        img_to_send.save(rawBytes, "JPEG")
        # Then i call the memory
        rawBytes.seek(0)
        # I converted tha image now to encoded base64 format
        img_base64 = base64.b64encode(rawBytes.read())
        return img_base64


@app.route('/home',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/detect',methods=['POST'])
def detect():
    blob = request.form['imgURL']

    if ('http' in blob):
        blob = requests.get(blob,timeout=30).content
        blob = base64.b64encode(blob)
        url = blob
    else:
        url=blob[TRUNCS:]
    draw = base64.b64decode(url)
    img  = np.asarray(bytearray(draw),dtype="uint8")
    detect_img = cv2.imdecode(img,cv2.IMREAD_COLOR)
    detect_img = cv2.cvtColor(detect_img,cv2.COLOR_BGR2RGB)
    # resized = cv2.resize(img,(50,50),interpolation=cv2.INTER_AREA)
    detect_img = np.asarray(detect_img,dtype="uint8")  ##  The numpy array of the image

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

    img_array = detect_img  ## the numpy array representation of the image not the tensor.

    print(img_array.shape)

    scale_factor = 1.05

    faces = face_cascade.detectMultiScale(img_array,scaleFactor=scale_factor,minNeighbors=4,minSize=(30,30))

    profile_faces = profile_cascade.detectMultiScale(img_array,scaleFactor=scale_factor,minNeighbors=4,minSize=(30,30))

   ## making a variable to store the cropped images.
    cropped_imgs = []


    # checking if the faces of of lenght zero.
    if len(faces) > 0:
   #Looping through all the faces detected my the A.I, cropping them and appending them to our "cropped_imgs" array.
        for (x,y,w,h) in faces:
            cropped_imgs.append(detect_img[y:y+h,x:x+w])
        print(faces)

        for (x,y,w,h) in faces:
            cv2.rectangle(img_array,(x,y),(x+w,y+h),(163,243,0),2)

        # plot_detected_images(cropped_imgs)

   ## Checking if the profile faces are of length zero.
    elif len(profile_faces) > 0:
        for (x,y,w,h) in profile_faces:
            cropped_imgs.append(detect_img[y:y+h,x:x+w])
        print(faces)


        for (x,y,w,h) in profile_faces:
            cv2.rectangle(img_array,(x,y),(x+w,y+h),(163,243,0),4)

        # plot_detected_images(cropped_imgs)

    else:
         cropped_imgs.append(detect_img)


    data = {
      "0":str(convertArr2byte(img_array))
    }

    # Appending the cropped images to the data object for processing in the front-end

    for i,blob in enumerate(cropped_imgs):
        data[str(i+1)] = str(convertArr2byte(blob))

    # Finally, i returned  the detected image to the client.
    return jsonify(data)
    # return img_base64



@app.route('/predict',methods=['POST'])
def predict():

    all_img_to_pred = []
    img_count = 0
    data = {}

    img_files = request.form

    for img_index in img_files:
        img_file = img_files[img_index]
        img_file = img_file[TRUNCS:]
        img_file = base64.b64decode(img_file)
        img_file = np.asarray(bytearray(img_file),dtype='uint8')
        img_file = cv2.imdecode(img_file,0) / 255.0
        img_file = cv2.resize(img_file,(50,50))

        all_img_to_pred += [img_file]
        img_count+=1

    all_img_to_pred = torch.Tensor(all_img_to_pred).view(img_count,1,50,50)
    print(all_img_to_pred.shape)

    _,real_preds = model(all_img_to_pred).max(1)

    real_preds = real_preds.tolist()

    print(real_preds)

    for i in range(len(real_preds)):
        if real_preds[i] == 0:
            data[str(i)] = 'No Mask'
        elif real_preds[i] == 1:
            data[str(i)] = 'Mask detected'

    return jsonify(data)




if __name__ == "__main__":
     app.run(debug=True)
