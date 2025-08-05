import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import keras
from twilio.rest import Client
from sklearn.metrics import confusion_matrix

ASSET_FOLDER = 'static/assets/'

CSS ='static'

TENSORBOARD='static/tensorboard'

app = Flask(__name__)
app.config['ASSET_FOLDER'] = ASSET_FOLDER
app.config['TENSORBOARD']=TENSORBOARD
app.config['CSS']=CSS

css = os.path.join(app.config['CSS'], 'style.css')
bg_video = os.path.join(app.config['ASSET_FOLDER'], 'bg_video.mp4')

acc=os.path.join(app.config['TENSORBOARD'],'accuracy.png')
loss=os.path.join(app.config['TENSORBOARD'],'epoch_loss.png')
acc_loss_graphs=os.path.join(app.config['TENSORBOARD'],'acc_loss_graphs.png')
cm=os.path.join(app.config['TENSORBOARD'],'cm.png')
cr=os.path.join(app.config['TENSORBOARD'],'classification_report.png')

    
@app.route('/')
def upload_form():
    return render_template('index.html', type="", css=css, bgvideo=bg_video)

@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['file']
    print(file.filename)
    file.save(os.path.join("static/uploads", secure_filename(file.filename)))

    imvar = tf.keras.preprocessing.image.load_img(os.path.join("static/uploads", secure_filename(file.filename))).resize((176, 176))
    imarr = tf.keras.preprocessing.image.img_to_array(imvar)
    imarr = np.array([imarr])
    model = keras.models.load_model("model")
    impred = model.predict(imarr)

    def roundoff(arr):
        """To round off according to the argmax of each predicted label array."""
        arr[np.argwhere(arr != arr.max())] = 0
        arr[np.argwhere(arr == arr.max())] = 1
        return arr

    for classpreds in impred:
        impred = roundoff(classpreds)
    
    classcount = 1
    for count in range(4):
        if impred[count] == 1.0:
            break
        else:
            classcount+=1
    
    classdict = {1: "Mild Dementia", 2: "Moderate Dementia", 3: "No Dementia, Patient is Safe", 4: "Very Mild Dementia"}
    print(classdict[classcount])

    true_label = 1  
    predicted_label = classcount
    
    cm = confusion_matrix([true_label], [predicted_label])
    print("Confusion Matrix:")
    print(cm)



    sid="your twilio sid"
    auth_token="your twilio auth token"

    twilio_number="your twilio number"
    number="your number"
    msg=str(classdict[classcount])

    client=Client(sid,auth_token)

    msg=client.messages.create(body=msg,from_=twilio_number,to=number)
    print(msg.body)

    return render_template('index.html', type="Patient is suffering from "+str(classdict[classcount]), bgvideo=bg_video , css=css)


@app.route("/model")
def model():
    return render_template('model.html',accuracy=acc,loss=loss,acc_loss_graphs=acc_loss_graphs,cm=cm,cr=cr)

if __name__ == "__main__":
    app.run()





