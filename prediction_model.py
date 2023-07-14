from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os,cv2
from keras.models import load_model
import numpy as np
import tensorflow_hub as hub

file_path = ''
path=r'D:\Crop_Disease\static\potatoes.h5'
# model = load_model(
#        (path),
#        custom_objects={'KerasLayer':hub.KerasLayer})

model=load_model(r'D:\Crop_Disease\static\potatoes.h5')
 

def prediction(file_path):
    results=['Late Blight','Early Blight']
    img=cv2.imread(file_path)
    img=cv2.resize(img, (256,256),interpolation = cv2.INTER_NEAREST)
    img=np.expand_dims(img, axis=0)
    result=model.predict(img)
    result=results[np.argmax(result)]
    return result

# FLASK APPLICATION
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/trackCalorie/', methods=['POST','GET'])
def trackCalorie():
    return render_template('prediction.html')

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
      f = request.files['file']
      file_path=os.path.join(r'D:\Crop_Disease\static',secure_filename(f.filename))
      f.save(file_path)
      img_src=file_path
      #Call prediction to predict the output
      result=prediction(file_path)
      
      return render_template('prediction.html', prediction_text="The disease is {}".format(result), img_source=img_src)




if __name__=="__main__":
    app.run(debug=True)