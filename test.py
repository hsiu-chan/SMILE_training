import numpy as np
from keras.models import load_model

def tooth_predict(img_path,model_path):
    import cv2
    img=cv2.imread(img_path)
    w,h,d =img.shape()
    img=cv2.resize(img,(512,512))
    model = load_model(model_path)
    pred=model.predict(np.expand_dims(img,axis=0))
    return cv2.resize(pred,(w,h))

    

if __name__ == '__main__':
    import cv2
    img_path=''
    model_path='model/cnn.h5'
    cv2.imshow(tooth_predict(img_path,model_path))
