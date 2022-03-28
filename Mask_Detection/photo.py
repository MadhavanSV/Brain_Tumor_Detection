import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from PIL import Image
import os
from os import listdir, makedirs
from os.path import isfile, join
import getpass
import time

def maskdetection(path):
    #load the model
    model=load_model('Mask_detector_model.h5')
	#loading the cascades
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	#webcam face recognition
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H:%M:%S',t)
    username = getpass.getuser()
    destination = os.path.join('/home/' + username +'/Pictures/Mask Detection')
    check= os.path.isdir(destination)
    foldername = os.path.basename(os.path.normpath(path))
    if not check:
      makedirs(destination)
      destpathmask= os.path.join(destination,foldername)
      check_maskpath = os.path.isdir(destpathmask)
      with_mask=os.path.join(destpathmask , "Mask")
      without_mask = os.path.join(destpathmask , "No Mask")
      if not check_maskpath:
        makedirs(destpathmask)
        makedirs(with_mask)
        makedirs(without_mask)
      else:
        os.rename(destpathmask, destpathmask+ " "+ timestamp)
        os.makedirs(destpathmask) 
        makedirs(with_mask)
        makedirs(without_mask)   
    else:
        destpathmask= os.path.join(destination,foldername)
        check_maskpath = os.path.isdir(destpathmask)
        with_mask=os.path.join(destpathmask , "Mask")
        without_mask = os.path.join(destpathmask , "No Mask")
        if not check_maskpath:
           makedirs(destpathmask)
           makedirs(with_mask)
           makedirs(without_mask)
        else:
            os.rename(destpathmask, destpathmask+ " "+ timestamp)
            os.makedirs(destpathmask)
            makedirs(with_mask)
            makedirs(without_mask)
    video_capture=cv2.VideoCapture(path)
    seconds = 0.00
    width= int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videopath = os.path.join(destpathmask,foldername)
    writer= cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
    while(video_capture.isOpened()):
        ret,frame=video_capture.read()
        if ret == True:
            faces=face_cascade.detectMultiScale(frame,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                face=frame[y:y+h,x:x+w]
                cropped_face=face
                if type(face) is np.ndarray:
                    face=cv2.resize(face,(224,224))
                    im=Image.fromarray(face,'RGB')
                    img_array=np.array(im)
                    img_array=np.expand_dims(img_array,axis=0)
                    pred=model.predict(img_array)
                    time_milli = video_capture.get(cv2.CAP_PROP_POS_MSEC)
                    seconds_1 = time_milli / 1000
                    if seconds_1 != seconds:
                        if(pred[0][0]>=1):
                            prediction='Mask'
                            cv2.putText(cropped_face,prediction,(5,5),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                            time_milli = video_capture.get(cv2.CAP_PROP_POS_MSEC)
                            seconds = time_milli / 1000
                            filename = 'Frame_with_mask {} .jpg'.format(seconds)
                            # Save the images in given path
                            cv2.imwrite(os.path.join(with_mask,filename), frame)
                        else:
                            prediction='No Mask'
                            cv2.putText(cropped_face,prediction,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                            time_milli = video_capture.get(cv2.CAP_PROP_POS_MSEC)
                            seconds = time_milli / 1000
                            filename = 'Frame_without_mask {} .jpg'.format(seconds)
                            # Save the images in given path
                            cv2.imwrite(os.path.join(without_mask,filename), frame)      
                    else:
                        break    
                else:
                    cv2.putText(frame,'No Face Found',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            writer.write(frame)
            #count += 1
        else:
            break
    video_capture.release()
    writer.release()
    cv2.destroyAllWindows()


