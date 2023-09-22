
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt

from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
import cv2
import numpy as np
import pickle
import os
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import gtts
from playsound import playsound

main = tkinter.Tk()
main.title("A Deep Neural Framework for Continuous Sign Language Recognition by Iterative Training") #designing main screen
main.geometry("1300x1200")

sign = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','S','T','U','V','W','X','Y']

global filename
global model
global X,Y
global labels
def readTrainDataset(path,i):
    print(path)
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            #print(dataset[i]+"/"+directory[j])
            img = cv2.imread(path+"/"+directory[j])
            img = cv2.resize(img, (100,100))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(100,100,3)
            X.append(im2arr)
            Y.append(i)

def readTestDataset(path,i):
    print(path)
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            #print(dataset[i]+"/"+directory[j])
            img = cv2.imread(path+"/"+directory[j])
            img = cv2.resize(img, (100,100))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(100,100,3)
            X.append(im2arr)
            Y.append(i)            


def uploadDataset(): 
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    

def Preprocessing():
    global X,Y
    global labels
    labels = 24
    text.delete('1.0', END)
    if os.path.exists('model/xtrain.txt.npy'):
        X = np.load("model/xtrain.txt.npy")
        Y = np.load("model/ytrain.txt.npy")
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        text.insert(END,"Total images found in signum database are : "+str(X.shape[0])+"\n")
        text.insert(END,"Total Signs founds in signum database are : "+str(labels)+"\n")
        text.insert(END,"Preprocessing completed\n")
    else:
        root = 'SignumDataset/TrainData'
        for i in range(len(dataset)):
            readTrainDataset(root+"/"+dataset[i],i)

        root = 'SignumDataset/TestData'
        for i in range(len(dataset)):
            readTestDataset(root+"/"+dataset[i],i)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/xtrain.txt",X)
        np.save("model/ytrain.txt",Y)
        text.insert(END,"Total images found in signum database are : "+str(X.shape[0])+"\n")
        text.insert(END,"Total Signs founds in signum database are : "+str(labels)+"\n")
        text.insert(END,"Preprocessing completed\n")

        
def CNNTrainModel():
    global model
    text.delete('1.0', END)
    if os.path.exists('model/model.h5'):
        model = load_model('model/model.h5')
        print(model.summary())
    else:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(X.shape[1],X.shape[2],X.shape[3])))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Bidirectional(LSTM(64, return_sequences=True,input_shape=input_shape)))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(24))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        X = X.astype('float32')
        X = X/255
        hist = model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        model.save_weights('model/weights.h5')
        model.save('model/model.h5')
        f = open('model/wer.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        print(model.summary())
    text.insert(END,"CNN BiLSTM RNN model generated. See black console to view CNN layers\n")        
    f = open('model/wer.pckl', 'rb')
    loss = pickle.load(f)
    f.close()
    loss = loss['loss']
    text.insert(END,"CNN BiLSTM WER : "+str(loss[9]*100))



def signRecognize():
    videofile = filedialog.askopenfilename(initialdir="video")
    video = cv2.VideoCapture(videofile)
    i=0
    while(True):
        ret, frame = video.read()
        if ret == True:
            img = cv2.resize(frame, (100,100))
            img = np.array(img)
            img = img.reshape(100,100,3)
            img = img.astype('float32')
            img = img/255
            X = []
            X.append(img)
            X = np.asarray(X)
            predict = model.predict(X)
            predict = np.argmax(predict)
            recognize = sign[predict]
            frame = cv2.resize(frame,(400,400))
            cv2.putText(frame, 'Sign Recognized as : '+recognize, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)
            cv2.imshow('Recognization Output', frame)
            tts = gtts.gTTS(recognize)
            tts.save("play/"+str(i)+".mp3")
            playsound("play/"+str(i)+".mp3",True)
            print(i)
            os.remove("play/"+str(i)+".mp3")
            i = i + 1
            if cv2.waitKey(350) & 0xFF == ord('q'):
                break                
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    

def graph():
    f = open('model/wer.pckl', 'rb')
    loss = pickle.load(f)
    f.close()
    WER = loss['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.plot(WER, 'ro-', color = 'indigo')
    plt.legend(['WER'], loc='upper left')
    plt.title('WER (Word Error Rate) Graph')
    plt.show()

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='A Deep Neural Framework for Continuous Sign Language Recognition by Iterative Training')
title.config(bg='deep sky blue', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Signum Sign Language Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=Preprocessing)
processButton.place(x=400,y=550)
processButton.config(font=font1) 

featuresButton = Button(main, text="Train CNN-BiLSTM Deep Neural Networks", command=CNNTrainModel)
featuresButton.place(x=620,y=550)
featuresButton.config(font=font1) 

svmButton = Button(main, text="Upload Video & Recognize Signs", command=signRecognize)
svmButton.place(x=50,y=600)
svmButton.config(font=font1) 

classifyButton = Button(main, text="WER (Word Error Rate) Graph", command=graph)
classifyButton.place(x=350,y=600)
classifyButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=620,y=600)
exitButton.config(font=font1) 

main.config(bg='LightSteelBlue3')
main.mainloop()
