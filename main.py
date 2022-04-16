import cv2,os
import numpy as np
from jsontflite import jamestf

pathModel = 'garbage.tflite'

model = jamestf(pathModel)

classes = ['cardboard','glass','metal','paper','plastic','trash']

for cl in os.listdir('test'):
    pathdir = os.path.join('test',cl)
    for imgname in os.listdir(pathdir):
        img = cv2.imread(os.path.join(pathdir,imgname))
        x = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)
        inference = model.inference(x,(224,224))
        indexclasses = np.argmax(inference)
        print(cl,classes[indexclasses])

       

