import streamlit as st

import time
import torchvision
from torchvision import  transforms 
import torch
from torch import no_grad

import requests


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.title("Object Detection using Faster_RCNN")
#  91 Classes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
len(COCO_INSTANCE_CATEGORY_NAMES)

@st.cache(persist=True)
def get_predictions(pred,threshold=0.8,objects=None ):
    """
    This function will assign a string name to a predicted class and eliminate predictions whose likelihood  is under a threshold 
    
    pred: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class yhat, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    thre
    """


    predicted_classes= [(COCO_INSTANCE_CATEGORY_NAMES[i],p,[(box[0], box[1]), (box[2], box[3])]) for i,p,box in zip(list(pred[0]['labels'].numpy()),pred[0]['scores'].detach().numpy(),list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes=[  stuff  for stuff in predicted_classes  if stuff[1]>threshold ]
    
    if objects  and predicted_classes :
        predicted_classes=[ (name, p, box) for name, p, box in predicted_classes if name in  objects ]
    return predicted_classes

def draw_box(predicted_classes,image,rect_th= 10,text_size= 3,text_th=3):
    """
    draws box around each object 
    
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface 
   
    """

    img=(np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8).copy()
    for predicted_class in predicted_classes:
   
        label=predicted_class[0]
        probability=predicted_class[1]
        box=predicted_class[2]
        print(box[0])
        cv2.rectangle(img, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])),(0, 10, 255), rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,label,(int(box[0][0]), int(box[0][1])),  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 
        cv2.putText(img,label+": "+str(round(probability,2)), (int(box[0][0]), int(box[0][1])),  cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0),thickness=text_th)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    del(img)
    del(image)

def save_RAM(image_=False):
    global image, img, pred
    torch.cuda.empty_cache()
    del(img)
    del(pred)
    if image_:
        image.close()
        del(image)


# Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image  pre-trained on COCO

model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()

for name, param in model_.named_parameters():
    param.requires_grad = False
print("done")

def model(x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat

half = 0.5
transform = transforms.Compose([transforms.ToTensor()])

threshold = st.slider("Select the prediction threshold",0.1,1.0)

select = st.selectbox("Select Image",['image1','image2','image3','image4','image5','image6'])

if select == 'image1':
    img_path='jeff_hinton.png'
    image = Image.open(img_path)
    image.resize([int(half * s) for s in image.size])

if select == 'image2':
    img_path='DLguys.jpeg'
    image = Image.open(img_path)
    image.resize([int(half * s) for s in image.size])

if select == 'image3':
    img_path='watts_photos2758112663727581126637_b5d4d192d4_b.jpeg'
    image = Image.open(img_path)
    image.resize([int(half * s) for s in image.size])

if select == 'image4':
    img_path='istockphoto-187786732-612x612.jpeg'
    image = Image.open(img_path)
    image.resize([int(half * s) for s in image.size])

if select == 'image5':
    img_path='cameraman.jpeg'
    image = Image.open(img_path)
    image.resize([int(half * s) for s in image.size])


if select == 'image6':
    img_path='slider-image-2.jpeg'
    image = Image.open(img_path)
    image.resize([int(half * s) for s in image.size])



col1,col2 = st.columns(2)


plt.imshow(np.array(image))
plt.show()

img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=threshold,)


with st.spinner("Detecting.."):
    time.sleep(2)
    st.success("Done!")
    with col1:
        st.header("Actual")
        st.image(image,width = 350)
        st.write(select)

    with col2:
        st.header("Predicted")
        draw_box(pred_thresh,img,rect_th= 2,text_size= 0.5,text_th=1)
    
        l1 = pred[0]['labels']
        l1.tolist()
        length = len(l1)
        score = pred[0]['scores']
        score.tolist()
        string =" "
        #    st.write(length)
        st.write("Predicted Lable :")
        for i in range(length):
            index=pred[0]['labels'][i].item()
            if score[i]>0.8:
                string1 = COCO_INSTANCE_CATEGORY_NAMES[index]
                string = "{} {},".format(string,string1)
               
        st.write(string)


del pred_thresh




