import cv2 as cv
from ultralytics import YOLO
import math
from sort import *

classNames=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits=[216,300,390,300]
model=YOLO('../yolo-weights/yolov8l.pt')
counter=[]

cap=cv.VideoCapture('data/traffic.mp4')
mask=cv.imread('data/mask.png')
while True:
    rect,frame=cap.read()
    frame=cv.resize(frame,(780,420))
    mask=cv.resize(mask,(780,420))
    maskRegion=cv.bitwise_and(frame,mask)
    results=model(maskRegion)

    detections=np.empty((0,5))

    for r in results:
        for box in r.boxes:
            x1,y1,w,h=box.xywh[0]
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2,w,h=int(x1),int(y1),int(x2),int(y2),int(w),int(h)
            conf=math.ceil(box.conf)
            cls=int(box.cls[0])
            cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            cv.putText(frame,f'{classNames[cls]}',(x1,y1),cv.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),1) 
            cv.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),2)
            currentArray=([x1,y1,x2,y2,conf])
            detections = np.vstack((detections, currentArray))
            cx,cy=int(x1+w/2),int(y1+h/2)
            cv.circle(frame,(cx,cy),2,(0,0,255),2)
    resultsTracker=tracker.update(detections)

    for result in resultsTracker:
        x1,y1,w,h=box.xywh[0]
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2,id=int(x1),int(y1),int(x2),int(y2),int(id)
        cx,cy=int(x1+w/2),int(y1+h/2)
        # cv.putText(frame,f'{id}',(x1,y1),cv.FONT_HERSHEY_DUPLEX,0.8,(0,0,255),2)
        if limits[0]<cx<limits[2] and limits[1]-10<cy<limits[1]+10:
            if counter.count(id)==0:
                counter.append(id)




    cv.rectangle(frame,(0,0),(320,60),(255,0,255),-1)            
    cv.putText(frame,f'Number of cars: {len(counter)}',(8,35),cv.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
    cv.imshow('Frame',frame)

    if cv.waitKey(10) & 0xFF==ord('d'):
        break

cv.destroyAllWindows()

