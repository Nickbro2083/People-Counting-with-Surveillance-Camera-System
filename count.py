import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *


model=YOLO('People.pt')

area1=[(312,388),(289,390),(474,469),(497,462)]
area2=[(279,392),(250,397),(423,477),(454,469)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('walk.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
tracker=Tracker()   

people_enter={}
enter=set()

people_exit={}
exit=set()

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue


    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a = results[0].boxes.data

    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
            x3,y3,x4,y4,id=bbox
            #vào
            results=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
            if results >=0:
                people_enter[id]=(x4,y4)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
            if id in people_enter:
                results1=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
                if results1>=0:
                    cv2.rectangle(frame,(x3,y3),(x2,y2),(0,255,0),2)
                    cv2.circle(frame,(x4,y4),5,(255,0,255),-1)
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,0,0),1)
                    enter.add(id)
            #ra
            results2=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
            if results2 >=0:
                people_exit[id]=(x4,y4)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
            if id in people_exit:
                results3=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
                if results3>=0:
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                    cv2.circle(frame,(x4,y4),5,(255,0,255),-1)
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,0,0),1)
                    exit.add(id)
    #Khung 1
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    #Khung 2
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    #Khung đếm
    v=(len(enter))
    r=(len(exit))
    cv2.putText(frame,f'vao: {v}',(60,80),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
    cv2.putText(frame,f'ra: {r}',(60,120),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

