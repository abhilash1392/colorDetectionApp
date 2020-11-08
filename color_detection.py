import argparse
import cv2 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help='ImagePath')
args=vars(ap.parse_args())
img_path=args['image']
#Reading  image with OpenCV 
img=cv2.imread(img_path)
clicked=False 
b=0
g=0
r=0
xpos=0 
ypos=0
# Reading csv file with pandas and giving names to each column 
index=['color','color_name','hex','R','G','B']
csv=pd.read_csv('color.csv',names=index,header=None)
def draw_function(event,x,y,flags,param):
    if event ==cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos,clicked
        
        clicked=True 
        xpos=x 
        ypos=y 
        b,g,r=img[y,x]
        b=int(b)
        g=int(g)
        r=int(r)


def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d=abs(R-int(csv.loc[i,"R"]))+abs(G-int(csv.loc[i,"G"]))+abs(B-int(csv.loc[i,"B"]))
        if (d<=minimum):
            minimum=d
            cname=csv.loc[i,"color_name"]
    return cname

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_function)

while (1):
    cv2.imshow("image",img)
    if (clicked):
        cv2.rectangle(img,(20,20),(750,60),(b,g,r),-1)
        text=getColorName(r,g,b)+'R='+str(r)+' G='+str(g)+' B='+str(b)
        cv2.putText(img,text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)
        if (r+g+b>=600):
            cv2.putText(img,text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)
        clicked=False 
    
    if cv2.waitKey(20) & 0xFF==27:
        break

cv2.destroyAllWindows()

