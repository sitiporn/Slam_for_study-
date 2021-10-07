import cv2
import  time
from display import Display
from featurextractor import FeatureExtractor
import numpy as np  
W = 1920 // 2 
H = 1080 // 2

F = 200  
disp = Display(W,H)
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
print(K)
f_est_avg = []
fe = FeatureExtractor(K)

def process_frame(img):
    
  img = cv2.resize(img,(W,H))
  matches = fe.extract(img)
  print("%d matches "%(len(matches)))
   
 # for p in kps:
 #     u, v = map(lambda x: int(round(x)), p.pt)
 #     cv2.circle(img,(u,v),color=(0,255,0),radius=3)
  # events = sdl2.ext.get_events()
  # cv2.imshow('image',img)
 # print(matches)
  for pt1 , pt2 in matches:
     u1, v1 = fe.denormalize(pt1)
     u2, v2 = fe.denormalize(pt2)
     cv2.circle(img,(u1,v1),color=(0,255,0),radius=3)
     cv2.line(img,(u1,v1),(u2,v2),color=(255,0,0))


  disp.paint(img)


if __name__ == '__main__':
    
    cap = cv2.VideoCapture("downtown.mp4")



    while cap.isOpened():
        
        ret, frame = cap.read() 
        if ret == True:   
            process_frame(frame)

        else:
            break
