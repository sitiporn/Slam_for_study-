import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from skimage.transform import FundamentalMatrixTransform
np.set_printoptions(suppress=True)

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x,np.ones((x.shape[0],1))], axis=1)
def extractRt(E):

    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U, d, Vt = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
    
    R = np.dot(np.dot(U,W.T),Vt)
 
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0  
        R = np.dot(np.dot(U,W.T),Vt)
    
    t = U[:, 2]

    return (R,t)
 
class FeatureExtractor(object):
  
    def __init__(self,K):

        self.orb = cv.ORB_create() 
        self.bf =  cv.BFMatcher(cv.NORM_HAMMING)
        self.last = None
        self.K = K  
        self.Kinv = np.linalg.inv(self.K)
    
    def nomialize(self,pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:,0:2]


    def denormalize(self,pt):
        ret = np.dot(self.K, np.array([pt[0],pt[1],1.0]))
        
        return int(round(ret[0])), int(round(ret[1]))
    def extract(self,img):
        #detection  
        feats = cv.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel = 0.01, minDistance =3)
        
        #extraction 
        kps = [cv.KeyPoint(x=f[0][0],y=f[0][1],_size =20) for f in feats]
        kps, des = self.orb.compute(img,kps)
        
        #matching
        
        ret = []

        if self.last is not None:
                     # detect k-nearest between current and previous  
                     # current -> des , last -> self.last['des']
            matches = self.bf.knnMatch(des,self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                  kp1 =  kps[m.queryIdx].pt
                  kp2 =  self.last['kps'][m.trainIdx].pt
                  ret.append((kp1,kp2))
                # filter 
        '''
        fundametal matrix can check that point are correspondes on each others.

        '''
        if len(ret) > 0:
            ret = np.array(ret)
            ret[:,0,:] = self.nomialize(ret[:,0, :]) 
            ret[:,1,:] = self.nomialize(ret[:,1, :])
          
           # subtract to move to 0
           # ret[:,:,0] -= img.shape[0] //2
           # ret[:,:,1] -= img.shape[1] //2

            print("ret.shape:",ret.shape)
            model , inliers = ransac((ret[:,0],ret[:,1]),EssentialMatrixTransform, min_samples =8,residual_threshold=0.005,max_trials=100)
            
            ret = ret[inliers]
            R , t = extractRt(model.params)
            print(R,t)
                     # print(v) 
            print("sum(inliners) , len(inliers) :",sum(inliers) ,len(inliers))

        self.last = {'kps':kps,'des':des} 

        
        return ret



