import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
def warpPerspective(img, M, dsize):
    R,C = dsize
    dst = np.zeros((img.shape[1]+1,R))
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            res = np.dot(M, [i,j,1])
            i2,j2,_ = (res / res[2]).astype(int)
            dst[j2,i2] = img[j,i]
    
    return dst

def homography(pts_src_t,pts_dst_t,type):
    if(type=="auto"):
        pts_src = np.zeros((pts_src_t.shape[0],pts_src_t.shape[2]))
        pts_dst = np.zeros((pts_dst_t.shape[0],pts_src_t.shape[2]))
        pts_src =  pts_src_t[:,0,:]
        pts_dst =  pts_dst_t[:,0,:]
    else:
        pts_src =  pts_src_t
        pts_dst =  pts_dst_t
    A = None
    B = None
    for i in range(pts_src.shape[0]):
        p1 = pts_src[i, :]
        p2 = pts_dst[i, :]
        new_stack = np.zeros((2, 8))
        new_stack[0, 0:3] = np.append(p1, 1.0) #1st iter p1 = [1 5]
        
        new_stack[0, 6:8] = np.array([-p1[0]*p2[0], -p1[1]*p2[0]])
        new_stack[1, 3:6] = np.append(p1, 1.0)
        new_stack[1, 6:8] = np.array([-p1[0]*p2[1], -p1[1]*p2[1]])
        if A is None:
            A = new_stack
            B = p2
        else:
            A = np.vstack((A, new_stack))
            B = np.concatenate((B, p2))
    # print("A = ",A)
    # print("B = ",B)
    sol = np.linalg.lstsq(A, B)[0]
    result = np.append(sol, 1.0).reshape((3,3))
    return result

img_ = cv2.imread('right1.jpg')
#img_ = cv2.imread('right1.jpg')
#img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img = cv2.imread('left1.jpg')
#img = cv2.imread('left1.jpg')
#img = cv2.resize(img, (0,0), fx=1, fy=1)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
# find key points
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.3*n.distance:
        good.append(m)
draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)
img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
# cv2.imshow("original_image_drawMatches.jpg", img3)
MIN_MATCH_COUNT = 4
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # print("src points = ",src_pts)
    # print("dst points = ",dst_pts)
    M = homography(src_pts, dst_pts,"auto")
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    # cv2.imshow("img.jpg", img2)
else:
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
    # img1_pil = array(img1_pil)
    plt.imshow(img1_pil)
    src_pts = plt.ginput(4)
    # img2_pil = array(img2_pil)
    plt.imshow(img2_pil)
    dst_pts = plt.ginput(4)
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    M = homography(src_pts, dst_pts,"manual")
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    # cv2.imshow("img.jpg", img2)

dst = warpPerspective(img1,M,(img.shape[1] + img_.shape[1], img.shape[0]))
dst[0:img.shape[0],0:img.shape[1]] = img2
# cv2.imshow("img.jpg", dst)
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
plt.imshow(dst)
plt.show()
#cv2.imshow("output.jpg", trim(dst))
#cv2.imwrite("result1.jpg", trim(dst))
cv2.waitKey(0)
cv2.destroyAllWindows()