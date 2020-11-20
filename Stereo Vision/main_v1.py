import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
window_size = 100
img_right = cv2.imread('images/right1.jpg') #right1.jpg #right.png
img_right = cv2.resize(img_right, (500,500), fx=1, fy=1)
img_right_gray = cv2.cvtColor(copy.copy(img_right),cv2.COLOR_BGR2GRAY)
img_right_rgb = cv2.cvtColor(copy.copy(img_right),cv2.COLOR_BGR2RGB)
img_left = cv2.imread('images/left1.jpg') #left1.jpg #left.png
img_left = cv2.resize(img_left, (500,500), fx=1, fy=1)
img_left_gray = cv2.cvtColor(copy.copy(img_left),cv2.COLOR_BGR2GRAY)
img_left_rgb = cv2.cvtColor(copy.copy(img_left),cv2.COLOR_BGR2RGB)
print(img_left_gray.shape,img_right_gray.shape)
# cv2.imshow("img right",img_right_gray)
# cv2.imshow("img left",img_left_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img_left_pil = Image.fromarray(img_left_gray)
plt.imshow(img_left_pil)
point = tuple(np.array((plt.ginput(1)[0]),int))
print(point)
end_point = (point[0]+window_size,point[1]+window_size)
patch_left = img_left_gray[int(point[1]):int(point[1])+window_size,int(point[0]):int(point[0])+window_size]
#print(patch_left.shape)

# cv2.imshow("patch left",patch_left)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
strip_right = img_right_gray[int(point[1]):int(point[1])+window_size,:]
plt.imshow(strip_right)
plt.show()

def depth():
    stereo = cv2.StereoBM_create(16,blockSize=5)

    disparity = stereo.compute(img_left_gray,img_right_gray)
    norm = cv2.normalize(disparity,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    cv2.imshow("norm",norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plt.imshow(norm)
    # plt.show()

depth()
def SQD(patch_left,strip_right):
    min_diff = float('inf')
    best = 0
    searching_strip = strip_right.shape[1] - window_size
    for x in range(searching_strip):
        patch_right = strip_right[:,x:x+window_size]
        diff = patch_right-patch_left
        SQD =  np.sum(diff**2)
        
        if(SQD < min_diff):
            min_diff = SQD
            best = x
    return best,min_diff

best_SQD,min_diff = SQD(patch_left,strip_right)
def SAD(patch_left,strip_right):
    min_diff = float('inf')
    best = 0
    searching_strip = strip_right.shape[1] - window_size
    for x in range(searching_strip):
        patch_right = strip_right[:,x:x+window_size]
        diff = np.absolute(patch_right-patch_left)
        SAD =  np.sum(diff)
        if(SAD < min_diff):
            min_diff = SAD
            best = x
    return best,min_diff

best_SAD,min_diff = SAD(patch_left,strip_right)
print(best_SAD)
draw_patch_right = cv2.rectangle(copy.copy(img_right_gray), (best_SAD,point[1]),(best_SAD+window_size ,end_point[1]) , (0,0,0 ) , 5) 
draw_patch_left = cv2.rectangle(copy.copy(img_left_gray), point, end_point, (0, 0,0 ) , 5) 
cv2.imshow("patch left ",draw_patch_left)
cv2.imshow("patch right",draw_patch_right)
cv2.waitKey(0)
cv2.destroyAllWindows()

def dynamic_prog(left_scan,right_scan):
    segma = 2
    c0 = 1
    
    size_left = left_scan.shape[0]
    size_right = right_scan.shape[0]
    d =  np.zeros((size_left,size_right))
    D =  np.full((size_left,size_right),np.inf)
    disparity = np.zeros((size_left,size_right))
    disp = np.zeros((size_right))

    print("left = ",left_scan.shape,"right = ",right_scan.shape)
    d[0,0] = ((left_scan[0] -right_scan[0])**2)/ segma**2

    D[0,0] = d[0,0]
    for i in range(0,size_left):
        for j in range(0,size_right):
            if(i == 0 and j == 0):
                continue
            d[i,j] = ((left_scan[i] -right_scan[j])**2)/ segma**2
            D[i,j] = np.amin(np.array([D[i-1,j-1]+d[i,j],D[i-1,j]+c0,D[i,j-1]+c0]))

    i = size_left-1
    j = size_right-1
    while(1):
        ar = [D[i-1,j-1],D[i-1,j],D[i,j-1]]
        min_val = min(ar)
        
        min_index = ar.index(min_val)
        if(min_index == 0):
            
            i = i-1
            j = j-1      
        elif(min_index == 1):
            i = i-1   
        elif(min_index ==2):
            j = j-1
            
        disparity[i,j] = 255
        # count -=1
        if(i == 0 and j == 0):
            break
        
    return D,disparity

def disparity(strip_left,strip_right,window_size):
    block = 0
    searching_strip = strip_left.shape[1] - window_size
    # plt.figure(figsize=(40,20),facecolor='b',edgecolor='r',linewidth=50)

    i = 0
    box_blocks = int(searching_strip/window_size)
    print("box blocks = ", box_blocks)
    for block in range(0,int(searching_strip),window_size):
        patch_left = strip_left[:,block:block+window_size]
        x_right,min_diff = SQD(patch_left,strip_right)
        draw_patch_left = cv2.rectangle(copy.copy(strip_left), (block,0),(block+window_size ,window_size) , (0,0,0 ) , 5) 
        draw_patch_right = cv2.rectangle(copy.copy(strip_right), (x_right,0),(x_right+window_size ,window_size) , (0,0,0 ) , 5) 


        plt.subplot2grid((box_blocks+1,2), (i,0))
        plt.imshow(draw_patch_left)
        
        
        plt.subplot2grid((box_blocks+1,2), (i,1))
        plt.imshow(draw_patch_right)
        i += 1
    plt.show()


disp_left = img_left_gray[int(point[1]):int(point[1])+window_size,:]
disp_right = img_right_gray[int(point[1]):int(point[1])+window_size,:]




x_left = disp_left[0,:]
x_right = disp_right[0,:]
# D,disp = dynamic_prog(x_left,x_right)
# plt.imshow(disp)
# plt.show()
print("D = \n",D,"\nDisp = \n",disp)
disparity(disp_left,disp_right,window_size)
