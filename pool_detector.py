import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from scipy.io.wavfile import read as wavread
from scipy import signal
from scipy.fftpack import fft, fftfreq, ifft
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image

def gauss_classifier(X):
        N = X.shape[1]
        # get mean and covariances of data
        Mean = np.mean(X,axis=1,keepdims=True)
        Cov = (X-Mean).dot((X-Mean).T)
        Cov /= (N-1)
        # return both of them as a dict 
        g_dict = {'mean':Mean, 'cov':Cov}
        return g_dict

# COV = (1/K)(X - X_mean).(X - X_mean)^T
def PCA(X,n): 
	X_2 = X - np.mean(X,axis=1,keepdims=True)    							# removed the mean
	COV = (X_2.dot(X_2.T))/(X_2.shape[1]-1)        							# computed the covariance matrix  
	eigenvalues, eigenvecs = scipy.sparse.linalg.eigsh(COV,k=n)  		    # Got the eigenvectors and eigenvalues
	W = np.diag(1./(np.sqrt(eigenvalues))).dot(eigenvecs.T)                 # Lecture 5 - Slide 18 --> W = diag(eigenvalues^-1)*U^T
	return W
	
# Defining some 'magic' numbers and necessary lists/arrays
image_x = 840
image_y = 840
full_image = image_x * image_y
rgb = 3
real_dim = (image_x,image_y,rgb)
X_train = []
Y_train = []
X_test = []
im_row = []
im_col = []
pool_mean = None
pool_cov = None
notpool_mean = None
notpool_cov = None
pca_dims = 28
outer = 3
im_dims = 26
extra_notpool = 21
blue_const = [1,1.009,1.1]

pools = Image.open("pools.png")
not_pools = Image.open("not_pools.png")
mixed = Image.open("ekalismall.png")
testing = Image.open("ekalismall2.png")

plt.subplot(2,2,1), plt.imshow(pools), plt.title('Training data for pools'), plt.axis('off')
plt.subplot(2,2,2), plt.imshow(not_pools), plt.title('Training data for non-pools'), plt.axis('off')
plt.subplot(2,2,3), plt.imshow(mixed), plt.title('Provided satellite image'), plt.axis('off')
plt.subplot(2,2,4), plt.imshow(testing), plt.title('Provided testing image'), plt.axis('off')

pools = np.array(pools.resize((image_x,image_y)))[:,:,:rgb]
not_pools = np.array(not_pools.resize((image_x,image_y)))[:,:,:rgb]
mixed = np.array(mixed.resize((image_x,image_y)))[:,:,:rgb]
testing = np.array(testing.resize((image_x,image_y)))[:,:,:rgb]

def add_blue(X,x,y):
    blue_const = [1,1.005,1.1]
    X = np.reshape(X,(full_image,rgb))
    N = X.shape[0]
    for i in range(N):
        X[i] = blue_const * X[i]
    return np.reshape(X,(x,y,3))

def clipper(X):
    X = X[:-6]
    return X

def draw_pools(pred,im_row,im_col,blue,result,im_size): 
    for k in range(len_pool):
        blue[:, 0] = 100
        blue[:, 1] = 140
        blue[:, 2] = 231
        i = im_row[k]
        j = im_col[k]
        result[i:im_row[k]+im_size, j:im_col[k]+im_size] = blue
    plt.imshow(result), plt.title('Image with Marked Pools'), plt.axis('off'), plt.show()
    
pools = add_blue(pools,image_x,image_y)
testing = add_blue(testing,image_y,image_x)
result = testing.copy()
pools_size = int(pools.shape[1] / im_dims)
img_dim = int(image_x / im_dims)
blue = np.ones((pools_size,rgb))
max_pool = image_y - pools_size
max_pool2 = image_x - pools_size
max_npool = image_y - img_dim
max_npool2 = image_x - img_dim
row_npool = 0
row_pool = 0
col_pool = 0
col_npool = 0

for i in range(0, max_pool, pools_size):
    row_pool += 1
    for j in range(0, max_pool2, pools_size):
        col_pool += 1
        Y_train.append(1)
        row_st = i+outer
        row_end = i+pools_size-outer
        col_st = j+outer
        col_end = j+pools_size-outer
        X_train.append(pools[row_st:row_end, col_st:col_end])
#print(row_pool)
#print(col_pool)
X_train = clipper(X_train)
Y_train = clipper(Y_train)

for i in range(0, max_npool, img_dim):
    row_npool += 1
    for j in range(0, max_npool2, img_dim):
        col_npool +=1
        Y_train.append(0)
        row_st = i+outer
        row_end = i+img_dim-outer
        col_st = j+outer
        col_end = j+img_dim-outer
        X_train.append(not_pools[row_st:row_end, col_st:col_end])
#print(row_npool)
#print(col_npool)
X_train = clipper(X_train)
Y_train = clipper(Y_train)

row_ex = 0
col_ex = 0
for i in range(extra_notpool):
    row_ex +=1
    for j in range(max_npool2):
        col_ex +=1
        Y_train.append(0)
        row_st = i+outer
        row_end = i+img_dim-outer
        col_st = j+outer
        col_end = j+img_dim-outer
        X_train.append(mixed[row_st:row_end, col_st:col_end])
#print(row_ex)
#print(col_ex)  

X_train = np.array(X_train)
Y_train = np.array(Y_train)
image_ex = row_ex + col_ex
full_size = (X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[rgb])
image_pool = row_pool + col_pool
image_npool = row_npool + col_npool
X_train = np.reshape(X_train, full_size).T

for i in range(0, testing.shape[0] - img_dim, 2):
    for j in range(0, testing.shape[1] - img_dim, 2):
        im_row.append(i)
        im_col.append(j) 
        row_st = i+outer
        row_end = i+img_dim-outer
        col_st = j+outer
        col_end = j+img_dim-outer
        img = testing[row_st:row_end, col_st:col_end]
        img = np.reshape(img, (img.shape[0] * img.shape[1] * img.shape[2]))
        X_test.append(img)
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))
test = image_pool + image_npool
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))

W_train = PCA(X_train, pca_dims)
Z_train = W_train.dot(X_train - np.mean(X_train,axis=1,keepdims=True))

notpool_data = Z_train[:,Y_train == 0]
pool_data = Z_train[:,Y_train == 1]

gauss_pool = gauss_classifier(pool_data)
gauss_notpool = gauss_classifier(notpool_data)

def probs(X,g):
    M = X.shape[0]
    X_2 = X - g['mean']
    IC =(g['cov'])
    probs = (-1*np.log(np.linalg.det(IC)) + M*np.log(2*np.pi) + np.sum((IC).dot(X_2)*X_2,axis=0))*(-1/2)
    return probs

Z_test = W_train.dot((X_test - np.mean(X_test, axis = 1, keepdims = True)).T)

pool_probs = probs(Z_test, gauss_pool)
notpool_probs = probs(Z_test, gauss_notpool)
pred = (pool_probs > notpool_probs).astype(int)
im_row = np.array(im_row)
im_col = np.array(im_col)

pred = (pool_probs > notpool_probs).astype(int)
pools_idx = np.where(pred==1)
pred = pred[pools_idx]
len_pool = len(pred)

im_row = im_row[pools_idx]
im_col = im_col[pools_idx]

draw_pools(pred,im_row,im_col,blue,result,img_dim)

