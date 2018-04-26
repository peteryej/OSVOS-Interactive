import os
import sys
import time
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import osvos
from dataset import Dataset
import evalsegm as es


gpu_id = 0

seq_name = 'dancing' #'blackswan' #'drone' #"PersonFinder_large" #"CarChaser_large" #"car-turn" "bmx-trees" "man-bike"
file_name = seq_name+"singledotr10-20ite"

#test_folder = os.path.join('evalDatasets', seq_name, 'test')
#test_frames = sorted(os.listdir(test_folder))
#test_imgs = [os.path.join('evalDatasets', seq_name, 'test', frame) for frame in test_frames]

test_folder = os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)
test_frames = sorted(os.listdir(test_folder))
test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]

GT_path = None #os.path.join('DAVIS', 'Annotations', '480p', seq_name)
GT_imgs = []

# Train parameters
parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
logs_path = os.path.join('models', file_name)

learning_rate = 1e-8
side_supervision = 3

# init the OSVOS model
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)

        sess = osvos.train_init(parent_path, side_supervision, learning_rate, logs_path, global_step, iter_mean_grad=1, ckpt_name=seq_name)




# display images
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
plt.show()


class onlineTrainPredict:
    def __init__(self, image):
        self.image = image
        
    def connect(self):
        fig.canvas.callbacks.connect('button_press_event', self.on_click)
        

    def on_click(self, event):
        if event.inaxes is not None:
            print event.xdata, event.ydata
            dotRadius=10
            if event.button == 3:
                circleColor = 'green'
            else:
                circleColor = 'yellow'
            circle = mpatches.Circle((event.xdata, event.ydata), radius=dotRadius, color=circleColor, ec="none")
            ax.add_patch(circle)
            plt.draw()
            self.makeAnnotationTrain(event.xdata, event.ydata, dotRadius, circleColor, self.image) 
        else:
            print 'Clicked ouside axes bounds but inside plot window'

            
    def makeAnnotationTrain(self, x, y, r, color, train_image):
        global sess
        train_image = np.array(Image.open(train_image), dtype=np.uint8)
        im = Image.new('RGB', (train_image.shape[1], train_image.shape[0]), (0, 0, 0)) 
        draw = ImageDraw.Draw(im) 
        if color == 'green':
            circleFill = (50,0,0)
        else:
            circleFill = (1,0,0)
        
        draw.ellipse((x-r,y-r, x+r, y+r), fill=circleFill)
        train_label = np.array(im)[:,:,0]
        #train_label = np.select([train_label==0], [120])
        #if secondClick:
        #    train_image = processTrainImg(train_image)
        start = time.time()
        sess = osvos.train_run(sess, train_image, train_label)
        duration = time.time()-start
        print "training time is %.4f" % duration
        """
        mask, im_over = self.maskOverlay(train_image)
        
        if color == 'green':
            sess = osvos.train_run(sess, train_image, train_label)
            self.maskOverlay(train_image, color)
            
        ax.imshow(im_over.astype(np.uint8))
        """

        
    def maskOverlay(self, image, GTImage=None):
        img = np.array(Image.open(image), dtype=np.uint8)
        start = time.time()
        # predict on the current image
        mask = osvos.test_run(sess, img)  # boolean mask
        
        duration = time.time()-start
        print "prediction time is %.4f" % duration

        #make mask overlay on image
        if np.max(mask) is not 0:
            mask = mask/np.max(mask)
        im_over = np.ndarray(img.shape)
        im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
        im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
        im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
        
        ax.imshow(im_over.astype(np.uint8))   
        
        if GTImage is not None:
            iou = es.mean_IU(mask, GTImage)
            print "miou is %.4f" % iou
            
        
        

        
        
        
overlay_color = [255, 0, 0]
transparency = 0.6


def getGTImages(GTPath):
    GT_frames = sorted(os.listdir(GTPath))
    global GT_imgs 
    for img in GT_frames:
        GT_imgs.append(np.array(Image.open(os.path.join(GTPath, img))))
    
if GT_path is not None:
    getGTImages(GT_path)

    
for i in range(len(test_frames)):	
    print i
    #test_img = test_folder+ '/'+str(i)+'.jpg'
    test_img = test_imgs[i]
    trainPred = onlineTrainPredict(test_img)
    trainPred.connect()
    trainPred.maskOverlay(test_img)
    #trainPred.maskOverlay(test_imgs[i], GT_imgs[i])

    plt.pause(0.00001)
    plt.cla()

    
    
    
    











