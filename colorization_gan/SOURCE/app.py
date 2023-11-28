#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as
#published by the Free Software Foundation, either version 3 of the
#License, or (at your option) any later version.
#Smooth Contours is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU Affero General Public License for more details.
#You should have received a copy of the GNU Affero General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.

import sys
import argparse
import os
import tensorflow as tf
import config as config
import numpy as np
import cv2
from keras import applications
from keras.models import load_model

import gradio as gr


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)

def reconstruct_no(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2RGB)
    return result

def _load_model():
    save_path = os.path.join(config.MODEL_DIR, config.PRETRAINED)
    print(save_path)
    global colorizationModel
    colorizationModel = load_model(save_path)
    # this is key : save the graph after loading the model
    global graph
    graph = tf.get_default_graph()

def sample_images(img):
    labimg = cv2.cvtColor(cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
    labimg = np.reshape(labimg[:,:,0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1))
    labimg = np.expand_dims(labimg, 0) / 255
    labimg = np.tile(labimg, [1,1,1,3])
    
    labimg_ori = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 0]
    labimg_ori = np.expand_dims(labimg_ori, axis=2)
    labimg_ori = labimg_ori.astype(np.float64) / 255 

    with graph.as_default():
        predY, _ = colorizationModel.predict(labimg)
    height, width, channels = img.shape
    predictedAB = cv2.resize(deprocess(predY[0]), (width,height))
    predResult= reconstruct_no(deprocess(labimg_ori), predictedAB)

    return predResult



if __name__ == '__main__':
    _load_model()
    demo = gr.Interface(sample_images, gr.Image(), "image")
    demo.launch(share=True, enable_queue=False)
