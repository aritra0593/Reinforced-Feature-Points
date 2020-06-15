#!/usr/bin/env python
# coding: utf-8




import numpy as np
import cv2 as cv
import h5py
import math
import argparse
import os
import sys
import pickle
from network import SuperPointFrontend
import torch
import torch.nn as nn
import torch.nn.functional as F


    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(-x) / np.sum(np.exp(-x), axis=0)




def parsing():
    if '-f' in sys.argv:
        sys.argv.remove('-f')

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('dataset', type=str, default='reichstag',
      help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--weights_path', type=str, default='weights/baseline_mixed_loss.pth',
      help='Path to pretrained weights file (default: baseline_mixed_loss.pth).')
    parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--threshold', type=float, default=0.001, help='inlier threshold')
    parser.add_argument('--ratio', type=float, default=1.0, help='lowes ratio test')
    parser.add_argument('--vthreshold', type=float, default=0.5, help='visibility threshold')
    parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
    opt = parser.parse_args()
    opt.cuda = True
    return opt


opt = parsing()
nfeatures = 2000 # Number of keypoints to sample
inlierThreshold = opt.threshold 
counter = 1000 #number of images to test

print('==> Loading pre-trained network.')
# This class runs the SuperPoint network and processes its outputs.
fe = SuperPointFrontend(weights_path=opt.weights_path,nms_dist=opt.nms_dist,conf_thresh=opt.conf_thresh,nn_thresh=opt.nn_thresh,cuda=opt.cuda)
print('==> Successfully loaded pre-trained network.')


print('Using dataset: ', opt.dataset)




data_dir = 'datasets/' + opt.dataset + '/test/' #directory of the images for the testing
out_dir = 'output/' + opt.dataset + '/'  #directory for keeping the output results text files 
if not os.path.isdir(out_dir): os.makedirs(out_dir)
    
    
img_db = 'images.txt'
vis_db = 'visibility.txt'
cal_db = 'calibration.txt'

img_db = open(data_dir + img_db, 'r')
vis_db = open(data_dir + vis_db, 'r')
cal_db = open(data_dir + cal_db, 'r')

img_files = img_db.readlines()
vis_files = vis_db.readlines()
cal_files = cal_db.readlines()

img_db.close()
vis_db.close()
cal_db.close()

cal_db = {}

#creating the database of ground-truth relative poses
for i, cal_file in enumerate(cal_files):
    cal = h5py.File(data_dir + cal_file[:-1], 'r')

    K = np.array(cal['K'])
    R = np.array(cal['R'])
    T = np.array(cal['T'])
    imsize = np.array(cal['imsize'])

    K[0, 2] += imsize[0, 0] * 0.5
    K[1, 2] += imsize[0, 1] * 0.5

    cal_db[i] = (K, R, T)




with open("text_data/vis_" + opt.dataset + ".txt", "rb") as fp:   #Loading the pre-selected image-pairs to be tested 
    vis_pairs = pickle.load(fp)

result_log = open(out_dir+'/ours_result.txt', 'w', 1)


for i, vis_pair in enumerate(vis_pairs):

    img1_idx = vis_pair[0] #1st image of the image-pairs
    img2_idx = vis_pair[1] #2nd image of the image-pairs

    print("\nProcessing pair %d of %d. (%d, %d)" % (i, len(vis_pairs), img1_idx, img2_idx))

    img1 = cv.imread(data_dir + img_files[img1_idx][:-1])
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY).astype('float32') / 255.
    
    img2 = cv.imread(data_dir + img_files[img2_idx][:-1])
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY).astype('float32') / 255.
    
    
    heatmap1, coarse_desc1 = fe.run(img1)
    pts_1, desc_1 = fe.key_pt_sampling(img1, heatmap1, coarse_desc1, nfeatures) #Getting keypoints and descriptors for 1st image
    
    heatmap2, coarse_desc2 = fe.run(img2)
    pts_2, desc_2 = fe.key_pt_sampling(img2, heatmap2, coarse_desc2, nfeatures) #Getting keypoints and descriptors for 2nd image


    desc1 = desc_1.T
    desc2 = desc_2.T

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(desc1,desc2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    
    print("Found %d total matches." % len(matches))
    
    if (len(matches) > 5): #Checking whether minimum amount of matches are found to calculate the pose-matrix
    
        good_matches = []
        pts1 = []
        pts2 = []
        ratios = []

        for l in range(len(matches)):               
            pts1.append(pts_1[0:2,np.int(matches[l].queryIdx)])
            pts2.append(pts_2[0:2,np.int(matches[l].trainIdx)]) 

        pts1 = np.array([pts1])
        pts2 = np.array([pts2])
        

        K1 = cal_db[img1_idx][0]
        K2 = cal_db[img2_idx][0]
        pts1 = cv.undistortPoints(pts1, K1, None)
        pts2 = cv.undistortPoints(pts2, K2, None)
        K = np.eye(3, 3)


        E, mask = cv.findEssentialMat(pts1, pts2, K, method = cv.FM_RANSAC, threshold = inlierThreshold)
        inliers, R, t, mask = cv.recoverPose(E, pts1, pts2, K, mask = mask) #Getting the rotation matrix and translation vector from image pair

        print("Found %d good matches." % len(matches))


        GT_R1 = cal_db[img1_idx][1]
        GT_R2 = cal_db[img2_idx][1]
        gt_R = np.matmul(GT_R2, np.transpose(GT_R1)) #ground truth rotation matrix

        GT_t1 = cal_db[img1_idx][2]
        GT_t2 = cal_db[img2_idx][2]
        gt_T = GT_t2.T - np.matmul(gt_R, GT_t1.T) #ground truth translation vector

        
        dR = np.matmul(R, np.transpose(gt_R))
        dR = cv.Rodrigues(dR)[0]
        dR = np.linalg.norm(dR) * 180 / math.pi

        dT = float(np.dot(gt_T.T, t))
        dT /= float(np.linalg.norm(gt_T))
        if dT > 1 or dT < -1:
            print("Domain warning! dT:", dT)
            dT = max(-1, min(1, dT))
        dT = math.acos(dT) * 180 / math.pi
        
        print("Rotation Error: %.1f, Translation Error: %.1f" % (dR, dT))

        result_log.write('%f %f\n' % (dR, dT))

        counter -= 1
        if counter <= 0:
            break

result_log.close()




