import numpy as np
import cv2
import math
import torch.nn.functional as F
from networks import SuperPointNet, CNNet
import torch
import torch.nn as nn

class SuperPointFrontend(object):
    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh, cuda):
        self.name = 'SuperPoint'
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.cuda = cuda

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if self.cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        self.net.train()

    def run(self, img):

        assert img.ndim == 3, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[1], img.shape[2]
        inp = img.copy()
        inp = (inp.reshape(2, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(2, 1, H, W)


        if self.cuda:
            inp = inp.cuda()
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc_org = outs[0], outs[1]
        coarse_desc = torch.tensor(coarse_desc_org.data, requires_grad=True)

        semi = torch.squeeze(semi)
        semi_dense = torch.ones(semi.shape[0], semi.shape[2], semi.shape[3]).cuda()
        semi_dense[0, :, :] = torch.log(torch.sum(torch.exp(semi[0, :, :, :]), 0) + .00001)
        semi_dense[1, :, :] = torch.log(torch.sum(torch.exp(semi[1, :, :, :]), 0) + .00001)
        dense = torch.ones(semi.shape[0], semi.shape[1], semi.shape[2], semi.shape[3]).cuda()
        dense[0, :, :, :] = semi[0, :, :, :] - semi_dense[0, :, :]
        dense[1, :, :, :] = semi[1, :, :, :] - semi_dense[1, :, :]
        nodust = dense[:, :-1, :, :]
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2)
        nodust = nodust.transpose(2, 3)
        no_dust = torch.reshape(nodust, [2, Hc, Wc, self.cell, self.cell])
        no_dust = no_dust.transpose(2, 3)
        no_dust = torch.reshape(no_dust, [2, Hc * self.cell, Wc * self.cell])
        heatmap = torch.exp(no_dust)
        #         print(no_dust.type())

        t1 = torch.sum(torch.sum(heatmap[0, :, :])) + .00001
        t2 = torch.sum(torch.sum(heatmap[1, :, :])) + .00001
        heat_map = torch.ones(2, heatmap.shape[1], heatmap.shape[2])
        heat_map[0, :, :] = heatmap[0, :, :] / t1
        heat_map[1, :, :] = heatmap[1, :, :] / t2
        prob_map = heat_map.data.cpu().numpy()

        xs1, ys1 = np.where(prob_map[0, :, :] >= self.conf_thresh)  # Confidence threshold.
        print(len(xs1))
        return heat_map, coarse_desc, no_dust, coarse_desc_org

    def key_pt_sampling(self, img, heat_map, sampl_pts, coarse_desc):

        H, W = img.shape[1], img.shape[2]
        pts_stack = []
        #         desc_stack = []
        inv_prob_stack = []
        prob_map = heat_map.data.cpu().numpy()
        sampled1 = np.amin([sampl_pts, 2000])
        pts1 = np.zeros((3, sampled1))  # Populate point data sized 3xN.
        prob_array1 = np.ravel(prob_map[0, :, :])
        #         print(prob_array1)
        #         if (np.sum(prob_array1) != 1):
        #             print("I'm here bitches")
        #             prob_array1[np.argmax(prob_array1)] += 1 - np.sum(prob_array1)
        img_array1 = np.arange(prob_map[0, :, :].shape[0] * prob_map[0, :, :].shape[1])
        desc_stack = torch.zeros(2, 256, sampled1)

        temp = np.random.choice(img_array1, sampled1, p=prob_array1, replace=True)

        x_ind1 = (np.divide(temp, prob_map[0, :, :].shape[1])).astype(int)
        y_ind1 = (np.mod(temp, prob_map[0, :, :].shape[1])).astype(int)
        pts1[0, :] = y_ind1
        pts1[1, :] = x_ind1
        pts1[2, :] = prob_map[0, x_ind1, y_ind1]

        inv_prob1 = np.zeros((prob_map[0, :, :].shape[0], prob_map[0, :, :].shape[1]))
        #         inv_prob1[x_ind1, y_ind1] = 1/pts1[2,:]
        inv_prob1[x_ind1, y_ind1] = 1
        inv_prob_stack.append(inv_prob1)

        # --- Process descriptor.
        D1 = coarse_desc.shape[1]
        if pts1.shape[1] == 0:
            desc1 = np.zeros((D1, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts1 = torch.from_numpy(pts1[:2, :].copy())
            samp_pts1[0, :] = (samp_pts1[0, :] / (float(W) / 2.)) - 1.
            samp_pts1[1, :] = (samp_pts1[1, :] / (float(H) / 2.)) - 1.
            samp_pts1 = samp_pts1.transpose(0, 1).contiguous()
            samp_pts1 = samp_pts1.view(1, 1, -1, 2)
            samp_pts1 = samp_pts1.float()
            samp_pts1 = samp_pts1.cuda()

            coarse1 = coarse_desc[0, :, :, :].unsqueeze(0)
            desc1 = nn.functional.grid_sample(coarse1, samp_pts1)
            desc1 = desc1.reshape(D1, -1)
            desc1a = desc1 / (torch.norm(desc1, p=2, dim=0))
        #             desc1b = desc1a.data.cpu().numpy()
        pts_stack.append(pts1)
        #         desc_stack.append(desc1a)
        desc_stack[0, :, :] = desc1a

        xs2, ys2 = np.where(prob_map[1, :, :] >= self.conf_thresh / 100.0)  # Confidence threshold.
        #         if len(xs2) == 0:
        #             return np.zeros((3, 0)), None, None

        sampled2 = np.amin([sampl_pts, 2000])
        pts2 = np.zeros((3, sampled2))  # Populate point data sized 3xN.
        prob_array2 = np.ravel(prob_map[1, :, :])
        #         if (np.sum(prob_array2) != 1):
        #             prob_array2[np.argmax(prob_array2)] += 1 - np.sum(prob_array2)
        img_array2 = np.arange(prob_map[1, :, :].shape[0] * prob_map[1, :, :].shape[1])

        temp = np.random.choice(img_array2, sampled2, p=prob_array2, replace=True)
        x_ind2 = (np.divide(temp, prob_map[1, :, :].shape[1])).astype(int)
        y_ind2 = (np.mod(temp, prob_map[1, :, :].shape[1])).astype(int)
        pts2[0, :] = y_ind2
        pts2[1, :] = x_ind2
        pts2[2, :] = prob_map[1, x_ind2, y_ind2]

        inv_prob2 = np.zeros((prob_map[1, :, :].shape[0], prob_map[1, :, :].shape[1]))
        #         inv_prob2[x_ind2, y_ind2] = 1/pts2[2,:]
        inv_prob2[x_ind2, y_ind2] = 1
        inv_prob_stack.append(inv_prob2)

        # --- Process descriptor.
        D2 = coarse_desc.shape[1]
        if pts2.shape[1] == 0:
            desc2 = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts2 = torch.from_numpy(pts2[:2, :].copy())
            samp_pts2[0, :] = (samp_pts2[0, :] / (float(W) / 2.)) - 1.
            samp_pts2[1, :] = (samp_pts2[1, :] / (float(H) / 2.)) - 1.
            samp_pts2 = samp_pts2.transpose(0, 1).contiguous()
            samp_pts2 = samp_pts2.view(1, 1, -1, 2)
            samp_pts2 = samp_pts2.float()
            samp_pts2 = samp_pts2.cuda()

            coarse2 = coarse_desc[1, :, :, :].unsqueeze(0)
            desc2 = nn.functional.grid_sample(coarse2, samp_pts2)
            desc2 = desc2.reshape(D2, -1)
            desc2a = desc2 / (torch.norm(desc2, p=2, dim=0))
        #             desc2b = desc2a.data.cpu().numpy()

        pts_stack.append(pts2)
        #         desc_stack.append(desc2a)
        desc_stack[1, :, :] = desc2a
        inv_prob_arr = np.asarray(inv_prob_stack)

        return pts_stack, desc_stack, inv_prob_arr

class CNFrontend(object):
    def __init__(self, weights_path, blocks, cuda=True):
        self.name = 'ngransac'
        self.net = CNNet(blocks)

        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        self.net.train()

    def run(self, inputs):
        batch_size = inputs.size(0)
        data_size = inputs.size(2)
        out = self.net.forward(inputs)
        log_probs = F.logsigmoid(out)

        # normalization in log space such that probabilities sum to 1
        log_probs = log_probs.view(batch_size, -1)
        normalizer = torch.logsumexp(log_probs, dim=1)
        normalizer = normalizer.unsqueeze(1).expand(-1, data_size)
        log_probs = log_probs - normalizer
        log_probs = log_probs.view(batch_size, 1, data_size, 1)

        return log_probs



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(-x) / np.sum(np.exp(-x), axis=0)


def softmax_t(x):
    """Compute softmax values for each sets of scores in x."""
    return torch.exp(-x) / torch.sum(torch.exp(-x), 0)


def desc_map(pts_stack, desc_stack, cr_check):
    desc1 = desc_stack[0, :, :].data.cpu().numpy().T
    desc2 = desc_stack[1, :, :].data.cpu().numpy().T
    #     desc1 = desc_stack[0,:,:].transpose(0,1)
    #     desc2 = desc_stack[1,:,:].transpose(0,1)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cr_check)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches, desc1.shape[0]


def desc_sampling(matches, pts_stack, desc_stack):
    desc1 = desc_stack[0, :, :].data.cpu().numpy().T
    desc2 = desc_stack[1, :, :].data.cpu().numpy().T
    p_stack = []
    d_stack = []
    match_dist = np.zeros((len(matches), 1))

    d1 = np.zeros((256, len(matches)))
    d2 = np.zeros((256, len(matches)))

    for i in range(len(matches)):
        match_dist[i, 0] = matches[i].distance
        d1[:, i] = desc1[np.int(matches[i].queryIdx), :]
        d2[:, i] = desc2[np.int(matches[i].trainIdx), :]

    match_prob = softmax(match_dist).reshape(len(matches, ))
    #     match_prob = scipy.special.softmax(-match_dist).reshape(len(matches,))
    match_array = np.arange(len(matches))
    match_sampling = np.int(0.5 * len(matches))
    temp = np.random.choice(match_array, match_sampling, p=match_prob, replace=True)
    #     temp_prob = match_prob[temp]

    p1 = np.zeros((3, match_sampling))  # Populate point data sized 3xN.
    p2 = np.zeros((3, match_sampling))
    d_1 = np.zeros((256, match_sampling))
    d_2 = np.zeros((256, match_sampling))
    prob_samp = np.zeros((len(matches)))

    for j in range(match_sampling):
        p1[:, j] = pts_stack[0][:, np.int(matches[temp[j]].queryIdx)]
        p2[:, j] = pts_stack[1][:, np.int(matches[temp[j]].trainIdx)]
        d_1[:, j] = desc1[np.int(matches[temp[j]].queryIdx), :]
        d_2[:, j] = desc2[np.int(matches[temp[j]].trainIdx), :]

    p_stack.append(p1)
    p_stack.append(p2)
    d_stack.append(d_1)
    d_stack.append(d_2)

    prob_samp[temp] = 1
    #     print(prob_samp)

    d_grad = []
    d1_tensor = torch.tensor(d1, requires_grad=True)
    d2_tensor = torch.tensor(d2, requires_grad=True)
    new_norm = torch.norm(d1_tensor - d2_tensor, p=2, dim=0)
    #     print(new_norm.shape)
    softy = nn.functional.log_softmax(-new_norm, dim=0)
    #     softy = softmax_t(new_norm)
    #     res = softya - softy

    softy.backward(torch.from_numpy(prob_samp))
    d1_grad = torch.zeros(256, desc1.shape[0])
    d2_grad = torch.zeros(256, desc2.shape[0])
    for k in range(len(matches)):
        d1_grad[:, np.int(matches[k].queryIdx)] = d1_tensor.grad[:, k]
        d2_grad[:, np.int(matches[k].trainIdx)] = d2_tensor.grad[:, k]
    d_grad.append(d1_grad.data.cpu().numpy())
    d_grad.append(d2_grad.data.cpu().numpy())

    #     print(d1_grad[:,0])

    return p_stack, d_stack, d_grad, match_sampling


def error_calculator(pts_stack, desc_stack, lamda, matches, cal_db, img1_idx, img2_idx, db_index, inlierThreshold):
    desc1 = desc_stack[0].T
    desc2 = desc_stack[1].T
    tot_loss = np.zeros((1, 1))
    pts1 = np.zeros((1, matches, 2))
    pts2 = np.zeros((1, matches, 2))
    tot_loss = np.zeros((1, 1))

    pts1[0, :, :] = pts_stack[0].T[:, 0:2]
    pts2[0, :, :] = pts_stack[1].T[:, 0:2]

    K1 = cal_db[db_index][img1_idx][0]
    K2 = cal_db[db_index][img2_idx][0]

    pts1 = cv2.undistortPoints(pts1, K1, None)
    pts2 = cv2.undistortPoints(pts2, K2, None)
    K = np.eye(3, 3)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.FM_RANSAC, threshold=inlierThreshold)
    inliers, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    #     print("Found %d good matches." % len(matches), "Final inlier count: ", inliers)

    # print("Estimate: ")
    # print(R)
    # print(t)

    GT_R1 = cal_db[db_index][img1_idx][1]
    GT_R2 = cal_db[db_index][img2_idx][1]
    GT_R_Rel = np.matmul(GT_R2, np.transpose(GT_R1))

    GT_t1 = cal_db[db_index][img1_idx][2]
    GT_t2 = cal_db[db_index][img2_idx][2]
    GT_t_Rel = GT_t2.T - np.matmul(GT_R_Rel, GT_t1.T)

    # print("Ground Truth:")
    # print(GT_R_Rel)
    # print(GT_t_Rel)

    dR = np.matmul(R, np.transpose(GT_R_Rel))
    dR = cv2.Rodrigues(dR)[0]
    dR = np.linalg.norm(dR) * 180 / math.pi

    dT = float(np.dot(GT_t_Rel.T, t))
    dT /= float(np.linalg.norm(GT_t_Rel))
    dT = math.acos(dT) * 180 / math.pi

    tot_loss[0, 0] = np.amax([dR, dT])
    tempu_loss = np.copy(tot_loss[0, 0])
    if (tot_loss[0, 0] > 25):
        tot_loss[0, 0] = np.sqrt(25 * tempu_loss)
    if (tot_loss[0, 0] > 75):
        tot_loss[0, 0] = 75
    return tot_loss