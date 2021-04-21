import numpy as np
import h5py
import random


def create_batch(dataset, vis_thresh, id):
    cal_db_list = {}
    vis_pairs = []

    data_dir = 'datasets/' + dataset + '/train/'

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

    for i, cal_file in enumerate(cal_files):
        cal = h5py.File(data_dir + cal_file[:-1], 'r')

        K = np.array(cal['K'])
        R = np.array(cal['R'])
        T = np.array(cal['T'])
        imsize = np.array(cal['imsize'])
        #     print(imsize[0,0], imsize[0,1])

        #     K[0, 2] += imsize[0, 0] * 0.5
        #     K[1, 2] += imsize[0, 1] * 0.5
        K[0, 2] += 1024 * 0.5
        K[1, 2] += 1024 * 0.5

        cal_db_list[i] = (K, R, T)

    for i, vis_file in enumerate(vis_files):

        vis_file = open(data_dir + vis_file[:-1])
        vis_infos = vis_file.readlines()

        for j, vis_info in enumerate(vis_infos):
            vis_count = float(vis_info)
            if vis_count > vis_thresh:
                vis_pairs.append((i, j, 0))

        vis_file.close()

    random.shuffle(vis_pairs)
    if id == 1:
        vis_mod = vis_pairs[0:10000]
    else:
        vis_mod = vis_pairs.copy()

    return data_dir, img_files, cal_db_list, vis_mod

def build_dataset(dataset1, dataset2, vthresh1, vthresh2):
    data_dir_arr = []
    img_files_arr = []
    vis_pairs_arr = []
    cal_db_arr = []

    data_dir1, img_files1, cal_db1, vis_pair1 = create_batch(dataset1, vthresh1, 1)
    data_dir_arr.append(data_dir1)
    img_files_arr.append(img_files1)
    cal_db_arr.append(cal_db1)
    vis_pairs_arr = vis_pair1.copy()
    data_dir2, img_files2, cal_db2, vis_pair2 = create_batch(dataset2, vthresh2, 2)
    data_dir_arr.append(data_dir2)
    img_files_arr.append(img_files2)
    cal_db_arr.append(cal_db2)
    vis_pairs_arr += vis_pair2.copy()
    random.shuffle(vis_pairs_arr)

    return data_dir_arr, img_files_arr, vis_pairs_arr, cal_db_arr
