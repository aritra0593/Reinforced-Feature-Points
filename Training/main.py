import argparse
import random
import torch.optim as optim
from utils import SuperPointFrontend
from dataset import build_dataset
from engine import train_one_epoch

def get_args_parser():
    # Parse command line arguments.
    parser = argparse.ArgumentParser('Set network parameters', add_help=False)

    parser.add_argument('--dataset1', default='st_peters_square', type=str,
      help='Image directory of outdoor images for training the network')
    parser.add_argument('--dataset2', default='brown_bm_3---brown_bm_3-maxpairs-10000-random---skip-10-dilate-25', type=str,
      help='Image directory of indoor images for training the network')
    parser.add_argument('--weights1', default='weights/superpoint_v1.pth', type=str,
      help='Path to pretrained weights file for backend')
    parser.add_argument('--nms_dist', default=4, type=int, help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', default=0.00015, type=float, help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', default=0.7, type=float, help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--threshold', default=0.001, type=float, help='inlier threshold')
    parser.add_argument('--ratio', default=1.0, type=float, help='lowes ratio test')
    parser.add_argument('--vthreshold1', default=100.0, type=float, help='visibility threshold_outdoors')
    parser.add_argument('--vthreshold2', default=0.5, type=float, help='visibility threshold_indoors')
    parser.add_argument('--lr_bbone', default=0.0000001, type=float, help='learning rate for backbone')
    parser.add_argument('--samp_pts', default=600, type=int, help='number of keypoints sampled')
    parser.add_argument('--cr_check', action='store_false', help='cross check option for BFmatcher (default: true)')
    parser.add_argument('--cuda', action='store_false', help='Use cuda GPU to speed up network processing speed (default: true)')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=50, type=int, help = 'number of epochs')
    parser.add_argument('--output_dir', default='output', type=str, help='turn on training for blackbox')
    return parser

def main(args):
    model_bbone = SuperPointFrontend(weights_path=args.weights1, nms_dist=args.nms_dist, conf_thresh=args.conf_thresh,
                            nn_thresh=args.nn_thresh, cuda=args.cuda)

    optimizer = optim.Adam(model_bbone.net.parameters(), lr=args.lr_bbone)
    data_dir, img_files, vis_pairs, cal_db = build_dataset(args.dataset1, args.dataset2, args.vthreshold1, args.vthreshold2)
    for epoch in range(50):
        random.shuffle(vis_pairs)
        train_one_epoch(model_bbone, optimizer, args.cr_check, data_dir, img_files, cal_db, vis_pairs, args.samp_pts, args.threshold, epoch, args.output_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReInf training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)