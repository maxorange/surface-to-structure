import argparse
import cv2
import glob
import numpy as np
import os
import testers
import utils

def generate_data(args, tester):
    for filename in glob.glob(args.input_data):
        head, tail = os.path.split(filename)
        y = cv2.imread(filename, 0).astype(np.float32) / 255
        y = y.reshape((args.batch_size, args.npx, args.npx, 1))
        x = tester.generate(y)
        cv2.imwrite(os.path.join('out', tail), utils.tanh2uint16(x))
        print filename

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nsf', type=int, default=4, help='encoded pixel size of generator')
    parser.add_argument('--npx', type=int, default=512, help='output pixel size')
    parser.add_argument('--params_path', type=str, default='params/epoch-50.ckpt')
    parser.add_argument('--input_data', type=str, default='data/*.png')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    tester = testers.SurfaceToStructure(args)
    tester.restore(args.params_path)
    generate_data(args, tester)
    tester.close()
