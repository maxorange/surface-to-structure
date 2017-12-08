import argparse
import datasets
import trainers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1.0.0')
    parser.add_argument('--check', type=bool, default=False)
    parser.add_argument('--n_gpus', type=int, default=2)
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--adv_weights', type=float, default=[0.5, 0.5], nargs='+')
    parser.add_argument('--nsf_disc', type=int, default=[8, 64], nargs='+', help='patch sizes of discriminators')
    parser.add_argument('--nsf_gen', type=int, default=4, help='encoded pixel size of generator')
    parser.add_argument('--npx', type=int, default=512, help='output pixel size')
    parser.add_argument('--dataset_path', type=str, default='dataset/vhd/train/input/*.png')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    trainer = trainers.SurfaceToStructure(args)
    if not args.check:
        dataset = datasets.VHD(args)
        trainer.run(args, dataset)
    trainer.close()
