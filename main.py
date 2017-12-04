import argparse

import tensorflow as tf
from CASED import CASED
from utils import check_folder
from utils import show_all_variables

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of CASED"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, help='The size of test batch')
    parser.add_argument('--num_gpu', type=int, default=8, help='# of gpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
        assert args.test_batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = CASED(sess, epoch=args.epoch, batch_size=args.batch_size, test_batch_size=args.test_batch_size, num_gpu=args.num_gpu,
                      checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)


        # build graph
        model.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        model.train()
        print(" [*] Training finished!")



if __name__ == '__main__':
    main()
