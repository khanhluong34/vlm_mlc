import argparse

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('-c',
                    '--config-file',
                    help='config file',
                    default='voc',
                    choices=['voc', 'coco', 'cub', 'nuswide'])
parser.add_argument('-t',
                    '--test',
                    help='run test',
                    default=False,
                    action="store_true")
parser.add_argument('-r', '--round', help='round', default=1, type=int)
parser.add_argument('--resume', default=False, action='store_true')

parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--backbone', default='resnet50', choices=['resnet50', 'vitb32'])

args = parser.parse_args()
