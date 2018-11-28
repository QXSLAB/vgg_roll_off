import argparse

parser = argparse.ArgumentParser(description='ResNet Roll Off Classification')
parser.add_argument('--arch', default='resnet18', metavar='ARCH',
                    help='model architecture: resnet18, resnet34, resnet50, resnet101')

parser.add_argument('--cuda', default='0', metavar='N', type=str,
                    help='cuda id')
parser.add_argument('--root', default='', metavar='N', type=str,
                    help='data file')

args = parser.parse_args()

print(args)