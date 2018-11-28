import os

os.system('python resnet_h5py.py --cuda 3 --arch resnet18 '
          '--root D:/qpsk_awgn_sym256_sps8_esno10.dat >> esno10_sym256_sps8_resnet18.txt')

os.system('python resnet_h5py.py --cuda 3 --arch resnet18 '
          '--root D:/qpsk_awgn_sym256_sps8_esno12.dat >> esno12_sym256_sps8_resnet18.txt')

os.system('python resnet_h5py.py --cuda 3 --arch resnet18 '
          '--root D:/qpsk_awgn_sym256_sps8_esno14.dat >> esno14_sym256_sps8_resnet18.txt')

os.system('python resnet_h5py.py --cuda 3 --arch resnet18 '
          '--root D:/qpsk_awgn_sym256_sps8_esno16.dat >> esno16_sym256_sps8_resnet18.txt')

os.system('python resnet_h5py.py --cuda 3 --arch resnet18 '
          '--root D:/qpsk_awgn_sym256_sps8_esno18.dat >> esno18_sym256_sps8_resnet18.txt')

os.system('python resnet_h5py.py --cuda 3 --arch resnet18 '
          '--root D:/qpsk_awgn_sym256_sps8_esno20.dat >> esno20_sym256_sps8_resnet18.txt')