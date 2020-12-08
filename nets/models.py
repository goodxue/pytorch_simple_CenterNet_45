from .dla import DLASeg
import argparse

def create_model(opt=None):
    if opt is None:
        parser = argparse.ArgumentParser()
        opt = parser.parse_args()
    else:
        opt.dla_node = 'dcn'
        opt.head_kernel = 3

    model_class = DLASeg

    num_layers = 34
    heads = {'hm': opt.num_classes, 'reg': 2, 'wh': 2, \
      'tracking': 2,'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2}
    head_conv = 256
    num_head_conv = 1
    head_conv = {head: [head_conv \
      for i in range(num_head_conv if head != 'reg' else 1)] for head in heads}

    model = model_class(num_layers, heads=heads, head_convs=head_conv, opt=opt)
    return model