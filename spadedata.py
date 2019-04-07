from fastai.vision import conv_layer
from fastai.vision.data import SegmentationLabelList, SegmentationProcessor, SegmentationItemList
from fastai.vision.transform import get_transforms
from fastai.vision.image import open_image, open_mask, Image
from pathlib import Path
import matplotlib.pyplot as plt
from torch import nn

class SpadeLabelList(SegmentationItemList):
    "`ItemList` for segmentation masks."
#     _processor = SpadeProcessor
    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)
             #  loss function to be added

    def open(self, fn, **kwargs):
        return open_image(fn, **kwargs)
    
    def analyze_pred(self, pred, thresh:float=0.5):
        return pred
    
    def reconstruct(self, t):
        return Image(t)

class SpadeItemList(SegmentationLabelList):
    "`ItemList` suitable for segmentation tasks."
    _label_cls,_square_show_res = SpadeLabelList, False
    
    def show_xys(self, xs, ys, **kwargs):
        rows = len(xs)
        fig, ax = plt.subplots(rows, 2, figsize=(30,30))
        for i in range(rows):
            xs[i].show(ax=ax[i,0], figsize=(7,7))
            ys[i].show(ax=ax[i,1], figsize=(7,7))