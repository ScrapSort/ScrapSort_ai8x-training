import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from PIL import Image
import torch
import pandas as pd
import os
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datasets.sorting import contrastive_sorting_get_datasets,sorting_get_datasets
from models.sortingnet import SimpleSortingClassifierBN128
import ai8x

#train, test = contrastive_sorting_get_datasets(None)
model = SimpleSortingClassifierBN128()
##train, test = sorting_get_datasets(None)
#train.visualize_batch(model)
# print(img.size())
# np.save("rand.npy",img)