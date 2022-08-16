import argparse
import json
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

from time import time

import albumentations as alb
import faiss
import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from scipy.ndimage import zoom
from torch.multiprocessing import Pool, Process, set_start_method

try:
     set_start_method('spawn')
except RuntimeError:
    pass

from tqdm import tqdm

import data.transforms as T
from model.loader import load_pretrained_model
from utils.FeatureExtractor import FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--num_proc", default=48, type=int,
                    help="Multiprocessing to generate masks")
parser.add_argument('--mask_dir', default='', type=str, metavar='PATH',
                    help='save directory for masks')
parser.add_argument("--layer_name", default='', type=str,
                    help="Layer to index FeatureExtractor on")
parser.add_argument('--imgs_per_batch', type=int, default=1,
                    help="Images to run KMeans together on")
parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                    help='path to bootstrap checkpoint')
parser.add_argument("--data_dir", metavar="DIR",
                    help="path to serialized LMDB file")
parser.add_argument("--min_clusters", type=int, default='',
                    help="Minimum number of classes in segmentation")
parser.add_argument("--max_clusters", type=int, default='',
                    help="Maximum number of classes in segmentation")
args = parser.parse_args()


transform = alb.Compose([
                alb.Normalize(mean=T.IMAGENET_COLOR_MEAN, std=T.IMAGENET_COLOR_STD, p=1.0),
                ToTensorV2(),
            ])


def run_faiss_cpu(X, K):
    # This code is based on https://github.com/facebookresearch/faiss/blob/master/benchs/kmeans_mnist.py

    X = X.astype(np.float32)
    X = np.ascontiguousarray(X)
    n_data, d = X.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, K)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    
    index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(X, index)

    _, I = index.search(X, 1)

    return I


###################################

def process_one_class(process_id, mask_save_dir, layer_name, checkpoint_path,
                                    all_imgs, num_imgs, imgs_per_process, imgs_per_batch,
                                    min_clusters, max_clusters,
                                    ):
    print(f"Process id: {process_id} started")
    gpu = process_id % 8
    
    # Load model
    model = load_pretrained_model(checkpoint_path)
    model.cuda(gpu)

    for i in range(process_id * imgs_per_process,
                    process_id * imgs_per_process + imgs_per_process,
                    imgs_per_batch):
        if i >= num_imgs:
            break
        
        img_end = min(i + imgs_per_batch, process_id * imgs_per_process + imgs_per_process)

        img_locs = all_imgs[i:img_end]

        fmaps = []
        fmap_dims = []
        # min_fmap_dim = num_classes
        min_fmap_dim = np.random.randint(min_clusters, max_clusters)
        fmap_idx_start = 0
        fmap_idx_end = None
        fmap_indices = []
        # Non-square images of different dimensions
        for j, img_loc in enumerate(img_locs):
            file_name = os.path.basename(img_loc)
            
            image = np.array(Image.open(img_loc).convert('RGB'))
            img_height, img_width, _ = image.shape
            img_tensor = transform(image=image)["image"]
            img_tensor = img_tensor.unsqueeze(0).cuda(gpu)

            # Forward through model w/ hook
            moco_features = FeatureExtractor(model, layers=[layer_name])
            with torch.no_grad():
                layer_out = moco_features(img_tensor)
            
            # Cluster on features
            fmap = layer_out[layer_name][0]
            fmap = fmap.cpu().numpy()
            num_ft, ft_height, ft_width = fmap.shape

            fmaps.append(np.reshape(fmap, (num_ft, -1)).T)
            fmap_dims.append((ft_height, ft_width))
            min_fmap_dim = min(min_fmap_dim, ft_height * ft_width)
            fmap_idx_end = fmap_idx_start + ft_height * ft_width
            fmap_indices.append((fmap_idx_start, fmap_idx_end))
            fmap_idx_start = fmap_idx_end
        
        fmaps = np.vstack(fmaps)

        # L2 normalize features across batch
        norm = np.linalg.norm(fmaps, axis=1, keepdims=True)
        norm[norm == 0] = 1e-6
        cluster_features = fmaps / norm

        kmeans_labels = run_faiss_cpu(cluster_features, min_fmap_dim)

        # Save the segmentation labels
        for j, img_loc in enumerate(img_locs):
            file_name = os.path.basename(img_loc)
            
            image = np.array(Image.open(img_loc).convert('RGB'))
            img_height, img_width, _ = image.shape

            # Reshape to image dims
            # Retrieve fmap dimensions
            ft_height, ft_width = fmap_dims[j]

            # Index into returned labels and reshape
            idx_start, idx_end = fmap_indices[j]
            curr_labels = kmeans_labels[idx_start:idx_end].reshape((ft_height, ft_width))
            repeated_labels = zoom(curr_labels,
                                    (img_height * 1. / ft_height, img_width * 1. / ft_width),
                                    order=0, mode='nearest')
            save_labels = repeated_labels[:img_height, :img_width]

            save_path = os.path.join(mask_save_dir, file_name.replace('.jpg','.npy'))
            np.save(save_path, save_labels)
    
    return


if __name__ == '__main__':
    num_proc = args.num_proc
    mask_save_dir = args.mask_dir
    layer_name = args.layer_name
    imgs_per_batch = args.imgs_per_batch
    checkpoint_path = args.ckpt
    data_dir = args.data_dir

    min_clusters = args.min_clusters
    max_clusters = args.max_clusters + 1

    all_imgs = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn]

    processes_num = num_proc
    num_imgs = len(all_imgs)
    imgs_per_process = num_imgs // processes_num + 1

    processes = [Process(target=process_one_class,
                            args=(process_id, mask_save_dir, layer_name, checkpoint_path,
                                    all_imgs, num_imgs, imgs_per_process, imgs_per_batch,
                                    min_clusters, max_clusters))
                            for process_id in range(processes_num)]


    start = time()

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    end = time()
    print("Time elapsed: %.4f" % (end - start))
