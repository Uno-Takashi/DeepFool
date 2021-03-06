import os
import numpy as np
from pathlib import Path

from prepare_imagenet_data import *
import matplotlib.pyplot as plt


def visualization_pert(v):
    plt.imshow(v)
    plt.imshow()

def img2str(f,img):
    num_pert=np.argmax(f(img), axis=1).flatten()
    return cat2label_str(num_pert)

def cat2label_str(num_pert):
    labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n')
    return labels[np.int(num_pert)-1].split(',')[0]

def avg_add_clip_pert(avg_img,v):
    clipped_v = np.clip(undo_image_avg(avg_img[0,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(avg_img[0,:,:,:]), 0, 255)
    pert_img =  avg_img+ clipped_v[None, :, :, :]
    return pert_img


def fooling_rate_calc(v,dataset,f,batch_size=100):
    dataset_perturbed = dataset + v
    num_images =  np.shape(dataset)[0]
    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
        est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

    # Compute the fooling rate
    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    return fooling_rate


def fooling_rate_calc_datasets(dataset_original,dataset_perts,f,batch_size=100):
    dataset=dataset_original
    dataset_perturbed = dataset_perts
    num_images =  np.shape(dataset)[0]
    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
        est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

    # Compute the fooling rate
    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    print(est_labels_orig)
    print(est_labels_pert)
    return fooling_rate

def target_fooling_rate_calc(v,dataset,f,target,batch_size=100):
    dataset_perturbed = dataset + v
    num_images =  np.shape(dataset)[0]
    est_labels_pert = np.zeros((num_images))

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

    # Compute the fooling rate

    target_fooling_rate = float(np.sum(est_labels_pert == target) / float(num_images))
    return target_fooling_rate

def fooling_rate_calc_all(v,dataset,f,target,batch_size=100):
    dataset_perturbed = dataset + v
    num_images =  np.shape(dataset)[0]
    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
        est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

    # Compute the fooling rate
    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    target_fooling_rate = float(np.sum(est_labels_pert == target) / float(num_images))
    return fooling_rate,target_fooling_rate



def export_dataset(dataset,output_dir="output/"):
    output_dir = Path(output_dir)
    dataset=undo_image_list(dataset)

    data_num=int(np.shape(dataset)[0])

    for x in range(data_num):
        out_image=dataset[x]
        cv2.imwrite(str(output_dir.joinpath(str(x).zfill(5))) + ".png", out_image)