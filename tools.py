import os
import cv2
import numpy as np
import pandas as pd
from scipy.misc import imresize
import scipy.io



current_path = os.getcwd()
img_file_path = current_path + '/data/img_align_celeba'
property_file_path = current_path + '/data/list_attr_celeba.txt'

# img_file_path = '/floyd/input/data/img_align_celeba/'
# property_file_path = '/floyd/input/data/list_attr_celeba.txt'

def getformattedProperties(property_file, image_dir_path):
    rows = []
    with open(property_file) as file:
        for line in file:
            rows.append(line.strip().split())

    cols = ['image_path'] + rows[1]
    rows = rows[2:]
    df = pd.DataFrame(rows, columns=cols)
    df['image_path'] = df['image_path'].map( lambda x: os.path.join(image_dir_path, x ))
    return df


def getCeleb(constraint, constraint_type, domainA=None, domainB=None, test=False, n_test=900):
    global property_file_path, img_file_path
    property_file = property_file_path
    img_file = img_file_path
    img_data = getformattedProperties(property_file_path, img_file_path)

    if constraint is not None:
        img_data = img_data[ img_data[constraint] == str(constraint_type)]

    domainA_data = img_data[img_data[domainA] == '1'].image_path.values

    if domainB is not None:
        domainB_data = img_data[img_data[domainB] == '1'].image_path.values

    if test == True:
        return domainA_data[-n_test:], domainB_data[-n_test:]

    return domainA_data[:-n_test], domainB_data[:-n_test]

def read_images( filenames, domain=None, image_size=64):
    images = []
    for fn in filenames:
        image = cv2.imread(fn)
        if image is None:
            continue

        image = cv2.resize(image, (image_size,image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2,0,1)
        images.append( image )

    images = np.stack( images )
    return images
