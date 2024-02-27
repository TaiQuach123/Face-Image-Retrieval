from lib import *

def get_img_names(dataset_dir):
    '''Get image file name from image folder'''
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(dataset_dir):
        img_files.extend(filenames)
        break
    
    return img_files