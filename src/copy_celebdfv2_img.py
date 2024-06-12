
import torch
from PIL import Image 
import os, json, glob
import cv2
import pandas as pd
import shutil
from utils import log_print

def main(root = 'data/Celeb-DF-v2', train_type = "train", dst_path = 'data/Celeb-DF-v2-imgs'):
    cwd = os.getcwd()
    test_videos = []
    csv_path = os.path.join(root, "List_of_testing_videos.txt")
    test_video_path = [os.path.join(cwd, root + str("-mtcnn"), t.replace(".mp4","/") + str("*.png")) for t in pd.read_csv(csv_path, sep=" ", header=None).values[:,1].tolist()]
    for p in test_video_path:
        for sp in glob.glob(p):
            test_videos.append(sp)
    # print('test videos: ', test_videos, len(test_videos))

    video_path = root + str('-mtcnn/*/*/*.png')
    all_videos = glob.glob(os.path.join(cwd, video_path))

    if train_type == "train":
        train_videos = list(set(all_videos)-set(test_videos))

        real_imgs = []
        fake_imgs = []
        for video in train_videos:
            if "real" in video and os.path.isfile(video):
                real_imgs.append(video)
                # print(glob.glob(os.path.join(root, video)))
            elif os.path.isfile(video):
                fake_imgs.append(video)
        
        log_print("[{}]\t fake imgs count :{}, real imgs count :{}".format(train_type, len(fake_imgs),len(real_imgs)))

        dst_real_path = os.path.join(cwd, dst_path) + '/' + str(train_type) + '/'+'real'
        dst_fake_path = os.path.join(cwd, dst_path) + '/' + str(train_type) + '/'+'fake'
        log_print("dst real path :{}, dst fake path :{}".format(dst_real_path, dst_fake_path))
        if not os.path.isdir(dst_real_path):
            os.makedirs(dst_real_path)
        if not os.path.isdir(dst_fake_path):
            os.makedirs(dst_fake_path)
        for p in real_imgs:
            file_name = p.split('/')[-1]
            dst_real_path_f = os.path.join(dst_real_path, file_name)
            shutil.copyfile(p, dst_real_path_f)
        for p in fake_imgs:
            file_name = p.split('/')[-1]
            dst_fake_path_f = os.path.join(dst_fake_path, file_name)
            shutil.copyfile(p, dst_fake_path_f)
    elif train_type == "test":
        test_real_imgs = []
        test_fake_imgs = []
        for video in test_videos:
            if "real" in video and os.path.isfile(video):
                test_real_imgs.append(video)
                # print(glob.glob(os.path.join(root, video)))
            elif os.path.isfile(video):
                test_fake_imgs.append(video)
        
        log_print("[{}]\t fake imgs count :{}, real imgs count :{}".format(train_type, len(test_fake_imgs),len(test_real_imgs)))

        dst_real_path = os.path.join(cwd, dst_path) + '/' + str(train_type) + '/'+'real'
        dst_fake_path = os.path.join(cwd, dst_path) + '/' + str(train_type) + '/'+'fake'
        log_print("dst real path :{}, dst fake path :{}".format(dst_real_path, dst_fake_path))
        if not os.path.isdir(dst_real_path):
            os.makedirs(dst_real_path)
        if not os.path.isdir(dst_fake_path):
            os.makedirs(dst_fake_path)
        for p in test_real_imgs:
            file_name = p.split('/')[-1]
            dst_real_path_f = os.path.join(dst_real_path, file_name)
            shutil.copyfile(p, dst_real_path_f)
        for p in test_fake_imgs:
            file_name = p.split('/')[-1]
            dst_fake_path_f = os.path.join(dst_fake_path, file_name)
            shutil.copyfile(p, dst_fake_path_f)


if __name__ == '__main__':
    main(root = 'data/Celeb-DF-v2', train_type="test", dst_path = 'data/Celeb-DF-v2-imgs')