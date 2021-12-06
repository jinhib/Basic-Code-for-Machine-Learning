import shutil
import os

label = 'normal'

base_dir = 'C:/Users/Gachon/Desktop/exp/'
ori_dir = base_dir + 'Experiment_original_mel_60/' + label
# copy
for file_num in range(0, 7):

    copy_train_dir = base_dir + 'Experiment_original_mel_60/train/' + label
    copy_test_dir = base_dir + 'Experiment_original_mel_60/test/' + label
    file_list = os.listdir(ori_dir)
    idx = int(len(file_list) * 0.8)
    train_list = file_list[:idx]
    test_list = file_list[idx:]

    for i in train_list:
        shutil.copy(ori_dir+'/' + i, copy_train_dir)

    for i in test_list:
        shutil.copy(ori_dir + '/' + i, copy_test_dir)