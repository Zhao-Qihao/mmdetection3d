import os
from os import path as osp
import mmengine
import numpy as np
class_names = ['car',
        'truck',
        'bus',
        'bicycle',
        'pedestrian',
        'traffic_cone',
        'barrier',] 

class_order = [0, 1, 2, 3, 4, 5, 6]
categories = dict(zip(class_names, class_order))

def create_custom_dataset_infos(root_path, info_prefix):
    
    train_infos, val_infos = _fill_trainval_infos(root_path)
    metainfo = dict(categories=categories,
                    dataset = "custom",
                    version = "v1.0")
    
    if train_infos is not None:
        data = dict(data_list=train_infos, metainfo=metainfo)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        print(info_path)
        mmengine.dump(data, info_path)

    if val_infos is not None:
        data['data_list'] = val_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)


def _fill_trainval_infos(root_path):

    train_infos = []
    val_infos = []

    trainSet = root_path + '/ImageSets/train.txt'
    valSet = root_path + '/ImageSets/val.txt'
    train_dict  , val_dict = set(), set()
    with open(trainSet, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            train_dict.add(ann)
    with open(valSet, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            val_dict.add(ann)

    totalPoints = os.listdir(root_path + '/points')
    for i in range(len(totalPoints)):
        
        file_name = totalPoints[i][:-4]
        print(file_name)
        lidar_path = root_path + '/points/' + file_name + '.bin'
        # img_path = root_path + '/images/' + file_name + '.jpg'
        label_path = root_path + '/labels/' + file_name + '.txt'
        
        mmengine.check_file_exist(lidar_path)
        # mmengine.check_file_exist(img_path)
        mmengine.check_file_exist(label_path)
        
        # time_stamp_list = file_name.split('_')
        # time_stamp = int(time_stamp_list[0][-4:]) + int(time_stamp_list[1]) / (10 * len(time_stamp_list[1]))
        # print(time_stamp)
        info = {
            'sample_idx': i % len(train_dict),
            # 'timestamp': timestamp,
            'token': file_name, 
            'lidar_points': dict(),
            'images': dict(),
            'instances': [],
            'cam_instances': dict(),
        }
        
        # lidar_points 相关参数
        info['lidar_points']['lidar_path'] = file_name + '.bin'
        info['lidar_points']['num_pts_feats'] = 4
        info['lidar_points']['lidar2ego'] = np.array([
                                                    [1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]  # 最后一行补充为 [0, 0, 0, 1], 4*4矩阵
                                                    ])
        info['lidar_points']['ego2global'] = None

        cameras = [
            '0',
            '1',
            '2',
            '3',
            '4',
        ]


        # image 相关参数
        for cam_name in cameras:
            cam_path = file_name + '.jpg'
            # 初始化子字典
            info['images']["CAM_"+cam_name] = dict()

            # info['images']["CAM_"+cam_name]['img_path'] = root_path + '/image_' + cam_name + '/' + cam_path
            info['images']["CAM_"+cam_name]['img_path'] = cam_path
            info['images']["CAM_"+cam_name]['height'] = 1080
            info['images']["CAM_"+cam_name]['width'] = 1920
            # 加载标定文件并解析 P0 参数
            calib_file_path = osp.join(root_path, 'calibs', file_name + '.txt')
            with open(calib_file_path, 'r') as f:
                calib_data = f.readlines()
            # 提取 P 参数
            P_line = [line for line in calib_data if line.startswith('P'+cam_name+':')][0]
            P_values = list(map(float, P_line.split()[1:]))
            info['images']["CAM_"+cam_name]['cam2img'] = np.array([
                                                            [P_values[0], P_values[1], P_values[2]],
                                                            [P_values[4], P_values[5], P_values[6]],
                                                            [P_values[8], P_values[9], P_values[10]],  # 这里后续由不同的数据集修改
                                                        ])
            # 提取外参
            RT_line = [line for line in calib_data if line.startswith('lidar2cam'+cam_name+':')][0]
            RT_values = list(map(float, RT_line.split()[1:]))
            info['images']["CAM_"+cam_name]['lidar2cam'] = np.array([
                                                            [RT_values[0], RT_values[1], RT_values[2], RT_values[3]],
                                                            [RT_values[4], RT_values[5], RT_values[6], RT_values[7]],
                                                            [RT_values[8], RT_values[9], RT_values[10], RT_values[11]],
                                                            [0, 0, 0, 1]  # 最后一行补充为 [0, 0, 0, 1], 4*4矩阵
                                                        ])
            
            # print(info['images']["CAM_"+cam_name])



        with open(label_path, 'r', encoding='utf-8') as f:
            # i = 0
            for ann in f.readlines():
                ann = ann.strip('\n')
                ann = ann.split()
                if len(ann):
                    # instances
                    info['instances'].append(dict())
                    info['instances'][-1]['bbox_3d'] = [float(ann[0]), float(ann[1]), float(ann[2]), float(ann[3]), float(ann[4]), float(ann[5]), float(ann[6])]
                    info['instances'][-1]['bbox_label_3d'] = categories[ann[7]]
                    info['instances'][-1]['bbox_3d_isvalid'] = None # 如果没有需要使用的地方，可以用None代替
                    info['instances'][-1]['num_lidar_pts'] = None

        
        if file_name in train_dict:
            train_infos.append(info)
        else:
            val_infos.append(info)
        # print(train_infos)



    return train_infos, val_infos

if __name__ == '__main__':
    train_infos, val_infos = _fill_trainval_infos('/data/datasets/custom_dataset')
    print(len(train_infos))
    print(len(val_infos))