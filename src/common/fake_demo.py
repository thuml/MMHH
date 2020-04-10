from common.mmhh_config import data_config


def get_fake_train_list(s_dataset, t_dataset):
    return get_fake_list(s_dataset), get_fake_list(t_dataset)


def get_fake_test_list(config, t_dataset):
    config["data"]["database"]["list_path"] = get_fake_list(t_dataset)
    config["data"]["test"]["list_path"] = get_fake_list(t_dataset)
    config["R"] = 10


def get_fake_list(dataset):
    if dataset in ['ElectricDevices', 'Crop', 'InsectWingbeat']:
        return data_config["ElectricDevices"]["train"]
    fake_dir = '../data/fake'
    # fake_dir = 'data/fake'
    if dataset in ['shapenet_9', 'shapenet_13', 'shape_pro_13', 'shape_pro_9', 'modelnet_10', 'modelnet_40',
                   'modelnet_sm_11']:
        return fake_dir + '/voxel_list.txt'
    else:
        return fake_dir + '/image_list.txt'
