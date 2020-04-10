# coding=utf-8
import argparse
import os
import os.path as osp
import time
from pprint import pprint

import torch.multiprocessing
import torch.optim as optim

import mmhh_network
from mmhh import MMHH
from semi_batch import SemiBatch
from common.fake_demo import get_fake_train_list
from common.lr_scheduler import INVScheduler, StepScheduler
from tensorboardX import SummaryWriter
from common.logger import get_log
from common.mmhh_config import data_config, ImageLossType, DistanceType, Mission, TestType, BatchType, SemiInitType, \
    MarginParams, LrScheduleType
from test_mmhh import save_and_test

torch.multiprocessing.set_sharing_strategy('file_system')


def get_optimizer(m_logger, parameter_list, lr_schedule_type, init_lr=1.0, decay_step=10000, weight_decay=0.1):
    # type = "DANN_INV"
    # type = "HashNet_Step"
    if lr_schedule_type == LrScheduleType.HashNet_Step:
        m_logger.info("HashNet_Step")
        optimizer = optim.SGD(parameter_list, lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
        lr_scheduler = StepScheduler(gamma=0.5, step=2000, init_lr=0.0003)
    elif lr_schedule_type == LrScheduleType.Stair_Step:
        m_logger.info("Stair_Step")
        optimizer = optim.SGD(parameter_list, lr=0.5, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        lr_scheduler = StepScheduler(gamma=0.5, step=decay_step, init_lr=init_lr)
    elif lr_schedule_type == LrScheduleType.DANN_INV:
        m_logger.info("DANN_INV")
        optimizer = optim.SGD(parameter_list, lr=0.3, momentum=0.9, weight_decay=0.0005, nesterov=True)
        lr_scheduler = INVScheduler(gamma=0.0003, decay_rate=0.75, init_lr=0.0003)
    else:
        raise NotImplementedError
    group_ratios = [param_group["lr"] for param_group in optimizer.param_groups]
    return lr_scheduler, optimizer, group_ratios


def get_feature_net(m_logger, dataset, hash_bit, image_loss_type=ImageLossType.HashNet,
                    network_name='ResNetFc'):
    if dataset in ["shapenet_13", "shapenet_9", 'modelnet_10', 'modelnet_40', 'modelnet_sm_11']:
        raise NotImplementedError
    elif dataset in ["ElectricDevices", "Crop", 'InsectWingbeat']:
        raise NotImplementedError
    else:
        if network_name == 'ResNetFc':
            m_logger.info("feature net: ResNetFc")
            _feature_net = mmhh_network.ResNetFc(m_logger, 'ResNet50', hash_bit,
                                                 increase_scale=(image_loss_type == ImageLossType.HashNet))
        elif network_name == 'AlexNetFc':
            m_logger.info("feature net: AlexNetFc")
            _feature_net = mmhh_network.AlexNetFc(m_logger, hash_bit,
                                                  increase_scale=(image_loss_type == ImageLossType.HashNet))
        else:
            raise NotImplementedError
    return _feature_net


def get_next_iter_with_index(m_logger, data_loader, batch_size, data_loader_iter=None, balance_sampling=False):
    if data_loader_iter is None:
        data_loader_iter = iter(data_loader)
    try:
        inputs, labels, indices = data_loader_iter.next()
    except StopIteration:
        m_logger.info('stop iter, re-init')
        data_loader_iter, inputs, labels, indices = get_next_iter_with_index(m_logger, data_loader, batch_size,
                                                                             balance_sampling=balance_sampling)
    return data_loader_iter, inputs, labels, indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMHH')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--s_dataset', type=str, default='imagenet_13', help="dataloader dataset name")
    parser.add_argument('--t_dataset', type=str, default='', help="target dataset name")
    parser.add_argument('--batch_size', type=int, default=48, help="batch size")
    parser.add_argument('--snap_path', type=str, default='../snapshot/hash/', help="save path prefix")
    parser.add_argument('--hash_bit', type=int, default=48, help="output hash bit")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--annotation', type=str, default='empty', help="annotation for distinguishing")
    parser.add_argument('--loss_lambda', type=float, default='0.1', help="loss_lambda")
    parser.add_argument('--num_iters', type=int, default=1000, help="number of iterations")
    parser.add_argument('--pre_model_path', type=str, default='', help="continue training based on previous model")
    # network
    parser.add_argument('--image_network', type=str, default='AlexNetFc', help="ResNetFc or AlexNetFc")
    # different schema
    parser.add_argument('--image_loss_type', type=str, default='DHN', help="HashNet or DHN")
    parser.add_argument('--distance_type', type=str, default='Hamming',
                        help="Hamming, tSNE or Cauchy, Metric, Margin1, Margin2")
    parser.add_argument('--mission', type=str, default='Hashing',
                        help="Cross_Modal_Transfer, Hashing, Cross_Domain_Transfer")
    parser.add_argument('--log_dir', type=str, default='../log/', help="log dir")
    parser.add_argument('--lr_schedule_type', type=str, default='Stair_Step',
                        help="DANN_INV, HashNet_Step, ShapeNet_Step, Stair_Step, Timeseries_Step")

    parser.add_argument('--gamma', type=float, default=1.0, help="gamma")
    parser.add_argument('--sigmoid_param', type=float, default=1.0, help="sigmoid function")
    parser.add_argument('--radius', type=int, default=0, help="radius")

    parser.add_argument('--decay_step', type=int, default=200, help="decay_step")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="weight_decay")
    parser.add_argument('--similar_weight_type', type=str, default="config",
                        help="config: config-preset. auto: auto calc per batch, other number: manually")
    parser.add_argument('--norm_memory_batch', action="store_true", help="class_num")
    parser.add_argument('--batch_type', type=str, help="PairBatch, SemiMem, BatchInitMem, BatchSelectMem")
    parser.add_argument('--batch_params', type=str, default="300", help="params for batch type, optional")
    parser.add_argument('--semi_init_type', type=str, default="MODEL", help="MODEL, RANDOM")
    parser.add_argument('--semi_init_momentum', type=float, default=0.5, help="for debugs")
    parser.add_argument('--margin_params', type=str, default="0.5:0.8:1.0:1.0:2.0",
                        help="margin parameters: sim_in:dis_in:sim_out:dis_out:margin_radius")

    parser.add_argument('--fake_cpu_demo', type=str, default='False', help="use toy dataset to valid the process")
    parser.add_argument('--test_sample_ratio', type=float, default=0.1, help="class_num")
    parser.add_argument('--snapshot-interval', type=int, default=1000, help="the interval of snapshot")
    parser.add_argument('--opt-test', type=str, default='False', help='setting this will use the optimized evaluation')
    args = parser.parse_args()
    pprint(vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    use_gpu = torch.cuda.is_available()
    fake_cpu_demo = not use_gpu

    s_dataset = args.s_dataset
    t_dataset = s_dataset if args.t_dataset == '' else args.t_dataset
    batch_size = args.batch_size
    snap_path = args.snap_path
    hash_bit = args.hash_bit
    arg_lr = args.lr
    annotation = args.annotation
    loss_lambda = args.loss_lambda
    num_iterations = args.num_iters
    pre_model_path = args.pre_model_path
    fake_cpu_demo = args.fake_cpu_demo == 'True'
    radius = args.radius
    gamma = args.gamma
    sigmoid_param = args.sigmoid_param
    opt_test = args.opt_test == 'True'
    snapshot_interval = args.snapshot_interval
    test_sample_ratio = args.test_sample_ratio

    # network
    log_dir = args.log_dir
    image_network = args.image_network
    logger = get_log(log_dir, annotation)
    summary_writer = SummaryWriter('../runs/' + annotation)

    # Some Enum parameters
    if args.similar_weight_type == "auto":
        similar_weight = "auto"
    elif args.similar_weight_type == "config":
        similar_weight = data_config[s_dataset]["class_num"]
    else:
        similar_weight = float(args.similar_weight_type)
    batch_type = args.batch_type
    if batch_type:
        batch_type = BatchType[args.batch_type]
    else:
        batch_type = BatchType.SemiMem if similar_weight > 100 else BatchType.PairBatch
    lr_schedule_type = LrScheduleType[args.lr_schedule_type]
    batch_params = args.batch_params
    mission = Mission[args.mission]
    semi_init_type = SemiInitType[args.semi_init_type]
    image_loss_type = ImageLossType[args.image_loss_type]
    distance_type = DistanceType[args.distance_type]
    margin_params = MarginParams.from_string_params(args.margin_params)

    if not osp.exists(snap_path):
        os.system("mkdir -p " + snap_path)

    # Init model
    feature_net = get_feature_net(logger, s_dataset, hash_bit, image_loss_type=image_loss_type,
                                  network_name=image_network)
    if image_loss_type == ImageLossType.HashNet:
        loss_lambda = 0
    if pre_model_path != '':
        logger.info('load previous model: %s' % pre_model_path)
        model_instance = torch.load(pre_model_path)
    else:
        model_instance = MMHH(feature_net, hash_bit, trade_off=1.0,
                              use_gpu=use_gpu, loss_lambda=loss_lambda, similar_weight=similar_weight,
                              image_loss_type=image_loss_type, mission=mission, radius=radius,
                              distance_type=distance_type, gamma=gamma, sigmoid_param=sigmoid_param)

    # Prepare data
    if fake_cpu_demo:
        source_train_list, _ = get_fake_train_list(s_dataset, t_dataset)
    else:
        source_train_list = data_config[s_dataset]["train"]
    train_loader = data_config[s_dataset]["loader"](source_train_list, batch_size=batch_size,
                                                    resize_size=256, is_train=True, crop_size=224)
    # Set optimizer
    parameter_list = model_instance.get_parameter_list()
    lr_scheduler, optimizer, group_ratios = get_optimizer(logger, parameter_list, lr_schedule_type=lr_schedule_type,
                                                          init_lr=arg_lr, decay_step=args.decay_step,
                                                          weight_decay=args.weight_decay)
    semi_batch = None
    if batch_type in [BatchType.SemiMem, BatchType.BatchInitMem]:
        semi_batch = SemiBatch(len(train_loader.dataset), hash_bit, num_iterations,
                               init_momentum=args.semi_init_momentum, semi_init_type=semi_init_type,
                               model=model_instance, loader=train_loader, use_gpu=use_gpu)
    iter_batch = None
    all_st = time.perf_counter()

    logger.info("start train...")
    for iter_num in range(num_iterations):
        model_instance.set_train(True)

        iter_batch, inputs_batch, labels_batch, indices_batch = get_next_iter_with_index(logger, train_loader,
                                                                                         batch_size, iter_batch)
        if model_instance.use_gpu:
            inputs_batch, labels_batch = inputs_batch.cuda(), labels_batch.cuda()

        optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num, logger)
        optimizer.zero_grad()

        total_loss, hash_batch = model_instance.get_semi_hash_loss(logger, semi_batch, inputs_batch, labels_batch,
                                                                   iter_num, batch_type, batch_params, margin_params)
        total_loss.backward()
        optimizer.step()

        if batch_type in [BatchType.SemiMem, BatchType.BatchInitMem]:
            semi_batch.update_momentum(iter_num, batch_type, batch_params)
            semi_batch.update_memory(hash_batch, indices_batch, norm_memory_batch=args.norm_memory_batch)

    con_mAP = save_and_test(summary_writer, logger, model_instance, snap_path, annotation, s_dataset, t_dataset,
                            num_iterations, batch_size, use_gpu, radius=radius, opt_test=opt_test,
                            test_sample_ratio=test_sample_ratio)

    summary_writer.close()
    all_end = time.perf_counter()
    logger.info("finish train.")
    logger.info("All training is finished, total time: %.3f" % (all_end - all_st))
