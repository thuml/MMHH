import time

import argparse
import os
import os.path as osp

import numpy as np
import torch
from torch.autograd import Variable

from common.fake_demo import get_fake_test_list
from common.logger import get_log
from common.mmhh_config import data_config
from evaluate.measure_utils import \
    get_precision_recall_by_Hamming_Radius_optimized, get_precision_recall_by_Hamming_Radius, \
    mean_average_precision_normal, mean_average_precision_normal_optimized_topK


def save_and_test(summary_writer, m_logger, model_instance, snap_path, annotation, s_dataset, t_dataset, iter_num,
                  batch_size, use_gpu, fake_cpu_demo=False, radius=0, opt_test=True, test_sample_ratio=1.):
    """
    evaluate the performance during training
    :param model_instance:
    :param snap_path:
    :param annotation:
    :param s_dataset:
    :param t_dataset:
    :param iter_num:
    :param batch_size:
    :param use_gpu:
    :return:
    """
    torch.save(model_instance,
               snap_path + "{:s}_{:s}_{:s}_iter_{:05d}".format(annotation, s_dataset, t_dataset, iter_num))
    m_logger.info(snap_path + "{:s}_{:s}_{:s}_iter_{:05d}".format(annotation, s_dataset, t_dataset, iter_num))
    eval_st = time.perf_counter()
    m_config = get_test_config(t_dataset, batch_size)
    if fake_cpu_demo:
        print("fake test list")
        get_fake_test_list(m_config, t_dataset)

    model_instance.set_train(False)
    con_mAP, con_time = evaluate(m_logger, m_config, model_instance, use_gpu,
                                 '../test_output/{:s}_iter_{:05d}'.format(annotation, iter_num),
                                 radius, opt_test, test_sample_ratio)
    summary_writer.add_scalar('Test/conMAP', con_mAP, iter_num)
    summary_writer.add_scalar('Test/con_time', con_mAP, con_time)

    model_instance.set_train(True)
    eval_end = time.perf_counter()
    m_logger.info("iter_num %d, map: %.3f, use time: %.3f" %
                  (iter_num, con_mAP, (eval_end - eval_st)))
    return con_mAP


def save_code_and_label(params, path):
    database_code = params['database_code']
    validation_code = params['test_code']
    database_labels = params['database_labels']
    validation_labels = params['test_labels']
    np.save(path + "/database_code.npy", database_code)
    np.save(path + "/database_labels.npy", database_labels)
    np.save(path + "/test_code.npy", validation_code)
    np.save(path + "/test_labels.npy", validation_labels)


def code_predict(loader, model, name, use_gpu=True):
    start_test = True

    iter_val = iter(loader[name])
    print("name: %s; length: %d" % (name, len(loader[name])))
    display_interval = 100
    for i in range(len(loader[name])):
        if i % display_interval == 0:
            print("iter: %d" % i)
        data = iter_val.next()
        inputs = data[0]
        labels = data[1]
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        outputs = model.predict(inputs)
        if start_test:
            all_output = outputs.data.cpu().float()
            all_label = labels.float()
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)
    return all_output, all_label


def predict(config, model_instance, use_gpu, test_sample_ratio=1.0):
    dset_loaders = {}
    data_config = config["data"]

    print("loading base list")
    dset_loaders["database"] = config["loader"](data_config["database"]["list_path"],
                                                batch_size=data_config["database"]["batch_size"], resize_size=256,
                                                is_train=False, test_sample_ratio=test_sample_ratio)
    print("loading test list")
    dset_loaders["test"] = config["loader"](data_config["test"]["list_path"],
                                            batch_size=data_config["test"]["batch_size"], resize_size=256,
                                            is_train=False, test_sample_ratio=test_sample_ratio)
    print("start database predict")
    database_codes, database_labels = code_predict(dset_loaders, model_instance, "database",
                                                   use_gpu=use_gpu)
    print("start test predict")
    test_codes, test_labels = code_predict(dset_loaders, model_instance, "test",
                                           use_gpu=use_gpu)
    print("done predict")

    return {"database_code": database_codes.numpy(), "database_labels": database_labels.numpy(),
            "test_code": test_codes.numpy(), "test_labels": test_labels.numpy()}


def get_test_config(dataset, batch_size):
    test_config = {
        "prep": {"resize_size": 256, "crop_size": 224},
        "dataset": dataset,
        "batch_size": batch_size,
        "data": {
            "database": {
                "list_path": data_config[dataset]['database'],
                "batch_size": batch_size},
            "test": {
                "list_path": data_config[dataset]['test'],
                "batch_size": batch_size}
        },
        "R": data_config[dataset]['R'],
        "loader": data_config[dataset]["loader"]}
    return test_config


def evaluate(logger, config, model_instance, use_gpu, output_path=None, radius=0, opt_test=True,
             test_sample_ratio=1.0):
    print('R=%d' % int(config["R"]))
    # prepare data
    code_and_label = predict(config, model_instance, use_gpu, test_sample_ratio=test_sample_ratio)
    query_output = code_and_label["test_code"]
    query_labels = code_and_label["test_labels"]
    database_output = code_and_label["database_code"]
    database_labels = code_and_label["database_labels"]
    # sign_query_output = np.sign(query_output)
    # sign_database_output = np.sign(database_output)

    if output_path is not None:
        print("saving to %s" % output_path)
        if not osp.exists(output_path):
            os.system("mkdir -p " + output_path)
        save_code_and_label(code_and_label, output_path)
    st = time.time()
    con_mAP = evaluate_measurement(logger, database_output, database_labels, query_output, query_labels,
                                   config["R"], radius, opt_test)
    con_time = time.time() - st
    return con_mAP, con_time


def evaluate_measurement(logger, database_output, database_labels, query_output, query_labels, topR, radius, opt_test):
    if radius > 0:
        print("evaluate hamming %d" % radius)
        if opt_test:
            logger.info("radius {:d}, optimized evaluation".format(radius))

            precs, recs, mAP, _ = get_precision_recall_by_Hamming_Radius_optimized(
                database_output, database_labels, query_output, query_labels, radius=radius)
        else:
            logger.info("radius {:d}, primary evaluation".format(radius))
            precs, recs, mAP = get_precision_recall_by_Hamming_Radius(
                database_output, database_labels, query_output, query_labels, radius)
    else:
        print("no lookup, hamming %d" % radius)
        if opt_test:
            logger.info("linear scan, optimized evaluation")
            precs, recs, mAP = mean_average_precision_normal_optimized_topK(
                database_output, database_labels, query_output, query_labels, topR)
        else:
            logger.info("linear scan, primary evaluation")
            precs, recs, mAP = mean_average_precision_normal(
                database_output, database_labels, query_output, query_labels, topR)
    return mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='shapenet_13', help="dataset name")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--output_path', type=str, help="path to save the code and labels")
    parser.add_argument('--model_path', type=str, help="model path")
    parser.add_argument('--radius', type=int, default=2, help="radius")
    parser.add_argument('--opt-test', action='store_true', help='setting this will use the optimized evaluation')
    parser.add_argument('--log_dir', type=str, default='../log/', help="log dir")
    parser.add_argument('--annotation', type=str, default='empty', help="annotation for distinguishing")
    parser.add_argument('--test_sample_ratio', type=float, default=1.0, help="sample ratio to test")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    use_gpu = torch.cuda.is_available()
    output_path = args.output_path
    model_path = args.model_path
    dataset = args.dataset
    batch_size = args.batch_size
    radius = args.radius
    opt_test = args.opt_test
    log_dir = args.log_dir
    annotation = args.annotation
    test_sample_ratio = args.test_sample_ratio

    config = get_test_config(dataset, batch_size)
    model_instance = torch.load(args.model_path)
    model_instance.set_train(False)
    print("calc mean_average_precision")
    logger = get_log("../", annotation)
    con_mAP, use_time = evaluate(logger, config, model_instance, use_gpu, output_path, radius,
                                 opt_test=opt_test, test_sample_ratio=test_sample_ratio)
    logger.info("mAP: %.3f, use time: %.3f" % (con_mAP, use_time))
    print("saving done")
