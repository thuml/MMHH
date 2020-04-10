# coding=utf-8
from common.mmhh_config import ImageLossType, DistanceType, Mission, global_debugging, BatchType
from mmhh_loss import *
import torch


class MMHH(object):
    def __init__(self, feature_net, hash_bit,
                 trade_off=1.0, loss_lambda=0.1, use_gpu=True, distance_type=DistanceType.Hamming, similar_weight=1.,
                 image_loss_type=ImageLossType.HashNet, mission=Mission.Hashing, radius=2, gamma=1.0,
                 sigmoid_param=1.0):
        self.debug = global_debugging
        self.gamma = gamma
        self.hash_bit = hash_bit
        self.trade_off = trade_off
        self.loss_lambda = loss_lambda
        self.use_gpu = use_gpu
        self.is_train = False
        self.distance_type = distance_type
        self.similar_weight = similar_weight
        self.image_loss_type = image_loss_type
        self.mission = mission
        self.radius = radius
        self.sigmoid_param = sigmoid_param
        self.iter_num = 0
        if mission == Mission.Hashing:
            self.feature_network = feature_net

        if self.use_gpu:
            self.feature_network = self.feature_network.cuda()

    def get_semi_hash_loss(self, logger, semi_batch, inputs_batch, labels_batch, iter_num,
                           batch_type, batch_params, margin_params=None):
        self.iter_num = iter_num
        hash_batch = self.feature_network(inputs_batch)

        sigmoid_param = self.sigmoid_param
        if batch_type == BatchType.PairBatch:
            hash2, labels2 = hash_batch, labels_batch
        elif batch_type == BatchType.BatchInitMem:
            if iter_num < int(batch_params):
                hash2, labels2 = hash_batch, labels_batch
            else:
                hash2, labels2 = semi_batch.aug_memory, semi_batch.labels
        else:
            raise NotImplementedError("Wrong BatchType: " + str(batch_type))

        if self.distance_type == DistanceType.MMHH:
            hash_loss = mmhh_loss(hash_batch, hash2, labels_batch, labels2, margin_params,
                                  gamma=self.gamma, similar_weight=self.similar_weight)
        elif self.distance_type is DistanceType.Hamming:
            hash_loss = pairwise_loss(hash_batch, hash2, labels_batch, labels2,
                                      sigmoid_param=sigmoid_param, similar_weight=self.similar_weight)
        else:
            raise NotImplementedError
        # quantization loss
        if self.loss_lambda > 0:
            q_loss = quantization_loss(hash_batch)
        else:
            q_loss = 0
        if iter_num % 1 == 0:
            logger.info("Iter %05d hash loss %.5f, quan loss %.5f" % (iter_num, hash_loss.item(), q_loss.item()))
        return hash_loss + self.loss_lambda * q_loss, hash_batch.data

    def predict(self, inputs):
        return self.feature_network(inputs)

    def get_parameter_list(self):
        if self.mission in [Mission.Cross_Modal_Transfer, Mission.Cross_Domain_Transfer]:
            print("transfer, parameter involves d_net")
            return self.feature_network.get_parameter_list()
        elif self.mission == Mission.Hashing:
            return self.feature_network.get_parameter_list()
        else:
            raise NotImplementedError

    def set_train(self, mode):
        self.feature_network.train(mode)
        self.is_train = mode
