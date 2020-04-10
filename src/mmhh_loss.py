import torch
from torch.autograd import Variable

from common.mmhh_config import MarginParams


def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1.0, l_threshold=15.0, similar_weight=1.0):
    """
    Refer to https://github.com/thuml/HashNet
    :param outputs1:
    :param outputs2:
    :param label1:
    :param label2:
    :param sigmoid_param:
    :param l_threshold:
    :param similar_weight:
    :return:
    """
    if similar_weight == "auto":
        similar_weight = calc_similar_rate(label1, label2)
    assert similar_weight >= 0

    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)
    mask_dot = dot_product.data > l_threshold
    mask_exp = dot_product.data <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1 - similarity)
    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    loss = (torch.sum(torch.masked_select(exp_loss, Variable(mask_ep))) + torch.sum(
        torch.masked_select(dot_loss, Variable(mask_dp)))) * similar_weight + torch.sum(
        torch.masked_select(exp_loss, Variable(mask_en))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dn)))

    return loss / (torch.sum(mask_positive.float()) * similar_weight + torch.sum(mask_negative.float()))


def quantization_loss(outputs):
    return torch.sum(torch.log(torch.cosh(torch.abs(outputs) - 1))) / outputs.size(0)


def mmhh_loss(outputs1, outputs2, label1, label2, margin_params: MarginParams = None, gamma=1.0,
              similar_weight=1.0):
    if similar_weight == "auto":
        _, _, similar_weight = calc_similar_rate_triplet(label1, label2)
    # calculate similarity
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    weight_similarity = torch.abs(similarity - 1.0) + similarity * similar_weight

    dist_ham = calc_ham_dist(outputs1, outputs2)
    if margin_params:
        weight_similarity = margin_reweight(similarity, weight_similarity, dist_ham,
                                            margin_params.sim_in, margin_params.dis_in,
                                            margin_params.sim_out, margin_params.dis_out, margin_params.margin)
    alpha = 0.5
    probs = alpha * gamma / (dist_ham + gamma)

    loss_matrix = weight_similarity * (similarity * torch.log(1.0 / probs) +
                                       (1 - similarity) * torch.log(1.0 / (1.0 - probs)))
    return loss_matrix.mean()


def calc_similar_rate(label1, label2):
    similar_sum = torch.sum(torch.mm(label1.data.float(), label2.data.float().t()) > 0)
    dis_sum = label1.shape[0] * label2.shape[0] - similar_sum
    return float(dis_sum) / float(similar_sum)


def calc_similar_rate_triplet(label1, label2):
    similar_sum = torch.sum(torch.mm(label1.data.float(), label2.data.float().t()) > 0)
    dis_sum = label1.shape[0] * label2.shape[0] - similar_sum
    return similar_sum, dis_sum, float(dis_sum) / float(similar_sum)


def calc_ham_dist(outputs1, outputs2):
    ip = torch.mm(outputs1, outputs2.t())
    mod = torch.mm((outputs1 ** 2).sum(dim=1).reshape(-1, 1), (outputs2 ** 2).sum(dim=1).reshape(1, -1))
    cos = ip / mod.sqrt()
    hash_bit = outputs1.shape[1]
    dist_ham = hash_bit / 2.0 * (1.0 - cos)
    return dist_ham


def margin_reweight(similarity, weight_similarity, dist_hum, in_sim_weight, in_dis_weight, out_sim_weight,
                    out_dis_weight, margin):
    mask_in = dist_hum.data <= margin
    mask_out = dist_hum.data > margin
    mask_sim = similarity.data > 0
    mask_dis = similarity.data <= 0
    mask_in_sim = mask_in & mask_sim
    mask_in_dis = mask_in & mask_dis
    mask_out_sim = mask_out & mask_sim
    mask_out_dis = mask_out & mask_dis
    weight_similarity[mask_in_sim] *= in_sim_weight
    weight_similarity[mask_in_dis] *= in_dis_weight
    weight_similarity[mask_out_sim] *= out_sim_weight
    weight_similarity[mask_out_dis] *= out_dis_weight
    return weight_similarity
