import numpy as np
import torch
from progress.bar import Bar
from copy import deepcopy

"""
Modified from https://github.com/ssppp/GreedyHash/blob/master/cal_map.py
by Fangrui Liu
fangrui.liu@ubc.ca
"""

def compress(retrieval, query, model, classes=10, multi_label=True, device="cpu"):
    retrievalB = []
    retrievalL = []
    queryB = []
    queryL = []
    model.eval()
    with torch.no_grad():
        with Bar("Accumulating Retrieval Code", max=len(retrieval)) as bar:
            for batch_step, (data, target) in enumerate(retrieval):
                var_data = data.to(device)
                code = model(var_data)[-1]
                retrievalB.append(deepcopy(code))
                retrievalL.append(deepcopy(target))
                del code, target
                bar.next()

        with Bar("Accumulating Retrieval Code", max=len(query)) as bar:
            for batch_step, (data, target) in enumerate(query):
                var_data = data.to(device)
                code = model(var_data)[-1]
                queryB.append(deepcopy(code))
                queryL.append(deepcopy(target))
                del code, target
                bar.next()

        retrievalB = torch.cat(retrievalB, dim=0)
        if multi_label:
            retrievalL = torch.cat(retrievalL, dim=0)
        else:
            retrievalL = torch.eye(classes)[torch.cat(retrievalL, dim=0)]

        queryB = torch.cat(queryB, dim=0)
        if multi_label:
            queryL = torch.cat(queryL, dim=0)
        else:
            queryL = torch.eye(classes)[torch.cat(queryL, dim=0)]
    return retrievalB, retrievalL, queryB, queryL


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.size(1) # max inner product value
    distH = 0.5 * (q - torch.matmul(B1, B2.transpose(0, 1)))
    return distH


def calculate_map(qB, rB, queryL, retrievalL, device="cpu"):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.size(0)
    map = 0
    with torch.no_grad():
        with Bar("Calculating MAP", max=num_query) as bar:
            for iter in range(num_query):
                # gnd : check if exists any retrieval items with same label
                gnd = torch.gt(torch.matmul(queryL[iter, :], retrievalL.transpose(0, 1)), 0).float()
                # tsum number of items with same label
                tsum = torch.sum(gnd).int()
                if tsum == 0:
                    continue
                # sort gnd by hamming dist
                hamm = calculate_hamming(qB[iter, :], rB)
                ind = torch.argsort(hamm)
                gnd = gnd[ind]

                count = torch.linspace(1, tsum, tsum).to(device) # [1,2, tsum]
                tindex = torch.where(gnd == 1)[0] + 1.0
                map_ = torch.mean(count / tindex.to(device))
                # print(map_)
                map = map + map_
                bar.next()
        map = map / num_query
        return map.item()


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.size(0)
    topkmap = 0
    with torch.no_grad():
        with Bar("Calculating TopK MAP", max=num_query) as bar:
            for iter in range(num_query):
                gnd = torch.gt(torch.matmul(queryL[iter, :], retrievalL.transpose(0, 1)), 0).float()
                hamm = calculate_hamming(qB[iter, :], rB)
                ind = torch.argsort(hamm)
                gnd = gnd[ind]

                tgnd = gnd[0:topk]
                tsum = torch.sum(tgnd).int()
                if tsum == 0:
                    continue
                count = torch.linspace(1, tsum, tsum)

                tindex = torch.where(tgnd == 1)[0] + 1.0
                topkmap_ = torch.mean(count / (tindex))
                # print(topkmap_)
                topkmap = topkmap + topkmap_
                bar.next()
        topkmap = topkmap / num_query
        return topkmap.item()

def mean_average_precision(validation_code, database_code, validation_labels, database_labels, R):
    
    query_num = validation_code.shape[0]

    sim = torch.matmul(database_code, validation_code.T)
    APx = []
    with torch.no_grad():
        with Bar("Calculating TopK MAP", max=query_num) as bar:
            for i in range(query_num):
                label = validation_labels[i, :]
                label[label == 0] = -1
                idx = torch.argsort(-sim[:, i])
                imatch = torch.sum((database_labels[idx[0:R], :] == label).int(), dim=1) > 0
                relevant_num = torch.sum(imatch)
                Lx = torch.cumsum(imatch.flatten(), 0)
                Px = Lx.float() / torch.arange(1, R+1, 1)
                if relevant_num != 0:
                    APx.append(torch.sum(Px * imatch) / relevant_num)
                bar.next()
        return torch.mean(torch.as_tensor(APx)).item()