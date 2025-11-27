import torch
import numpy as np
import argparse
import torch.nn as nn
import os
import _pickle as cPickle
import random
import math,copy
import torch.nn.functional as F
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast

from torch import nn
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager

@contextmanager
def null_context():
    yield

def is_blackout(matrix):
    arr = (np.sum(np.isnan(matrix).astype(int),axis=1) == matrix.shape[1])
    return arr.astype(int).sum() > 0


# def get_block_length(matrix):
#     num_missing = len(np.where(np.isnan(matrix))[0])
#     num_blocks = 0
#     for j in range(matrix.shape[1]):
#         temp = matrix[:,j]
#         for i in range(len(temp)-1):
#             if (np.isnan(temp[i]) and ~np.isnan(temp[i+1])):
#                 num_blocks += 1
#         if (np.isnan(temp[-1])):
#             num_blocks += 1
#     #num_blocks *= matrix.shape[1]
#     return int(num_missing/num_blocks)

# def get_block_length(matrix):
#     temp = np.where(np.isnan(matrix))
#     time = temp[0][0]
#     ts = temp[1][0]
#     i = 0
#     while (np.isnan(matrix[time+i,ts])):
#         i += 1
#     return i

# def get_block_length(matrix):
#     tss = np.unique(np.where(np.isnan(matrix))[1])
#     block_size = float('inf')
#     for ts in tss:
#         time = np.where(np.isnan(matrix[:,ts]))[0][0]
#         i = 0
#         while (time+i < matrix.shape[0] and np.isnan(matrix[time+i,ts])):
#             i += 1
#         block_size = min(block_size,i)
#     return int(block_size)



def make_validation (matrix,num_missing=20):
    """
    make_validation (robust):
    - Produces train_matrix (with validation masks), val_points, test_points.
    - Guarantees at least one test_point if none detected by constructing random blocks.
    """
    np.random.seed(0)
    test_points = []

    # collect contiguous missing block lengths per series (if any)
    temp = []
    for ts in range(matrix.shape[1]):
        nan_idx = np.where(np.isnan(matrix[:, ts]))[0]
        if nan_idx.size == 0:
            continue
        runs = np.split(nan_idx, np.where(np.diff(nan_idx) != 1)[0] + 1)
        for r in runs:
            temp.append(len(r))
    temp = np.array(temp, dtype=float)

    # choose block_size robustly
    if temp.size == 0:
        series_len = matrix.shape[0]
        block_size = max(1, int(max(1, series_len * 0.01)))
        block_size = min(block_size, max(50, series_len // 10))
    else:
        if temp.shape[0] > 10:
            block_size = temp[int(temp.shape[0] / 10):-int(temp.shape[0] / 10) - 1].mean()
        else:
            block_size = temp.mean()

    try:
        block_size = int(max(1, round(float(block_size))))
    except:
        block_size = 1

    w = max(1, int(10 * np.log10(max(1, block_size))))
    val_block_size = int(min(block_size, w))
    num_missing = int(max(1, int(num_missing / max(1, val_block_size))))

    train_matrix = copy.deepcopy(matrix)
    val_points = []

    # build validation points (one per timeseries by default)
    validation_points = np.random.uniform(0, max(0, matrix.shape[0] - val_block_size), (matrix.shape[1],)).astype(int)
    for i, x in enumerate(validation_points):
        end = min(x + val_block_size, matrix.shape[0])
        train_matrix[x:end, i] = np.nan
        val_points.append([x, i, end - x])

    # find true test points from original NaNs, if present
    test_possible_points = np.where(np.isnan(matrix))
    if len(test_possible_points[0]) > 0:
        i = 0
        while i < len(test_possible_points[0]):
            ts_number = test_possible_points[1][i]
            start_t = test_possible_points[0][i]
            j = 1
            while (i + j) < len(test_possible_points[0]) and test_possible_points[0][i + j] == test_possible_points[0][i] + j and test_possible_points[1][i + j] == ts_number:
                j += 1
            test_points.append([start_t, ts_number, j])
            i += j

    # If no test_points found, create random test blocks (one per timeseries up to a small cap)
    if len(test_points) == 0:
        num_ts = matrix.shape[1]
        cap = min(max(1, num_ts // 10), 20)  # create up to 20 test blocks or 10% of series
        rng = np.random.default_rng(0)
        chosen_ts = rng.choice(np.arange(num_ts), size=min(cap, num_ts), replace=False)
        for ts in chosen_ts:
            start = int(rng.integers(0, max(1, matrix.shape[0] - val_block_size)))
            end_len = min(val_block_size, matrix.shape[0] - start)
            test_points.append([start, int(ts), int(end_len)])

    return train_matrix, matrix, np.array(val_points), np.array(test_points), int(block_size), w



    # test_possible_points = np.where(np.isnan(matrix.T))
    # i = 0
    # while i < len(test_possible_points[0]):
    #     ts_number = test_possible_points[0][i]
    #     if (test_possible_points[1][i]+block_size < matrix.shape[0] and np.isnan(matrix[test_possible_points[1][i]+block_size,ts_number])):
    #         j = block_size
    #         while (test_possible_points[1][i]+j < matrix.shape[0] and np.isnan(matrix[test_possible_points[1][i]+j,ts_number])):
    #             j += 1
    #         test_points.append([test_possible_points[1][i],ts_number,j])
    #         i += j
    #     else :
    #         test_points.append([test_possible_points[1][i],ts_number,block_size])
    #         i += block_size
    # return train_matrix,matrix,np.array(val_points),np.array(test_points)

    # for i in range(matrix.shape[1]):
    #     j =0
    #     while j < matrix.shape[0]:
    #         if (np.isnan(matrix[j][i])):
    #             time = 0
    #             while j < matrix.shape[0] and np.isnan(matrix[j][i]):
    #                 time+= 1
    #                 j += 1
    #             test_points.append([j-time,i,time])
    #         else :
    #             j += 1
    # return train_matrix,matrix,np.array(val_points),np.array(test_points)


