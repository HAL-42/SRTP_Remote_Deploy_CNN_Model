#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: server.py.py
@time: 2019/5/18 10:10
@desc:
"""


import socketserver
import cv2
import numpy as np
import os
from caffe_predict import *
from socket_send_recv import *
import time
import logging
from multiprocessing import Pool
import socket

from typing import Optional

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
kHeadLen = 16

#caffe
kCaffeRoot = os.path.expanduser('~/caffe/') # change with your install location
kModelName = 'dcc_crowdnet'
kModelPath = os.path.expanduser(os.path.join('..','models', kModelName))
kDataPath = os.path.expanduser(os.path.join('..','data', kModelName))
kWeightsPath = os.path.expanduser(os.path.join('..','weight', kModelName))
kModelWeight = os.path.join(kWeightsPath, 'dcc_crowdnet_train_iter_139000.caffemodel')
kModelDef = os.path.join(kModelPath, 'deploy_addCAFFE.prototxt')

kOutLayer = 'conv6'
kBatchSize = 40
kSliceSchemeOn = True

kHasGPU = True
kGPUId = 0

#constant
kPatchW = 225
kPatchH= 225

kNetDensityH = 28
kNetDensityW = 28

kLoop = 10

#debug
kIsOnline = False

# Caffe抽风，每次预测都需要单开一个进程。和网络关系不大，可以不看
# 调用的函数参见 caffe_predict.py data_preprocessing.py
def _MultiProcessPredict(img):
    InitCaffeEnv(os.path.expanduser('~/caffe/'), kHasGPU, kGPUId)
    net = LoadCaffeModel(kModelDef, kModelWeight)
    if kSliceSchemeOn:
        imgs = [img]
    else:
        print("Not an Good Idea to Use this Function")
        imgs = AdaptImgForCaffeImgScheme([img])
    predict_densities = PredictImgsByCaffe(imgs, net, kOutLayer, kPatchW, kPatchH,
                                           kNetDensityW, kNetDensityH, kBatchSize,
                                           slice_scheme_on=kSliceSchemeOn)
    return predict_densities


class MyTCPHandler(socketserver.BaseRequestHandler):
    # 每次有TCP连接过来，都会调用一次handle函数
    # self.request 可以简单认为是 sock
    def handle(self):
        print("Receive an TCP Request From " + str(self.client_address))

        # 没卵用，设置了依旧不是阻塞模式
        self.request.setblocking(True)
        # Debug
        print("Recv Buffer Size = ", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))

        # 我也忘了为什么要sleep，但应该有用
        time.sleep(1)
        print("Get An Reuquest From Client")

        # 计数代码，用来计算帧率和两帧之间时间间隔
        last_time = time.time()
        loop_index = time_sum = time_gap = 0
        while True:
            # 收取图像，先收取data_len，就是图像字节流的长度，再读取图像
            data_len = recv_data_len(self.request)
            if not data_len:
                break
            img = recv_img(self.request, data_len)
            if not isinstance(img, np.ndarray):
                break

            print("Receive an Image")
            print("Size of Img:" + str(img.shape))

            # 计算帧率
            time_gap = time.time() - last_time
            print("Time Beteen Two Img: " + str(time_gap))
            last_time = time.time()
            time_sum += time_gap
            loop_index = (loop_index + 1) % kLoop
            if not loop_index:
                print("fps: " + str(kLoop / time_sum))
                time_sum = 0

            # Compute Density
            # 如果是本地运行，或者不想涉及Caffe，请把上面常数kIsOnline设置为False
            # 此时密度图是随机生成的高斯分布，大小为原图像1/8
            if kIsOnline:
                pool = Pool(processes=1)
                res = pool.apply_async(_MultiProcessPredict, (img,))
                pool.close()
                pool.join()

                predict_densities = res.get()
            else:
                predict_densities = DummyPredictImgsByCaffe([img])

            # debug
            print("Here is the Predict Density")
            print(predict_densities[0].dtype)
            print(predict_densities[0])
            print(predict_densities[0].shape)

            # Send Back density to client
            if 0 != send_density(self.request, predict_densities[0]):
                break
        print("Connection End At Clinet!")


def StartServer(addr: str, port: int):
    try:
        # 开启多线程服务器，一次连接一个线程
        server = socketserver.ThreadingTCPServer((addr, port), MyTCPHandler)
        print("Server is Listening at " + str(addr) + ":" + str(port))
        server.serve_forever()
    except Exception as err:
        print(err)
        print("Server Failed to Listen at " + str(addr) + ":" + str(port))
        server.socket.close()
        return None

if __name__ == '__main__':
    StartServer('', 12345)

# def InitServer(addr: str, port: int):
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     try:
#         sock.bind((addr, port))
#         sock.listen(100)
#     except Exception as err:
#         print(err)
#         print("Server Failed to Listen at " + str(addr) + ":" + str(port))
#         return None
#     else:
#         print("Server is Listening at " + str(addr) + ":" + str(port))
#         return sock