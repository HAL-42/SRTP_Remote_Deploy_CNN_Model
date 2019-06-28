#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: socket_send_recv.py
@time: 2019/5/19 13:02
@desc:
"""
import cv2
import numpy as np
import functools
import time
# import socket
# import time

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
kHeadLen = 16
kMTU = 1024

kSendAll = False
kRecvAll = False

# 收取数据头（字符化的int，指出之后又多少字节的数据）
def recv_data_len(sock):
    recv = sock.recv(kHeadLen)
    if not recv:
        return None
    return int(recv.decode())


def recv_img(sock, data_len, is_density: bool=False):
    # 如果是密度图，还要收取密度图的长宽
    if is_density:
        density_h = int(sock.recv(kHeadLen).decode())
        density_w = int(sock.recv(kHeadLen).decode())
    string_data = b''
    # 发送-确认模式，发送端每发送1024字节就停止等待，直到对方发送确认信号
    if not kRecvAll:
        index = 0
        while index + kMTU <= data_len:
            string_data = string_data + sock.recv(kMTU)
            index += kMTU
            sock.send(str(index).ljust(kHeadLen).encode())
        else:
            string_data = string_data + sock.recv(data_len - index)
    else:
        # 流水线，发送端一次发出所有数据
        while len(string_data) < data_len:
            string_data = string_data + sock.recv(kMTU)
            # Debug
            # print("Current Data_len =", len(string_data))
        # string_data = sock.recv(100000000)
        # index = 0
        # while index + kMTU <= data_len:
        #     string_data = string_data + sock.recv(kMTU)
        #     index += kMTU
        #     # Debug
        #     print("Current Data_len =", len(string_data))
        #     print("Current index =", index)
        # else:
        #     string_data = string_data + sock.recv(data_len - index)

    if not string_data:
        return None

    #debug
    if data_len != len(string_data):
        print("Error May Occurred at recv_img Function!")
        print("Receive An Img with String Length of: " + str(len(string_data)))
        print("However, the DataLen is: " + str(data_len))
        return None
        # print("Try to sleep until peer resent finished")
        # print("Current Index = ", index)
        # time.sleep(10)
        # while index + kMTU <= data_len:
        #     string_data = string_data + sock.recv(kMTU)
        #     index += kMTU
        # else:
        #     string_data = string_data + sock.recv(data_len - index)
        # if data_len != len(string_data):
        #     print("Error Still!!!")
        #     print("Receive An Img with String Length of: " + str(len(string_data)))
        #     print("However, the DataLen is: " + str(data_len))
        #     return None

    if is_density:
        # 恢复为密度图像
        data = np.frombuffer(string_data, dtype='f4').reshape(density_h, density_w)
        return data
    else:
        # 恢复为图像（字节流反序列化+opencv解码）
        data = np.frombuffer(string_data, dtype=np.uint8).reshape(data_len, 1)
        return cv2.imdecode(data, 2|4)


# 偏函数
recv_density = functools.partial(recv_img, is_density=True)


# 发送图像
def send_img(sock, img: np.ndarray, is_density: bool=False):
    if is_density:
        # Serialization
        # 序列化为字节流，发送
        string_data = img.tostring()
    else:
        # Compression and Serialization
        # 编码为jpg再序列化，发送
        encoded_img = cv2.imencode('.jpg', img, encode_param)[1]
        string_data = encoded_img.tostring()

    # Send Data Length
    data_len = len(string_data)

    # Debug
    print("Sending string_data len is ", data_len)

    sock.send(str(data_len).ljust(kHeadLen).encode())
    # Send the Shape of density map
    if is_density:
        sock.send(str(img.shape[0]).ljust(kHeadLen).encode())
        sock.send(str(img.shape[1]).ljust(kHeadLen).encode())

    #sock.send(string_data)
    index = 0
    # Send 1024 byte per time
    # After send 1024 byte, waiting for the confirm message from the server
    if not kSendAll:
        while index + kMTU <= data_len:
            sock.send(string_data[index:index + kMTU])
            # debug
            # print("Send from" + str(index) + '->')
            index += kMTU
            if not sock.recv(kHeadLen):
                print("Can't Receive Confirm pack when index=" + str(index))
                return None
        else:
            sock.send(string_data[index:data_len])
            # debug
            # print("Send from" + str(index) + '->')
    else:
        # 直接发送模式
        send_info = sock.send(string_data)
        print("Send Info =", send_info)

    return 0


# 偏函数
send_density = functools.partial(send_img, is_density=True)