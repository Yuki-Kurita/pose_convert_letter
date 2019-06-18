# -*- coding:utf-8 -*-
import socket
import time
import os

def get_coo_vec():
    data_list = os.listdir("./imp/data")
    coo_data = {}
    coo_vec = {}
    vec_list = [['LCS-X', '4X', '2X'], ['LCS-Y', '4Y', '2Y'], ['LSE-X', '5X', '4X'], ['LSE-Y', '5Y', '4Y'],\
    ['LEH-X', '7X', '5X'], ['LEH-Y', '7Y', '5Y'], ['RCS-X', '8X', '2X'], ['RCS-Y', '8Y', '2Y'],\
    ['RSE-X', '9X', '8X'], ['RSE-Y', '9Y', '8Y'], ['REH-X', '11X', '9X'], ['REH-Y', '11Y', '9Y'], ['CS-X', '1X', '2X'], ['CS-Y', '1Y', '2Y'],\
    ['SLH-X', '12X', '1X'], ['SLH-Y', '12Y', '1Y'], ['LHK-X', '13X', '12X'], ['LHK-Y', '13Y', '12Y'], ['LKA-X', '14X', '13X'], ['LKA-Y', '14Y', '13Y'],\
    ['SRH-X', '16X', '1X'], ['SRH-Y', '16Y', '1Y'], ['RHK-X', '17X', '16X'], ['RHK-Y', '17Y', '16Y'], ['RKA-X', '18X', '17X'], ['RKA-Y', '18Y', '17Y']]
    latest_file = max([d.split('.')[0] for d in data_list]) + '.txt'
    with open('./imp/data/' + latest_file) as f:
        # 各関節をディクショナリに格納
        for line in f:
            coo_data[line.split(',')[0] + 'X'] = (line.split(',')[1]).replace("\n", '')
            coo_data[line.split(',')[0] + 'Y'] = (line.split(',')[2]).replace("\n", '')

        # 各関節に対するベクトルを求める
        for vec in vec_list:
            coo_vec[vec[0]] = int(coo_data[vec[1]]) - int(coo_data[vec[2]])
    return coo_vec

vec_list = ['LCS-X', 'LCS-Y', 'LSE-X', 'LSE-Y','LEH-X','LEH-Y', 'RCS-X', 'RCS-Y', 'RSE-X', 'RSE-Y', 'REH-X', 'REH-Y', 'CS-X', 'CS-Y',\
'SLH-X', 'SLH-Y', 'LHK-X', 'LHK-Y', 'LKA-X', 'LKA-Y', 'SRH-X', 'SRH-Y', 'RHK-X', 'RHK-Y', 'RKA-X', 'RKA-Y']
while(1):
    c = []
    coo_vec = get_coo_vec()
    for vec in vec_list:
        c.append(coo_vec[vec])
    print(c)
    time.sleep(5)
