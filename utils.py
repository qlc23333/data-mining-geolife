import numpy as np
import folium
import math
from datetime import datetime
from geopy import distance

def medianfilter(X, window_size):
    n = len(X)
    X = np.append(np.zeros(int(window_size / 2)), X)
    X = np.append(X, np.zeros(int(window_size / 2)))
    new_X = np.zeros(n)
    for i in range(n):
        new_X[i] = np.median(X[i:i + window_size])
    return new_X

def haversine(lat1, lon1, lat2, lon2):
    """
    计算两点在地球表面上的大圆距离（哈弗辛公式）
    :param lat1: 点1的纬度（度）
    :param lon1: 点1的经度（度）
    :param lat2: 点2的纬度（度）
    :param lon2: 点2的经度（度）
    :return: 两点之间的地球表面距离（公里）
    """
    R = 6371  # 地球半径，单位：公里
    # 将经纬度转换为弧度
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    # 经度和纬度的差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # 哈弗辛公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    # 大圆距离
    distance = R * c
    return distance

def calculate_distance(piont1, point2):
    # # 计算地球表面上的距离
    # surface_distance = haversine(lat1, lon1, lat2, lon2)
    # # 计算高度差
    # height_diff = h2 - h1
    # # 计算直线距离
    # straight_distance = math.sqrt(surface_distance**2 + (height_diff / 1000)**2)  # 高度差转换为公里
    straight_distance = distance.distance(piont1,point2).miles * 1.60934
    return straight_distance

def calculate_time(t1, t2):
    t1 = datetime.strptime(t1, '%H:%M:%S')
    t2 = datetime.strptime(t2, '%H:%M:%S')
    return (t2 - t1).total_seconds() / 3600.0

def calculate_speeds(d,t):
    return d / t

def get_staypoint(data, min_speed):
    return data[data['速度'] < min_speed]

def draw_on_map(data, polyline=True, marker=False):
    # 创建一个地图对象，初始中心设置为轨迹数据的第一个点
    m = folium.Map(location=[data['维度'].mean(), data['经度'].mean()], zoom_start=12)
    # 将轨迹数据添加到地图中，用Polyline绘制轨迹线
    if polyline == True:
        folium.PolyLine(
            locations=list(zip(data['维度'], data['经度'])),  # 经度和纬度
            color="blue",  # 轨迹线的颜色
            weight=6,  # 轨迹线的宽度
        ).add_to(m)
    # 如果想在轨迹点添加标记，可以用for循环逐个添加
    if marker == True:
        for i in range(len(data)):
            folium.Marker(
                location=[data.iloc[i]['维度'], data.iloc[i]['经度']],
                # popup=f"速度: {data.iloc[i]['速度']} km/h"  # 可以在标记中显示速度信息
            ).add_to(m)

    # 将地图保存成HTML文件
    return m
