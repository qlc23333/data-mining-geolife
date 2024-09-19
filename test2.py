import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from  sklearn.cluster import DBSCAN
from utils import medianfilter,calculate_distance,calculate_time,calculate_speeds,get_staypoint
import folium

plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False
path = r"D:\anaconda\jupyter_flie\认知计算\实验一\Geolife Trajectories 1.3\Data\000\Trajectory"
columns = ['维度', '经度', 'Zero', '海拔', '天数', '日期', '时间']
# data_stay = pd.DataFrame(columns=columns)
data_stay = {}
for file in os.listdir(path):
    data = pd.read_csv(os.path.join(path, file), skiprows=6, names=columns)
    data = data.dropna(axis=1)
    # 平滑
    window_size = 7
    data['维度'] = medianfilter(data['维度'].values,window_size)
    data['经度'] = medianfilter(data['经度'].values,window_size)
    data['海拔'] = medianfilter(data['海拔'].values,window_size)
     # 计算速度
    # 使用距离前一个点的距离和时间差计算速度
    for i in range(1, len(data)):
        d = calculate_distance((data.iloc[i-1]['维度'], data.iloc[i-1]['经度']),
                               (data.iloc[i]['维度'], data.iloc[i]['经度']))
        t = calculate_time(data.iloc[i-1]['时间'], data.iloc[i]['时间'])
        data.loc[i,'速度'] = calculate_speeds(d, t)
    data.loc[0,'速度'] = 4 # 给一个初始速度，防止被误判为停留点
    # 去除异常点
    data = data.loc[data['速度'] <= 120]
    data = data.reset_index(drop=True)
    data_staypoint = get_staypoint(data, 2) # 人的步行速度是3-6km/h
    # data_stay = pd.concat([data_stay, data_staypoint], axis=0, ignore_index=True)
    data_stay[str(file).replace('.plt','')] = data_staypoint

path = r'D:\anaconda\jupyter_flie\认知计算\实验一\000_stay'
for file in data_stay.keys():
    data_stay[file].to_csv(os.path.join(path,file+'.csv'),index=False)

columns = ['维度', '经度', 'Zero', '海拔', '天数', '日期', '时间', '速度']
data_staypoint = pd.DataFrame(columns=columns)
for i, file in enumerate(list(os.listdir(path))):
    if i == 30:
        break
    data = pd.read_csv(os.path.join(path, file),skiprows=1,names=columns)
    data = data.dropna(axis=1)
    data_staypoint = pd.concat([data_staypoint, data], ignore_index=True)
db = DBSCAN(eps=0.5, min_samples=100, metric=calculate_distance)
db.fit(data_staypoint[['维度', '经度']])
# 可视化聚集点
labels = db.labels_
# 将标签添加到原始数据中
data_staypoint.loc[:,'cluster'] = labels
# 将不同簇映射为不同颜色
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
plt.figure(figsize=(8, 6))
cluster_centers = {}
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 黑色用于噪声
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = data_staypoint[class_member_mask]
    plt.plot(xy['维度'], xy['经度'], 'o', color=tuple(col), markersize=10)
    # 绘制中心点
    if k != -1:
        # 计算簇的中心点（均值）
        cluster_center = xy[['维度', '经度']].mean().values
        cluster_centers[k] = cluster_center
        # 绘制簇的中心点
        plt.plot(cluster_center[0], cluster_center[1], 'x', color='red', markersize=10, mew=3)
plt.xlabel('维度')
plt.ylabel('经度')
plt.show()
cluster_centers = np.array(list(cluster_centers.values()))
m = folium.Map(location=[cluster_centers[:,0].mean(), cluster_centers[:,1].mean()], zoom_start=12)
for i in range(len(cluster_centers)):
            folium.Marker(
                location=[cluster_centers[i,0], cluster_centers[i,1]],
                # popup=f"速度: {data.iloc[i]['速度']} km/h"  # 可以在标记中显示速度信息
            ).add_to(m)
m.save("cluster_centers.html")
