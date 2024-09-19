import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import medianfilter,calculate_distance,calculate_time,calculate_speeds,get_staypoint,draw_on_map

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
# 问题一
# 读取用户000的轨迹数据
file_path = r'D:\anaconda\jupyter_flie\认知计算\实验一\Geolife Trajectories 1.3\Data\000\Trajectory\20081023025304.plt'  # 替换为具体路径
columns = ['维度', '经度', 'Zero', '海拔', '天数', '日期', '时间']
data = pd.read_csv(file_path, skiprows=6, names=columns)
# 绘制轨迹
plt.figure(figsize=(10, 6))
plt.scatter(data['维度'], data['经度'], marker='o')
plt.title('用户000在2008-10-23的运动轨迹')
plt.xlabel('维度')
plt.ylabel('经度')
plt.grid(True)
plt.show()

# 问题二
window_size = 7
data['维度'] = medianfilter(data['维度'].values,window_size)
data['经度'] = medianfilter(data['经度'].values,window_size)
data['海拔'] = medianfilter(data['海拔'].values,window_size)
# 绘制平滑后轨迹
plt.figure(figsize=(10, 6))
plt.scatter(data['维度'], data['经度'], marker='o')
plt.title('用户000在2008-10-23的运动轨迹')
plt.xlabel('维度')
plt.ylabel('经度')
plt.grid(True)
plt.show()
import folium

m = draw_on_map(data)
m.save("trajectory_map.html")

# 问题三
# 使用距离前一个点的距离和时间差计算速度
for i in range(1, len(data)):
    d = calculate_distance((data.iloc[i-1]['维度'], data.iloc[i-1]['经度']),
                           (data.iloc[i]['维度'], data.iloc[i]['经度']))
    t = calculate_time(data.iloc[i-1]['时间'], data.iloc[i]['时间'])
    data.loc[i,'速度'] = calculate_speeds(d, t)
data.loc[0,'速度'] = 4 # 给一个初始速度，防止被误判为停留点
# 去除异常点（速度大于120km/h）
data = data.loc[data['速度'] <= 120]
data = data.reset_index(drop=True)
def get_staypoint(data, min_speed):
    return data[data['速度'] < min_speed]
data_staypoint = get_staypoint(data, 1) # 人的步行速度是3-6km/h
plt.figure(figsize=(10, 6))
plt.scatter(data_staypoint['维度'], data_staypoint['经度'], marker='o')
plt.title('用户000在2008-10-23的停留点图')
plt.xlabel('维度')
plt.ylabel('经度')
plt.grid(True)
plt.show()
# 使用dbscan对停留点聚类获得经常访问的点
from  sklearn.cluster import DBSCAN
# def get_clusterpoints(data, eps, min_samples):
db = DBSCAN(eps=0.5, min_samples=12,metric=calculate_distance) # 半径参数设为500m,样本量为12
db.fit(data_staypoint[['维度','经度']])
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
            ).add_to(m)
m.save("cluster_centers.html")