import configparser
import sys

from fair_clustering import fair_clustering
from util.configutil import read_list
import matplotlib.pyplot as plt
import numpy as np
config_file = "config/example_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

# Create your own entry in `example_config.ini` and change this str to run
# your own trial
config_str = "dataset" if len(sys.argv) == 1 else sys.argv[1]

print("Using config_str = {}".format(config_str))

# Read variables
data_dir = config[config_str].get("data_dir")
#print(data_dir)
dataset = config[config_str].get("dataset")
dataset_name = dataset
clustering_config_file = config[config_str].get("config_file")
num_clusters = list(map(int, config[config_str].getlist("num_clusters")))
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")
violating = config["DEFAULT"].getboolean("violating")
violation = config["DEFAULT"].getfloat("violation")
print(dataset)
print(clustering_config_file)
shiyan = [0]
COLOR_BLIND_POF = []
ALG_GF_POF = []
Oral_GFTOGFDS_POF = []
SNC_GFTOGFDS_POF = []

COLOR_BLIND_GFVio = []
ALG_GF_GFVio = []
Oral_GFTOGFDS_GFVio = []
SNC_GFTOGFDS_GFVio = []

COLOR_BLIND_DSVio = []
ALG_GF_DSVio = []
Oral_GFTOGFDS_DSVio = []
SNC_GFTOGFDS_DSVio = []
for n_clusters in num_clusters:
    C_B_POF, A_G_POF, O_G_POF, S_G_POF, C_B_GFVio, A_G_GFVio, O_G_GFVio, S_G_GFVio, C_B_DSVio, A_G_DSVio, O_G_DSVio, S_G_DSVio = fair_clustering(dataset, clustering_config_file, data_dir, n_clusters, deltas, max_points, violating, violation, shiyan)

    COLOR_BLIND_POF.append(C_B_POF)
    ALG_GF_POF.append(A_G_POF)
    Oral_GFTOGFDS_POF.append(O_G_POF)
    SNC_GFTOGFDS_POF.append(S_G_POF)

    COLOR_BLIND_GFVio.append(C_B_GFVio)
    ALG_GF_GFVio.append(A_G_GFVio)
    Oral_GFTOGFDS_GFVio.append(O_G_GFVio)
    SNC_GFTOGFDS_GFVio.append(S_G_GFVio)

    COLOR_BLIND_DSVio.append(C_B_DSVio)
    ALG_GF_DSVio.append(A_G_DSVio)
    Oral_GFTOGFDS_DSVio.append(O_G_DSVio)
    SNC_GFTOGFDS_DSVio.append(S_G_DSVio)

print("POF")
print(COLOR_BLIND_POF)
print(ALG_GF_POF)
print(Oral_GFTOGFDS_POF)
print(SNC_GFTOGFDS_POF)
print("GFVIO")
print(COLOR_BLIND_GFVio)
print(ALG_GF_GFVio)
print(Oral_GFTOGFDS_GFVio)
print(SNC_GFTOGFDS_GFVio)
print("DSVIO")
print(COLOR_BLIND_DSVio)
print(ALG_GF_DSVio)
print(Oral_GFTOGFDS_DSVio)
print(SNC_GFTOGFDS_DSVio)

x_axis_data = num_clusters
y_axis_COLOR_BLIND_POF = COLOR_BLIND_POF
y_axis_ALG_GF_POF = ALG_GF_POF
y_axis_Oral_GFTOGFDS_POF = Oral_GFTOGFDS_POF
y_axis_SNC_GFTOGFDS_POF = SNC_GFTOGFDS_POF

plt.plot(x_axis_data, y_axis_ALG_GF_POF, 'yo-', linewidth = 1, label = "ALG-GF-PoF")
plt.plot(x_axis_data, y_axis_Oral_GFTOGFDS_POF, 'g*-', linewidth = 1, label = "ALG-GFDS-PoF")
plt.plot(x_axis_data, y_axis_SNC_GFTOGFDS_POF, 'rv-', linewidth = 1, label = "G2GD-PoF")
plt.ylabel('Price of Fairness')

plt.xlabel('Number of Clusters')
plt.title(dataset_name)
plt.legend()
plt.show()

plt.plot(x_axis_data, COLOR_BLIND_GFVio, 'b^-', linewidth = 1, label = "COLOR-BLIND-GFViolation")
plt.plot(x_axis_data, ALG_GF_GFVio, 'yo-', linewidth = 1, label = "ALG-GF-GFViolation")
plt.plot(x_axis_data, Oral_GFTOGFDS_GFVio, 'g*-', linewidth = 1, label = "ALG-GFDS-GFViolation")
plt.plot(x_axis_data, SNC_GFTOGFDS_GFVio, 'rv-', linewidth = 1, label = "G2GD-GFViolation")
plt.ylabel('GF-Violation')

plt.xlabel('Number of Clusters')
plt.title(dataset_name)
plt.legend()
plt.show()

plt.plot(x_axis_data, COLOR_BLIND_DSVio, 'b^-', linewidth = 1, label = "COLOR-BLIND-DSViolation")
plt.plot(x_axis_data, ALG_GF_DSVio, 'yo-', linewidth = 1, label = "ALG-GF-DSViolation")
plt.plot(x_axis_data, Oral_GFTOGFDS_DSVio, 'g*-', linewidth = 1, label = "ALG-GFDS-DSViolation")
plt.plot(x_axis_data, SNC_GFTOGFDS_DSVio, 'rv-', linewidth = 1, label = "G2GD-DSViolation")
plt.ylabel('DS-Violation')


plt.xlabel('Number of Clusters')
plt.title(dataset_name)
plt.legend()
plt.show()