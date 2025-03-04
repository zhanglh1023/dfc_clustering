import configparser
import time
from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
from cplex_fair_assignment_lp_solver import fair_partial_assignment
from cplex_violating_clustering_lp_solver import violating_lp_clustering
from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key,
                                 vanilla_clustering, write_fairness_trial)
from util.configutil import read_list
from calculate_result import CalCost
from calculate_result import CalCenter
from calculate_result import CalDistance
from calculate_result import CalGFViolation, CalDSViolation
from doubly_fair_clustering import SNC, Divide, Oral_Selected
import math
# This function takes a dataset and performs a fair clustering on it.
# Arguments:
#   dataset (str) : dataset to use
#   config_file (str) : config file to use (will be read by ConfigParser)
#   data_dir (str) : path to write output
#   num_clusters (int) : number of clusters to use
#   deltas (list[float]) : delta to use to tune alpha, beta for each color
#   max_points (int ; default = 0) : if the number of points in the dataset 
#       exceeds this number, the dataset will be subsampled to this amount.
# Output:
#   None (Writes to file in `data_dir`)  
def fair_clustering(dataset, config_file, data_dir, num_clusters, deltas, max_points, violating, violation, shiyan):
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # Read data in from a given csv_file found in config
    # df (pd.DataFrame) : holds the data
    df = read_data(config, dataset)
    # Subsample data if needed
    if max_points and len(df) > max_points:
       df = subsample_data(df, max_points)

    #print(df)
    # Clean the data (bucketize text data)
    df, _ = clean_data(df, config, dataset)

    #print(df)
    #print(_)
    # variable_of_interest (list[str]) : variables that we would like to collect statistics for
    variable_of_interest = config[dataset].getlist("variable_of_interest")
    print(variable_of_interest)

    # Assign each data point to a color, based on config file
    # attributes (dict[str -> defaultdict[int -> list[int]]]) : holds indices of points for each color class
    # color_flag (dict[str -> list[int]]) : holds map from point to color class it belongs to (reverse of `attributes`)
    attributes, color_flag = {}, {}
    for variable in variable_of_interest:
        colors = defaultdict(list)
        this_color_flag = [0] * len(df)
        
        condition_str = variable + "_conditions"
        bucket_conditions = config[dataset].getlist(condition_str)

        # For each row, if the row passes the bucket condition, 
        # then the row is added to that color class
        for i, row in df.iterrows():
            #print(i)
            #print(row)
            for bucket_idx, bucket in enumerate(bucket_conditions):
                if eval(bucket)(row[variable]):
                    #print(bucket_idx)
                    colors[bucket_idx].append(i)
                    this_color_flag[i] = bucket_idx

        attributes[variable] = colors
        color_flag[variable] = this_color_flag

    fairness_variable = config[dataset].getlist("fairness_variable")
    print(attributes)
    print(color_flag)
    Color = color_flag[fairness_variable[0]]
    print(Color)
    # representation (dict[str -> dict[int -> float]]) : representation of each color compared to the whole dataset
    representation = {}
    for var, bucket_dict in attributes.items():
        representation[var] = {k : (len(bucket_dict[k]) / len(df)) for k in bucket_dict.keys()}

    # Select only the desired columns
    selected_columns = config[dataset].getlist("columns")
    df = df[[col for col in selected_columns]]

    # Scale data if desired
    scaling = config["DEFAULT"].getboolean("scaling")
    if scaling:
        df = scale_data(df)


    print(df)
    print("################")
    #return
    # Cluster the data -- using the objective specified by clustering_method
    clustering_method = config["DEFAULT"]["clustering_method"]
    vanilla_assignment = []
    if not violating:
        t1 = time.monotonic()
        #initial_score, pred, cluster_centers, vanilla_assignment, vanilla_centers = vanilla_clustering(df, num_clusters, clustering_method)
        #initial_score, pred, cluster_centers = vanilla_clustering(df, num_clusters, clustering_method)
        pred, cluster_centers = vanilla_clustering(df, num_clusters, clustering_method)
        vanilla_centers = cluster_centers
        t2 = time.monotonic()
        cluster_time = t2 - t1
        print("Clustering time: {}".format(cluster_time))

        #for label in vanilla_assignment :
            #print(label)
        #for center in vanilla_centers :
            #print(center)
        #return
        ### Calculate fairness statistics
        # fairness ( dict[str -> defaultdict[int-> defaultdict[int -> int]]] )
        # fairness : is used to hold how much of each color belongs to each cluster
        fairness = {}
        # For each point in the dataset, assign it to the cluster and color it belongs too
        for attr, colors in attributes.items():
            fairness[attr] = defaultdict(partial(defaultdict, int))
            for i, row in enumerate(df.iterrows()):
                cluster = pred[i]
                for color in colors:
                    if i in colors[color]:
                        fairness[attr][cluster][color] += 1
                        continue

        # sizes (list[int]) : sizes of clusters
        sizes = [0 for _ in range(num_clusters)]
        for p in pred:
            sizes[p] += 1

        # ratios (dict[str -> dict[int -> list[float]]]): Ratios for colors in a cluster
        ratios = {}
        for attr, colors in attributes.items():
            attr_ratio = {}
            for cluster in range(num_clusters):
                attr_ratio[cluster] = [fairness[attr][cluster][color] / sizes[cluster] 
                                for color in sorted(colors.keys())]
            ratios[attr] = attr_ratio
    else:
        # These added so that output format is consistent among violating and
        # non-violating trials
        fake_initial_score, fake_pred, fake_cluster_centers, vanilla_assignment, vanilla_centers = vanilla_clustering(df, num_clusters, clustering_method)
        cluster_time, initial_score = 0, 0
        fairness, ratios = {}, {}
        sizes, cluster_centers = [], []

    data = df.values
    print(data)
    for i in data :
        mn = float("inf")
        id = 0
        for j in range(len(vanilla_centers)) :
            if(CalDistance(i, vanilla_centers[j]) < mn) :
                mn = CalDistance(i, vanilla_centers[j])
                id = j
        vanilla_assignment.append(id)
    for label in vanilla_assignment:
        print(label)
    for center in vanilla_centers:
        print(center)
    vanilla_cost = CalCost(clustering_method, data, vanilla_centers, vanilla_assignment)
    #return
    # dataset_ratio : Ratios for colors in the dataset
    dataset_ratio = {}
    for attr, color_dict in attributes.items():
        dataset_ratio[attr] = {int(color) : len(points_in_color) / len(df) 
                            for color, points_in_color in color_dict.items()}

    # fairness_vars (list[str]) : Variables to perform fairness balancing on
    fairness_vars = config[dataset].getlist("fairness_variable")
    print(fairness_vars[0])
    for delta in deltas:
        #   alpha_i = a_val * (representation of color i in dataset)
        #   beta_i  = b_val * (representation of color i in dataset)
        alpha, beta = {}, {}
        a_val, b_val = (1 + delta), 1 - delta
        for var, bucket_dict in attributes.items():
            alpha[var] = {k : a_val * representation[var][k] for k in bucket_dict.keys()}
            beta[var] = {k : b_val * representation[var][k] for k in bucket_dict.keys()}

        # Only include the entries for the variables we want to perform fairness on
        # (in `fairness_vars`). The others are kept for statistics.
        fp_color_flag, fp_alpha, fp_beta = (take_by_key(color_flag, fairness_vars),
                                            take_by_key(alpha, fairness_vars),
                                            take_by_key(beta, fairness_vars))
        print("################")
        print(fp_alpha)
        print(fp_beta)

        print(fp_color_flag)
        # Solves partial assignment and then performs rounding to get integral assignment
        if not violating:
            t1 = time.monotonic()
            res = fair_partial_assignment(df, cluster_centers, fp_alpha, fp_beta, fp_color_flag, clustering_method)
            t2 = time.monotonic()
            lp_time = t2 - t1

        else:
            t1 = time.monotonic()
            res = violating_lp_clustering(df, num_clusters, fp_alpha, fp_beta, fp_color_flag, clustering_method, violation)
            t2 = time.monotonic()
            lp_time = t2 - t1

            # Added so that output formatting is consistent among violating
            # and non-violating trials
            res["partial_objective"] = 0
            res["partial_assignment"] = []

        ### Output / Writing data to a file
        # output is a dictionary which will hold the data to be written to the
        #   outfile as key-value pairs. Outfile will be written in JSON format.
        shiyan[0] += 1
        output = {}

        # num_clusters for re-running trial
        output["num_clusters"] = num_clusters

        # Whether or not the LP found a solution
        output["success"] = res["success"]

        # Nonzero status -> error occurred
        output["status"] = res["status"]
        
        output["dataset_distribution"] = dataset_ratio

        # Save alphas and betas from trials
        output["alpha"] = alpha
        output["beta"] = beta

        # Save original clustering score
        #output["unfair_score"] = initial_score

        # Clustering score after addition of fairness
        output["fair_score"] = res["objective"]
        
        # Clustering score after initial LP
        output["partial_fair_score"] = res["partial_objective"]

        # Save size of each cluster
        output["sizes"] = sizes
       # print("sizes:")
       # print(sizes)
        output["attributes"] = attributes

        # Save the ratio of each color in its cluster
        output["ratios"] = ratios

        # These included at end because their data is large
        # Save points, colors for re-running trial
        # Partial assignments -- list bc. ndarray not serializable
        output["centers"] = [list(center) for center in cluster_centers]
        print("centers:")
        for center in cluster_centers:
            print(center)
        output["points"] = [list(point) for point in df.values]
       # print("points:")
        cnt = 0
        for point in df.values :
            cnt+=1
        #print(cnt)
        output["assignment"] = res["assignment"]
        gf_assignment = []
        for i in range(len(data)) :
            for j in range(num_clusters) :
                if res["assignment"][i * num_clusters + j] :
                    gf_assignment.append(j)
                    break
        #print(res["assignment"])
        #print(gf_assignment)
        #k-means:center
        #gf_centers = CalCenter(data, gf_assignment, num_clusters)
        gf_centers = cluster_centers
        #print(gf_centers)
        gf_cost = CalCost(clustering_method, data, gf_centers, gf_assignment)
        print(gf_cost / vanilla_cost)
        output["partial_assignment"] = res["partial_assignment"]
        output["name"] = dataset
        output["clustering_method"] = clustering_method
        output["scaling"] = scaling
        output["delta"] = delta
        output["time"] = lp_time
        output["cluster_time"] = cluster_time
        output["violating"] = violating
        output["violation"] = violation
        print(df)
        #print(data)
        print(gf_centers)
        print(gf_assignment)
        gf_fair = []
        for i in range(len(data)) :
            gf_fair.append(color_flag[fairness_variable[0]][i])
        #print(_)
        print(attributes)
        print(color_flag)
        points_num = []
        for i in range(len(gf_centers)) :
            points_num.append(0)
        for i in range(len(gf_assignment)) :
            points_num[gf_assignment[i]] += 1
        kl = []
        kr = []
        f = []
        fair_num = 0
        for i in range(len(data)) :
            f.append(0)
        for i in range(len(gf_fair)) :
            if(f[gf_fair[i]]) : continue
            fair_num += 1
            f[gf_fair[i]] += 1
        rh = [] #不同颜色数据在data中的占比
        for i in range(fair_num) :
            rh.append(0)
        for i in range(len(gf_fair)) :
            rh[gf_fair[i]] += 1
        for i in range(fair_num) :
            rh[i] /= len(data)
        for i in range(fair_num) :
            kl.append(math.floor(0.8 * rh[i] * len(gf_centers)))
            kr.append(math.ceil(rh[i] * len(gf_centers)))
        Q, selected_points, error = SNC(data, gf_centers, gf_assignment, gf_fair, kl, kr, points_num)
        for i in range(len(gf_centers)) :
            for j in range(len(Q[i])) :
                print(i, Q[i][j])
        OQ, O_selected_points, O_error = Oral_Selected(data, gf_centers, gf_assignment, gf_fair, kl, kr, points_num)
        data_distribution = []
        for i in range(len(gf_centers)):
            data_distribution.append([])
        for i in range(len(gf_centers)):
            for j in range(fair_num):
                data_distribution[i].append([])
        for i in range(len(data)):
            center_id = gf_assignment[i]
            fair_id = gf_fair[i]
            data_distribution[center_id][fair_id].append(i)

        print("是否存在簇没选出中心：")
        print(error, O_error)
        double_fair_centers, double_fair_assignment = Divide(data, gf_centers, Q, fair_num, data_distribution)
        Oral_centers, Oral_assignment = Divide(data, gf_centers, OQ, fair_num, data_distribution)
        '''double_fair_centers = []
        double_fair_centers_num = 0
        double_fair_assignment = []
        for i in range(len(data)):
            double_fair_assignment.append(0)
        for i in range(len(gf_centers)) :
            #print(i)
            id_array = []
            for j in range(len(Q[i])) :
                #print(Q[i][j])
                double_fair_centers.append(Q[i][j])
                id_array.append(double_fair_centers_num)
                double_fair_centers_num += 1
            id = 0
            for j in range(fair_num):
                if j % 2 == 0:
                    for k in range(len(data_distribution[i][j])):
                        double_fair_assignment[data_distribution[i][j][k]] = id_array[id]
                        id += 1
                        id %= len(id_array)
                else :
                    for k in range(len(data_distribution[i][j]) - 1, -1, -1):
                        double_fair_assignment[data_distribution[i][j][k]] = id_array[id]
                        id += 1
                        id %= len(id_array)'''
        #for i in range(len(double_fair_centers)) :
         #   print(double_fair_centers)
        #for i in range(len(data)):
            #print(data[i], double_fair_assignment[i])
        double_fair_cost = CalCost(clustering_method, data, double_fair_centers, double_fair_assignment)
        Oral_cost = CalCost(clustering_method, data, Oral_centers, Oral_assignment)
        mx_cluster_size = 0
        double_fair_cluster_size = []
        for i in range(len(double_fair_centers)):
            double_fair_cluster_size.append(0)
        for i in range(len(data)):
            double_fair_cluster_size[double_fair_assignment[i]] += 1
        for i in range(len(double_fair_centers)):
            mx_cluster_size = max(mx_cluster_size, double_fair_cluster_size[i])
        print(double_fair_cost, gf_cost, Oral_cost)
        approximate_ratio = double_fair_cost / gf_cost
        print(double_fair_cost / gf_cost, Oral_cost / gf_cost)
        print(double_fair_cost / vanilla_cost, Oral_cost / vanilla_cost)
        print(mx_cluster_size)
        print(approximate_ratio - mx_cluster_size - 1)
        print(double_fair_centers)
        print(gf_centers)

        COLOR_BLIND_POF = vanilla_cost / vanilla_cost
        ALG_GF_POF = gf_cost / vanilla_cost
        Oral_GFTOGFDS_POF = Oral_cost / vanilla_cost
        SNC_GFTOGFDS_POF = double_fair_cost / vanilla_cost
        beta = []
        alpha = []
        for i in range(len(kl)):
            beta.append(fp_beta[fairness_variable[0]][i])
        for i in range(len(kl)):
            alpha.append(fp_alpha[fairness_variable[0]][i])
        #print(beta)
        #print(alpha)

        #print(data)
        #print(vanilla_centers)
        #print(vanilla_assignment)
        #print(gf_centers)
        #print(gf_assignment)
        #print(Color)
        COLOR_BLIND_GFVio = CalGFViolation(data, vanilla_centers, vanilla_assignment, Color, beta, alpha)
        ALG_GF_GFVio = CalGFViolation(data, gf_centers, gf_assignment, Color, beta, alpha)
        Oral_GFTOGFDS_GFVio = CalGFViolation(data, Oral_centers, Oral_assignment, Color, beta, alpha)
        SNC_GFTOGFDS_GFVio = CalGFViolation(data, double_fair_centers, double_fair_assignment , Color, beta, alpha)
        #print(CalGFViolation(data, vanilla_centers, vanilla_assignment, Color, beta, alpha))
        #print(CalGFViolation(data, gf_centers, gf_assignment, Color, beta, alpha))
        #print(CalGFViolation(data, Oral_centers, Oral_assignment, Color, beta, alpha))
        #print(CalGFViolation(data, double_fair_centers, double_fair_assignment , Color, beta, alpha))
        #print(kl)
        #print(kr)
        #print(CalDSViolation(data, vanilla_centers, vanilla_assignment, Color, kl, kr))
        #print(CalDSViolation(data, gf_centers, gf_assignment, Color, kl, kr))
        #print(CalDSViolation(data, Oral_centers, Oral_assignment, Color, kl, kr))
        #print(CalDSViolation(data, double_fair_centers, double_fair_assignment, Color, kl, kr))
        COLOR_BLIND_DSVio = CalDSViolation(data, vanilla_centers, vanilla_assignment, Color, kl, kr)
        ALG_GF_DSVio = CalDSViolation(data, gf_centers, gf_assignment, Color, kl, kr)
        Oral_GFTOGFDS_DSVio = CalDSViolation(data, Oral_centers, Oral_assignment, Color, kl, kr)
        SNC_GFTOGFDS_DSVio = CalDSViolation(data, double_fair_centers, double_fair_assignment, Color, kl, kr)
        return COLOR_BLIND_POF, ALG_GF_POF, Oral_GFTOGFDS_POF, SNC_GFTOGFDS_POF, COLOR_BLIND_GFVio, ALG_GF_GFVio, Oral_GFTOGFDS_GFVio, SNC_GFTOGFDS_GFVio, COLOR_BLIND_DSVio, ALG_GF_DSVio, Oral_GFTOGFDS_DSVio, SNC_GFTOGFDS_DSVio
        # Writes the data in `output` to a file in data_dir

        #write_fairness_trial(output, data_dir, "", shiyan)

        # Added because sometimes the LP for the next iteration solves so 
        # fast that `write_fairness_trial` cannot write to disk
        time.sleep(1) 
