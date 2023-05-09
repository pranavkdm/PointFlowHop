import pickle
import threading
import numpy as np
import open3d as o3d
import sklearn
from ai import cs
from pyntcloud import PyntCloud
import rpointhop
import utils

def log_matrix(frame,T_total, LOG_FOUT):
    # print("Here")
    # LOG_FOUT.write(str(frame+1)+' ')
    LOG_FOUT.write(str(T_total[0,0])+' ')
    # print(str(T_total[0,0])+' '+str(T_total[0,1])+' '+str(T_total[0,2])+' '+str(T_total[0,3])+' '+str(T_total[1,0])+' '+str(T_total[1,1])+' '+str(T_total[1,2])+' '+str(T_total[1,3])+' '+str(T_total[2,0])+' '+str(T_total[2,1])+' '+str(T_total[2,2])+' '+str(T_total[2,3]))
    LOG_FOUT.write(str(T_total[0,1])+' ')
    LOG_FOUT.write(str(T_total[0,2])+' ')
    LOG_FOUT.write(str(T_total[0,3])+' ')
    LOG_FOUT.write(str(T_total[1,0])+' ')
    LOG_FOUT.write(str(T_total[1,1])+' ')
    LOG_FOUT.write(str(T_total[1,2])+' ')
    LOG_FOUT.write(str(T_total[1,3])+' ')
    LOG_FOUT.write(str(T_total[2,0])+' ')
    LOG_FOUT.write(str(T_total[2,1])+' ')
    LOG_FOUT.write(str(T_total[2,2])+' ')
    LOG_FOUT.write(str(T_total[2,3]))
    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

def calc_feature(pc_temp, pc_bin, pc_gather):
    value = np.multiply(pc_temp, pc_bin)
    value = np.sum(value, axis=2, keepdims=True)
    num = np.sum(pc_bin, axis=2, keepdims=True)
    final = np.squeeze(value/num, axis=(2,))
    pc_gather.append(final)

def get_feature(pc,n_neighbors=48):
    '''
    input : pc - point cloud (N x 3)
    return : pc_feature - point cloud feature (N x 14)
    '''
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)  
    pointcloud = PyntCloud.from_instance("open3d", pcd)

    neighbors = pointcloud.get_neighbors(k=n_neighbors)
    eigenvalues = pointcloud.add_scalar_field("eigen_values", k_neighbors=neighbors)
	
    anisotropy = pointcloud.add_scalar_field("anisotropy", ev=eigenvalues)	
    curvature = pointcloud.add_scalar_field("curvature", ev=eigenvalues)
    eigenentropy = pointcloud.add_scalar_field("eigenentropy", ev=eigenvalues)
    eigensum = pointcloud.add_scalar_field("eigen_sum", ev=eigenvalues)
    linearity = pointcloud.add_scalar_field("linearity", ev=eigenvalues)
    omnivariance = pointcloud.add_scalar_field("omnivariance", ev=eigenvalues)
    planarity = pointcloud.add_scalar_field("planarity", ev=eigenvalues)
    sphericity = pointcloud.add_scalar_field("sphericity", ev=eigenvalues)

    pc_feature = np.asarray(pointcloud.points)

    return pc_feature

def read_feature(pc):

    # modify path to dataset!!
    # path = '/Workspace/Odometry/dataset/KITTI/sequences/10/velodyne/'+str(nos).zfill(6)+'.bin'
    # path = '/mnt/pranav2/dataset/sequences/04/velodyne/'+str(nos).zfill(6)+'.bin'
    # pcl = np.fromfile(str(path), dtype=np.float32, count=-1).reshape([-1,4])
    # pc = pcl[:,:3]

    data = get_feature(pc)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:,:3])  
    pointcloud = PyntCloud.from_instance("open3d", pcd)
    neighbors = pointcloud.get_neighbors(k=48)

    ########### GE sampling ###############
    pc = data[data[:,2]>-1.2]
    neighbors = neighbors[tuple([data[:,2]>-1.2])]
    pcr = pc[pc[:,10]>0.6]
    neighbors = neighbors[pc[:,10]>0.6]
    pcr2 = pcr[pcr[:,8]<0.6]
    neighbors = neighbors[pcr[:,8]<0.6]
    rng = np.random.default_rng()
    # np.random.seed(1)
    index = rng.choice(pcr2.shape[0],size=2048,replace=True)
    pcr2 = pcr2[index,:]
    neighbors = neighbors[index,:]

    ############## random sampling ############
    # rng = np.random.default_rng()
    # np.random.seed(1)
    # index = rng.choice(data.shape[0],size=2048,replace=False)
    # pcr2 = data[index,:]
    # neighbors = neighbors[index,:]

    pts_fea_expand = utils.index_points(np.expand_dims(data, axis=0), np.expand_dims(neighbors, axis=0))
    # pts_fea_expand = pts_fea_expand.transpose(0, 2, 1, 3)  # (B, K, n_sample, dim)
    pts = pts_fea_expand[...,:3]
    eig = pts_fea_expand[...,6:]
    return np.expand_dims(pcr2[:,:3], axis=0), np.expand_dims(pts[0], axis=0), np.expand_dims(eig[0], axis=0)

def attribute(pc):

    data, pc_n, pc_temp = read_feature(pc) 
    pc_n_center = np.expand_dims(pc_n[:, :, 0, :], axis=2)
    pc_n_uncentered = pc_n - pc_n_center
    pc_temp = np.concatenate((pc_temp,pc_n_uncentered),axis=-1)

    pc_idx = []
    pc_idx.append(pc_n_uncentered[:, :, :, 0] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 0] <= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 1] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 1] <= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 2] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 2] <= 0)

    pc_bin = []
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[5])*1.0, axis=3))

    pc_gather1 = []
    pc_gather2 = []
    pc_gather3 = []
    pc_gather4 = []
    pc_gather5 = []
    pc_gather6 = []
    pc_gather7 = []
    pc_gather8 = []
    threads = []
    t1 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[0], pc_gather1))
    threads.append(t1)
    t2 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[1], pc_gather2))
    threads.append(t2)
    t3 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[2], pc_gather3))
    threads.append(t3)
    t4 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[3], pc_gather4))
    threads.append(t4)
    t5 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[4], pc_gather5))
    threads.append(t5)
    t6 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[5], pc_gather6))
    threads.append(t6)
    t7 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[6], pc_gather7))
    threads.append(t7)
    t8 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[7], pc_gather8))
    threads.append(t8)
    for t in threads:
        t.setDaemon(False)
        t.start()
    for t in threads:
        # if t.isAlive():
        t.join()
    pc_gather = pc_gather1 + pc_gather2 + pc_gather3 + pc_gather4 + pc_gather5 + pc_gather6 + pc_gather7 + pc_gather8

    pc_fea = np.concatenate(pc_gather, axis=2)

    return pc_fea, data 

def partition(data, feature, theta, phi):

    theta = theta*180/np.pi
    phi = phi*180/np.pi

    delta = 45
    q_0 = data[np.where(np.logical_and(phi>0+delta, phi<=90+delta))]
    f_0 = feature[np.where(np.logical_and(phi>0+delta, phi<=90+delta))]
    theta_0 = theta[np.where(np.logical_and(phi>0+delta, phi<=90+delta))]

    q_1 = data[np.where(np.logical_or(phi>90+delta, phi<=-180+delta))]
    theta_1 = theta[np.where(np.logical_or(phi>90+delta, phi<=-180+delta))]
    f_1 = feature[np.where(np.logical_or(phi>90+delta, phi<=-180+delta))]

    q_2 = data[np.where(np.logical_and(phi<=0+delta, phi>-90+delta))]
    theta_2 = theta[np.where(np.logical_and(phi<=0+delta, phi>-90+delta))]
    f_2 = feature[np.where(np.logical_and(phi<=0+delta, phi>-90+delta))]

    q_3 = data[np.where(np.logical_and(phi<=-90+delta, phi>=-180+delta))]
    theta_3 = theta[np.where(np.logical_and(phi<=-90+delta, phi>=-180+delta))]
    f_3 = feature[np.where(np.logical_and(phi<=-90+delta, phi>=-180+delta))]

    q_00 = q_0[theta_0>=0]
    q_01 = q_0[theta_0<0]
    q_10 = q_1[theta_1>=0]
    q_11 = q_1[theta_1<0]
    q_20 = q_2[theta_2>=0]
    q_21 = q_2[theta_2<0]
    q_30 = q_3[theta_3>=0]
    q_31 = q_3[theta_3<0]

    f_00 = f_0[theta_0>=0]
    f_01 = f_0[theta_0<0]
    f_10 = f_1[theta_1>=0]
    f_11 = f_1[theta_1<0]
    f_20 = f_2[theta_2>=0]
    f_21 = f_2[theta_2<0]
    f_30 = f_3[theta_3>=0]
    f_31 = f_3[theta_3<0]

    data_p = [q_00, q_01, q_10, q_11, q_20, q_21, q_30, q_31]
    feature_p = [f_00, f_01, f_10, f_11, f_20, f_21, f_30, f_31]

    return data_p, feature_p

