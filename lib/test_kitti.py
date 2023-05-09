import open3d as o3d
import pickle
import threading
import numpy as np
import sklearn
from ai import cs
from pyntcloud import PyntCloud
import rpointhop
import utils
import kitti_utils

# lidar_to_cam = np.asarray([-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03, -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02, 9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01])
# lidar_to_cam = lidar_to_cam.reshape(3,4)
# ones = np.zeros((1,4))
# ones[0,3] = 1
# lidar_to_cam = np.concatenate((lidar_to_cam, ones),axis=0)

lidar_to_cam = np.ones((4, 4))

LOG_FOUT = open('stereoKITTI.txt', 'a')

def get_transformation_ransac(data,features,frame):

    data_x = data[0]
    data_y = data[1]
    feature_x = features[0]
    feature_y = features[1]
    
    x_r, x_theta, x_phi = cs.cart2sp(data_x[:,0], data_x[:,1], data_x[:,2])
    y_r, y_theta, y_phi = cs.cart2sp(data_y[:,0], data_y[:,1], data_y[:,2])

    data_x_p, feature_x_p = kitti_utils.partition(data_x, feature_x, x_theta, x_phi)
    data_y_p, feature_y_p = kitti_utils.partition(data_y, feature_y, y_theta, y_phi)

    # distances = sklearn.metrics.pairwise.euclidean_distances(feature_y_p[0],feature_x_p[0])
    # pred = np.argmin(distances,axis=0)
    # data_x_c = data_x_p[0]
    # data_y_c = data_y_p[0][pred]
    data_x_c = np.array([])
    data_y_c = np.array([])

    for i in range(0, len(data_x_p)):
        if(feature_y_p[i].shape[0] == 0 or feature_x_p[i].shape[0] == 0):
            continue
        else:
            if data_x_c.shape[0] == 0:
                distances = sklearn.metrics.pairwise.euclidean_distances(feature_y_p[i],feature_x_p[i])
                pred = np.argmin(distances,axis=0)
                data_x_c = data_x_p[i]
                data_y_c = data_y_p[i][pred]
            else:
                distances = sklearn.metrics.pairwise.euclidean_distances(feature_y_p[i],feature_x_p[i])
                pred = np.argmin(distances,axis=0)
                data_x_c = np.concatenate((data_x_c,data_x_p[i]), axis=0)
                data_y_c = np.concatenate((data_y_c,data_y_p[i][pred]), axis=0)

    x_r, x_theta, x_phi = cs.cart2sp(data_x_c[:,0], data_x_c[:,1], data_x_c[:,2])
    y_r, y_theta, y_phi = cs.cart2sp(data_y_c[:,0], data_y_c[:,1], data_y_c[:,2])

    x_phi = x_phi * 180/np.pi
    y_phi = y_phi * 180/np.pi

    diff_phi = np.abs(x_phi-y_phi)
    diff_r = np.abs(x_r-y_r)
    data_x_c = data_x_c[np.logical_and(diff_phi<3.5,diff_r<4)]   #3.5  4
    data_y_c = data_y_c[np.logical_and(diff_phi<3.5,diff_r<4)]

    data_x_c_aug = np.concatenate((data_x_c, np.ones((data_x_c.shape[0],1))),axis=1)
    data_y_c_aug = np.concatenate((data_y_c, np.ones((data_y_c.shape[0],1))),axis=1)

    l_x = (lidar_to_cam @ data_x_c_aug.T).T
    l_y = (lidar_to_cam @ data_y_c_aug.T).T

    data_x_c = l_x[:,:3]
    data_y_c = l_y[:,:3]

    vec = np.expand_dims(np.arange(data_x_c.shape[0]), 1)
    vec = np.concatenate((vec,vec),axis=1)
    # vec = np.concatenate((ind_y,ind_x),axis=1)
    vect = o3d.utility.Vector2iVector(vec)

    data_x_c1 = o3d.geometry.PointCloud()
    data_x_c1.points = o3d.utility.Vector3dVector(data_x_c)

    data_y_c1 = o3d.geometry.PointCloud()
    data_y_c1.points = o3d.utility.Vector3dVector(data_y_c)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(data_y_c1, data_x_c1, vect, 0.1)
    # result = o3d.registration.registration_ransac_based_on_correspondence(data_y_c1, data_x_c1, vect, 0.1)

    Rt = result.transformation

    # x_mean = np.mean(data_x_c,axis=0,keepdims=True)
    # y_mean = np.mean(data_y_c,axis=0,keepdims=True)

    # data_x_c = data_x_c - x_mean
    # data_y_c = data_y_c - y_mean

    # cov = (data_x_c.T@data_y_c)
    # u, s, v = np.linalg.svd(cov)
    # R = v.T@u.T

    # if (np.linalg.det(R) < 0):
    #     u, s, v = np.linalg.svd(cov)
    #     reflect = np.eye(3)
    #     reflect[2,2] = -1
    #     v = v.T@reflect
    #     R = v@u.T

    # t = -R@x_mean.T+y_mean.T

    # Rt = np.concatenate((R.T, -t),axis=1)
    # Rt = np.concatenate((Rt, ones),axis=0)

    # T_total = T_total @ Rt

    kitti_utils.log_matrix(frame, Rt, LOG_FOUT)

    return Rt

def main():

    with open('pointhop.pkl', 'rb') as f:
        params = pickle.load(f, encoding='latin')
    # T_total = np.eye(4)
    # T_total = np.array([1.000000e+00, 1.197625e-11, 1.704638e-10, 1.665335e-16, 1.197625e-11, 1.000000e+00, 3.562503e-10, -1.110223e-16, 1.704638e-10, 3.562503e-10, 1.000000e+00, 2.220446e-16, 0, 0, 0, 1])
    # T_total = T_total.reshape(4,4)
    # kitti_utils.log_matrix(0, T_total, LOG_FOUT)

    for i in range(150):
        path = '\\Workspace\\SceneFlow\\datasets\\kitti_rm_ground\\'+str(i).zfill(6)+'.npz'
        pc_data = np.load(path)
        data_0 = pc_data['pos1']
        data_1 = pc_data['pos2']

        attribute_0, data_0 = kitti_utils.attribute(data_0)
        attribute_1, data_1 = kitti_utils.attribute(data_1)
        attribute_h = np.concatenate((attribute_0, attribute_1), axis=0)
        data = np.concatenate((data_0, data_1), axis=0)

        leaf_node = rpointhop.pointhop_pred(data, attribute_h, pca_params=params, n_sample=[36,8,16])
        features = np.moveaxis(np.squeeze(np.asarray(leaf_node)), 0, 2)
        Rt = get_transformation_ransac(data, features, i)
        print(i)

if __name__=='__main__':
    main()
