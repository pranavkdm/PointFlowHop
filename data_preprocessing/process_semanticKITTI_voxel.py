import os
import argparse
import re

import open3d as o3d
import numpy as np
from pyntcloud import PyntCloud
from multiprocessing import Pool
import time


# Some of the functions are taken from pykitti https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def load_velo_scan(file):
    """
    Load and parse a velodyne binary file
    """
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def load_poses(file):
    """
    Load and parse ground truth poses
    """
    tmp_poses = np.genfromtxt(file, delimiter=' ').reshape(-1, 3, 4)
    poses = np.repeat(np.expand_dims(np.eye(4), 0), tmp_poses.shape[0], axis=0)
    poses[:, 0:3, :] = tmp_poses
    return poses

def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary
    """
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

# This part of the code is taken from the semanticKITTI API
def open_label(filename):
    """ 
    Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
        raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(filename))))

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    return label

def set_label(label, points):
    """ 
    Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
        raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == points.shape[0]:
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16    # instance id in upper half
    else:
        print("Points shape: ", points.shape)
        print("Label shape: ", label.shape)
        raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((sem_label + (inst_label << 16) == label).all())

    return sem_label, inst_label

def transform_point_cloud(x1, R, t):
    """
    Transforms the point cloud using the giver transformation paramaters
    
    Args:
        x1  (np array): points of the point cloud [n,3]
        R   (np array): estimated rotation matrice [3,3]
        t   (np array): estimated translation vectors [3,1]
    Returns:
        x1_t (np array): points of the transformed point clouds [n,3]
    """
    x1_t = (np.matmul(R, x1.transpose()) + t).transpose()

    return x1_t

def sorted_alphanum(file_list_ordered):
    """
    Sorts the list alphanumerically
    Args:
        file_list_ordered (list): list of files to be sorted
    Return:
        sorted_list (list): input list sorted alphanumerically
    """
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    sorted_list = sorted(file_list_ordered, key=alphanum_key)

    return sorted_list

def get_file_list(path, extension=None):
    """
    Build a list of all the files in the provided path
    Args:
        path (str): path to the directory 
        extension (str): only return files with this extension
    Return:
        file_list (list): list of all the files (with the provided extension) sorted alphanumerically
    """
    if extension is None:
        file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        file_list = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)

    return file_list

def get_folder_list(path):
    """
    Build a list of all the folders in the provided path
    Args:
        path (str): path to the directory 
    Returns:
        folder_list (list): list of all the folders sorted alphanumerically
    """
    folder_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folder_list = sorted_alphanum(folder_list)
    
    return folder_list

def match_consecutive_point_cloud(pc_s, R1, t1, R2, t2):
    pc = np.matmul(np.linalg.inv(R2), (np.matmul(R1, pc_s.transpose()) + t1) - t2).transpose()
    return pc

def get_eigen_features(pc, n_neighbors=48):
    """
    Args: 
        pc (np array): points of the point cloud [n, 3]
        n_neighbors (int): number of neighbors selected
    Returns: 
        pc_feature (np array): features of the point cloud [n, 14]
    """
    
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

def cal_chamfer_dist(source_pts, target_pts, length):
    if(target_pts.size == 0):
        return 24 * length * length
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts[:, 0:3].reshape(-1, 3))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts[:, 0:3].reshape(-1, 3))
    
    source2target_dists = source_pcd.compute_point_cloud_distance(target_pcd)
    target2source_dists = target_pcd.compute_point_cloud_distance(source_pcd)
    source2target_dists = np.asarray(source2target_dists)
    target2source_dists = np.asarray(target2source_dists)
    source2target_dists = np.square(source2target_dists)
    target2source_dists = np.square(target2source_dists)
    chamfer_dist = source2target_dists.mean() + target2source_dists.mean()
    
    return chamfer_dist

def find_neighbors(voxel_center, source_pts, target_pts, length):
    source_idx = np.where((source_pts[:,0] > voxel_center[0] - length) & (source_pts[:,0] < voxel_center[0] + length) & 
                          (source_pts[:,1] > voxel_center[1] - length) & (source_pts[:,1] < voxel_center[1] + length) & 
                          (source_pts[:,2] > voxel_center[2] - length) & (source_pts[:,2] < voxel_center[2] + length))[0]
    target_idx = np.where((target_pts[:,0] > voxel_center[0] - length) & (target_pts[:,0] < voxel_center[0] + length) & 
                          (target_pts[:,1] > voxel_center[1] - length) & (target_pts[:,1] < voxel_center[1] + length) & 
                          (target_pts[:,2] > voxel_center[2] - length) & (target_pts[:,2] < voxel_center[2] + length))[0]
    
    source_pts = source_pts[source_idx, :]
    target_pts = target_pts[target_idx, :]
    
    return source_pts, target_pts, source_idx

def chamfer_dist(source_pts, target_pts, length=1.2):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(source_pts)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=length * 2)
    voxels_all = voxel_grid.get_voxels()
    
    chamfer = np.ones(source_pts.shape[0]) * -1
    
    for voxel in voxels_all:
        voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        source_neighbors_pts, target_neighbors_pts, source_neighbors_idx = find_neighbors(voxel_center, source_pts, target_pts, length)
        chamfer[source_neighbors_idx] = cal_chamfer_dist(source_neighbors_pts, target_neighbors_pts, length)
    
    return chamfer

class semanticKITTIProcesor:
    def __init__(self, raw_data_path, save_path, save_ply, save_near, n_processes):
        self.root_path = raw_data_path
        self.save_path = save_path
        self.save_ply = save_ply
        self.save_near = save_near
        self.n_processes = n_processes

        # self.scenes = get_folder_list(self.root_path)
        self.scenes = ["\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\00",
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\01", 
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\02", 
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\03",
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\04",
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\05",
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\06",
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\07",
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\08",
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\09",
                       "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\10"]

    def run_processing(self):
        if self.n_processes < 1:
            self.n_processes = 1

        pool = Pool(self.n_processes)
        pool.map(self.process_scene, self.scenes)
        pool.close()
        pool.join()

    def process_scene(self, scene):
        scene_name = scene.split(os.sep)[-1]

        # Create a save file if not existing
        if not os.path.exists(os.path.join(self.save_path, scene_name)):
            os.makedirs(os.path.join(self.save_path, scene_name))
        
        # Load transformation paramters
        poses = load_poses(os.path.join(scene, 'poses.txt'))
        tr_velo_cam = read_calib_file(os.path.join(scene, 'calib.txt'))['Tr'].reshape(3, 4)
        tr_velo_cam = np.concatenate((tr_velo_cam, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
        frames = get_file_list(os.path.join(scene, 'velodyne'), extension='.bin')

        if os.path.isdir(os.path.join(scene,'labels')):
            labels = get_file_list(os.path.join(scene,'labels'), extension='.label')
            test_scene = False
                    
            assert len(frames) == len(labels), "Number of point cloud fils and label files is not the same!"
        
        else:
            test_scene = True
        
        # Operate on each time frame 
        for idx in range(len(frames) - 1):
            # print(idx)
            frame_name_s = frames[idx].split(os.sep)[-1].split('.')[0]
            frame_name_t = frames[idx + 1].split(os.sep)[-1].split('.')[0]

            pc_s = load_velo_scan(frames[idx])[:, :3]
            pc_t = load_velo_scan(frames[idx + 1])[:, :3]

            # Transform both point cloud to the camera coordinate system (check KITTI webpage)
            pc_s = transform_point_cloud(pc_s, tr_velo_cam[:3, :3], tr_velo_cam[:3, 3:4])
            pc_t = transform_point_cloud(pc_t, tr_velo_cam[:3, :3], tr_velo_cam[:3, 3:4])
            
            # Transform the source point cloud from original coordinates to the target point cloud's coordinates
            R_s = poses[idx, 0:3, 0:3].reshape(3, 3)
            t_s = poses[idx, 0:3, 3].reshape(3, 1)
            R_t = poses[idx + 1, 0:3, 0:3].reshape(3, 3)
            t_t = poses[idx + 1, 0:3, 3].reshape(3, 1)
            
            pc_s = match_consecutive_point_cloud(pc_s, R_s, t_s, R_t, t_t)
            
            # Rotate 180 degrees around z axis (to be in accordance to KITTI flow as used by other datsets)
            pc_s[:, 0], pc_s[:, 1] = -pc_s[:, 0], -pc_s[:, 1]
            pc_t[:, 0], pc_t[:, 1] = -pc_t[:, 0], -pc_t[:, 1]
                
            # Extract eigen features
            eigen_features_s = get_eigen_features(pc_s)
            eigen_features_t = get_eigen_features(pc_t)

            if not test_scene:
                # Load the labels
                sem_label_s, inst_label_s = set_label(open_label(labels[idx]), pc_s)
                sem_label_t, inst_label_t = set_label(open_label(labels[idx + 1]), pc_t)
                
                # Remove ground points by naively thresholding the vertical coordinate
                # ground_mask_s = pc_s[:, 1] > -1.4
                # ground_mask_t = pc_t[:, 1] > -1.4
                # pc_s = pc_s[ground_mask_s, :]
                # pc_t = pc_t[ground_mask_t, :]
                
                # sem_label_s = sem_label_s[ground_mask_s]
                # inst_label_s = inst_label_s[ground_mask_s]

                # sem_label_t = sem_label_t[ground_mask_t]
                # inst_label_t = inst_label_t[ground_mask_t]
                
                # eigen_features_s = eigen_features_s[ground_mask_s]
                # eigen_features_t = eigen_features_t[ground_mask_t]
                
                class_mask_s = np.where((sem_label_s != 0) & (sem_label_s != 1) & (sem_label_s != 40) & (sem_label_s != 44) & (sem_label_s != 48) & (sem_label_s != 49) & (sem_label_s != 60) & (sem_label_s != 72))[0]
                class_mask_t = np.where((sem_label_t != 0) & (sem_label_t != 1) & (sem_label_t != 40) & (sem_label_t != 44) & (sem_label_t != 48) & (sem_label_t != 49) & (sem_label_t != 60) & (sem_label_t != 72))[0]
                pc_s = pc_s[class_mask_s, :]
                pc_t = pc_t[class_mask_t, :]

                sem_label_s = sem_label_s[class_mask_s]
                inst_label_s = inst_label_s[class_mask_s]

                sem_label_t = sem_label_t[class_mask_t]
                inst_label_t = inst_label_t[class_mask_t]
                
                eigen_features_s = eigen_features_s[class_mask_s]
                eigen_features_t = eigen_features_t[class_mask_t]
                
                # Remove points which are behind the car (to be in accordance with the stereo datasets)
                front_mask_s = pc_s[:, 2] > 1.5
                front_mask_t = pc_t[:, 2] > 1.5
                pc_s = pc_s[front_mask_s, :]
                pc_t = pc_t[front_mask_t, :]

                sem_label_s = sem_label_s[front_mask_s]
                inst_label_s = inst_label_s[front_mask_s]

                sem_label_t = sem_label_t[front_mask_t]
                inst_label_t = inst_label_t[front_mask_t]
                
                eigen_features_s = eigen_features_s[front_mask_s]
                eigen_features_t = eigen_features_t[front_mask_t]

                # Remove points whose depth is more than 35m to prevent depth values from explosion
                if self.save_near:
                    near_mask_s = pc_s[:, 2] < 35
                    near_mask_t = pc_t[:, 2] < 35
                    pc_s = pc_s[near_mask_s, :]
                    pc_t = pc_t[near_mask_t, :] 

                    sem_label_s = sem_label_s[near_mask_s]
                    inst_label_s = inst_label_s[near_mask_s]

                    sem_label_t = sem_label_t[near_mask_t]
                    inst_label_t = inst_label_t[near_mask_t]
                    
                    eigen_features_s = eigen_features_s[near_mask_s]
                    eigen_features_t = eigen_features_t[near_mask_t]
                
                # Extract static objects
                # Dynamic labels are 1 if moving and 0 if static
                static_idx_s = np.where(sem_label_s < 100)[0]
                static_idx_t = np.where(sem_label_t < 100)[0]
                dynamic_label_s = np.ones_like(sem_label_s)
                dynamic_label_s[static_idx_s] = 0

                dynamic_label_t = np.ones_like(sem_label_t)
                dynamic_label_t[static_idx_t] = 0
                
                # Calculate Chamfer distance
                dist = np.array(chamfer_dist(pc_s, pc_t))

                np.savez_compressed(os.path.join(self.save_path, scene_name, '{}_{}.npz'.format(frame_name_s, frame_name_t)),
                                    pc1=pc_s,
                                    pc2=pc_t,
                                    eigen_features_s = eigen_features_s,
                                    eigen_features_t = eigen_features_t,
                                    sem_label_s=sem_label_s,
                                    inst_label_s=inst_label_s,
                                    dynamic_label_s=dynamic_label_s,
                                    chamfer_dist=dist)
            else:
                # Remove ground points by naively thresholding the vertical coordinate
                ground_mask_s = pc_s[:, 1] > -1.4
                ground_mask_t = pc_t[:, 1] > -1.4
                pc_s = pc_s[ground_mask_s, :]
                pc_t = pc_t[ground_mask_t, :]
                
                eigen_features_s = eigen_features_s[ground_mask_s]
                eigen_features_t = eigen_features_t[ground_mask_t]
                
                # Remove points which are behind the car (to be in accordance with the stereo datasets)
                front_mask_s = pc_s[:, 2] > 1.5
                front_mask_t = pc_t[:, 2] > 1.5
                pc_s = pc_s[front_mask_s, :]
                pc_t = pc_t[front_mask_t,:]
                
                eigen_features_s = eigen_features_s[front_mask_s]
                eigen_features_t = eigen_features_t[front_mask_t]

                # Remove points whose depth is more than 30m to prevent depth values from explosion
                if self.save_near:
                    near_mask_s = pc_s[:, 2] < 35
                    near_mask_t = pc_t[:, 2] < 35
                    pc_s = pc_s[near_mask_s, :]
                    pc_t = pc_t[near_mask_t, :]
                    
                    eigen_features_s = eigen_features_s[near_mask_s]
                    eigen_features_t = eigen_features_t[near_mask_t]
                    
                # Calculate Chamfer distance
                dist = np.array(chamfer_dist(pc_s, pc_t))

                np.savez_compressed(os.path.join(self.save_path, scene_name, '{}_{}.npz'.format(frame_name_s, frame_name_t)),
                                    pc1=pc_s,
                                    pc2=pc_t,
                                    eigen_features_s = eigen_features_s,
                                    eigen_features_t = eigen_features_t,
                                    chamfer_dist=dist)
            
            # Save point clouds as ply files
            if self.save_ply:
                pcd_s = o3d.geometry.PointCloud()
                pcd_t = o3d.geometry.PointCloud()
                pcd_s.points = o3d.utility.Vector3dVector(pc_s)
                pcd_t.points = o3d.utility.Vector3dVector(pc_t)

                o3d.io.write_point_cloud(os.path.join(self.save_path, scene_name, '{}.ply'.format(frame_name_s)), pcd_s)
                o3d.io.write_point_cloud(os.path.join(self.save_path, scene_name, '{}.ply'.format(frame_name_t)), pcd_t)

if __name__ == '__main__': 
    processor = semanticKITTIProcesor(raw_data_path = "\\Workspace\\Odometry\\dataset\\KITTI\\sequences\\00",
                                      save_path = "\\Workspace\\SceneFlow\\datasets\\training_dataset",
                                      save_ply = False,
                                      save_near = True,
                                      n_processes = 16)

    processor.run_processing()