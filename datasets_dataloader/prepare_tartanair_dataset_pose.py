import os
import numpy as np
import sys
from glob import glob

#import tqdm
from os import path as osp
from scipy.spatial.transform import Rotation

scenes = [
    "office",  
    "office2",   
    "abandonedfactory", 
    "abandonedfactory_night", 
    "amusement",   
    "carwelding",   
    "endofworld",   
    "gascola",   
    "hospital", 	
    "japanesealley",   
    "neighborhood",   
    "ocean",   
    "oldtown",   
    "seasidetown",   
    "seasonsforest",   
    "seasonsforest_winter", 	
    "soulcity",   
    "westerndesert", 
    ]
diff_levels = ['Easy', 'Hard']


"""
The camera pose file is a text file containing the translation and orientation 
of the camera in a fixed coordinate frame. 

- Each line in the text file contains a single pose.

- The number of lines/poses is the same as the number of 
  image frames in that trajectory.

- The format of each line is 'tx ty tz qx qy qz qw'.

- tx ty tz (3 floats) give the position of the optical center of the 
  color camera with respect to the world origin in the world frame.

- qx qy qz qw (4 floats) give the orientation of the optical center of 
  the color camera in the form of a unit quaternion with respect to 
  the world frame.

- The camera motion is defined in the NED frame. That is to say, 
  the x-axis is pointing to the camera's forward, the y-axis is 
  pointing to the camera's right, the z-axis is pointing to 
  the camera's downward.
"""

def pos_quat2SE_matrix(quat_data # [tx ty tz qx qy qz qw]
        ):
    SO = Rotation.from_quat(quat_data[3:7]).as_matrix()
    SE = np.eye(4)
    SE[0:3,0:3] = SO
    SE[0:3,3]   = quat_data[0:3]
    #print (SE)
    return SE

# 
def ned2cam(quat_data):
    '''
    transfer a ned traj to camera frame traj
    '''
    # wned: world coordinate in NED (x Forward, y Right, z Down) format;
    # cned: camera coordinate in NED (x Forward, y Right, z Down) format;
    # w: world coordinate in OpenCV style (x Right, y Down, z Forward);
    # c: camera coordinate in OpenCV style (x Right, y Down, z Forward);
    # To find T_wned_2_w is to project each axis of x^wned, y^wned, z^wned, 
    # into axis x^w, y^w, z^w,
    # i.e., P^w = T_{wned}^{w} * P^{wned}
    T = np.array([
                  [0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32)
    T_wned_2_w = T
    # Similarly, we can find the transformation from cned to c;
    T_cned_2_c = T
    T_c_2_cnet = np.linalg.inv(T_cned_2_c)
    T_cned_2_wned = pos_quat2SE_matrix(quat_data)
    #NOTE: We want to find the pose between c and w in OpenCV style coordinates;
    # That is to say to find the cam-to-world pose T^{w}_{c}, to map P^w = T^{w}_{c} * P^{c};
    # Using the chain-rule:
    # T^{w}_{c} = T^{w}_{wned} * T^{wned}_{cned} * T^{cned}_{c}
    T_cam_2_world = np.matmul(np.matmul(T_wned_2_w, T_cned_2_wned), T_c_2_cnet)
    return T_cam_2_world

if __name__ == '__main__':

    data_root = "Data1/Tartanair"
    dst_root = "Data2/TartanAir"
    for seq in scenes:
        for diff_lel in diff_levels:
            cur_dir = osp.join(data_root, seq, seq, diff_lel)
            p_paths = sorted(
            # one example: */TartanAir/office/Easy/P000/
            glob(osp.join(cur_dir, f"*/"))
            )
            for scan in p_paths:
                print ("scan = ", scan)
                for cam in ["left", "right"]:
                    pose_src_file = osp.join(scan, f'pose_{cam}.txt')
                    # e.g., scan = */Tartanair/abandonedfactory/abandonedfactory/Hard/P008
                    # to get "P008";
                    if scan.endswith("/"):
                        cur_P0X = scan[:-1].split("/")[-1] 
                    else:
                        cur_P0X = scan.split("/")[-1] 
                    print ("cur_POX = ", cur_P0X)
                    #pose_dir = osp.join(scan, f"pose_me_{cam}")
                    dst_pose_dir = osp.join(dst_root, seq, diff_lel, cur_P0X, f"pose_me_{cam}")
                    os.makedirs(dst_pose_dir, exist_ok=True)
                    #pose = np.loadtxt(pose_src_file).astype(np.float32).reshape(-1,7)
                    pose_quats = np.loadtxt(pose_src_file).astype(np.float32)
                    print ("??? pose_quats ", pose_quats.shape)
                    img_paths = glob(osp.join(scan, 'image_left/*_left.png'))
                    assert len(img_paths) == pose_quats.shape[0], "Requires #image == #pose"
                    print (f"read from {pose_src_file}, and save to {dst_pose_dir}")
                    for i in range(pose_quats.shape[0]):
                        T_cam2world_invE = ned2cam(pose_quats[i])
                        pose_txtfile = osp.join(dst_pose_dir, f"{i:06d}_left.txt")
                        np.savetxt(pose_txtfile, T_cam2world_invE)
                        #if i > 5:
                        #  sys.exit()