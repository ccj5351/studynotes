import torch
import tqdm
import numpy as np
from pathlib import Path
import multiprocessing as mp

from tools import pfmutil as pfm 
from models.hit_net.planefit_op.build.lib import HitnetModule
from os.path import join as pjoin

def np2torch(x, t=True, bgr=False):
    if len(x.shape) == 2:
        x = x[..., None]
    if bgr:
        x = x[..., [2, 1, 0]]
    if t:
        x = np.transpose(x, (2, 0, 1))
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255
    x = torch.from_numpy(x.copy())
    return x

def process(file_path):
    while True:
        for ids, lock in enumerate(process.lock_list): 
            if lock.acquire(block=False):
                disparity_path, data_type = file_path.split(' ')[2:4]
                data_root = (process.root)[data_type]
                # To replace old suffix (e.g., ".png") in the path with the new ".pfm";
                #pfm_path = (process.root / "disparity" / file_path).with_suffix(".pfm")
                #dxy_path = (process.root / "our_slant_window" / file_path).with_suffix(".npy")
                #dxy_path.parent.mkdir(exist_ok=True, parents=True)
                
                pfm_path = pjoin(data_root, disparity_path)
                dxy_path = Path(pfm_path.replace('disparity', 'our_slant_window')).with_suffix(".npy")
                #print (f"pfm={pfm_path}, dxy={dxy_path}")
                dxy_path.parent.mkdir(exist_ok=True, parents=True)
                with torch.no_grad():
                    x = np2torch(pfm.readPFM(pfm_path)).unsqueeze(0).cuda(ids)
                    x = HitnetModule.plane_fitting(
                        input=x, #[1,1,H,W]
                        iter=256, 
                        sigma=0.1, 
                        kernel_size=9, 
                        min_disp=0, 
                        max_disp=1e5) #[B,2,H,W]
                    x = x[0].cpu().numpy()#[2,H,W]
                    #print ("x shape = ", x.shape, type(x[0,0,0])) # (2, H, W), float32
                    #e.g., == . x shape =  (2, 540, 960) <class 'numpy.float32'>;

                np.save(dxy_path, x)
                #dx_path = pfm_path.replace('disparity', 'our_slant_window')[:-len('.pfm')]+ "_dx.pfm"
                #dy_path = pfm_path.replace('disparity', 'our_slant_window')[:-len('.pfm')]+ "_dy.pfm"
                #pfm.save(dx_path, x[0].astype(np.float32))
                #pfm.save(dy_path, x[1].astype(np.float32))
                lock.release()
                return


def process_init(lock_list, root):
    process.lock_list = lock_list
    process.root = root


def main(root, list_path):
    #root = Path(root)
    root = root

    with open(list_path, "rt") as fp:
        # oneline: left_img_path right_img_path left_disparity_path dataset_name
        # E.g.,: 
        # sceneflow/driving/frames_finalpass/15mm_focallength/scene_backwards/slow/left/0224.png sceneflow/driving/frames_finalpass/15mm_focallength/scene_backwards/slow/right/0224.png sceneflow/driving/disparity/15mm_focallength/scene_backwards/slow/left/0224.pfm driving
        # frames_finalpass/TRAIN/B/0144/left/0010.png frames_finalpass/TRAIN/B/0144/right/0010.png disparity/TRAIN/B/0144/left/0010.pfm flyingthings3d
        #file_list = [Path(line.strip()) for line in fp]
        file_list = [line.strip() for line in fp]
    gpu_num = 8
    lock_list = [mp.Lock() for _ in range(gpu_num)]
    with mp.Pool(8, process_init, [lock_list, root]) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process, file_list), total=len(file_list)))


"""
How to run this file:
- cd ~/mobile-stereo-proj/
- python3 -m datasets.plane_preprocess.plane_fitting_sf
"""
if __name__ == "__main__":
    # make a soft link:
    # 1) ln -s /nfs/STG/SemanticDenseMapping/panji/Flow_datasets/FlyingThings3D/disparity /nfs/STG/SemanticDenseMapping/changjiang/data/sceneflow/FlyingThings3D/disparity
    # 2) ln -s /nfs/STG/SemanticDenseMapping/panji/Flow_datasets/FlyingThings3D/frames_cleanpass  /nfs/STG/SemanticDenseMapping/changjiang/data/sceneflow/FlyingThings3D/frames_cleanpass
    # 3) ln -s /nfs/STG/SemanticDenseMapping/panji/Flow_datasets/FlyingThings3D/frames_finalpass /nfs/STG/SemanticDenseMapping/changjiang/data/sceneflow/FlyingThings3D/frames_finalpass
    # 4) ln -s /nfs/STG/SemanticDenseMapping/panji/Flow_datasets/FlyingThings3D/optical_flow /nfs/STG/SemanticDenseMapping/changjiang/data/sceneflow/FlyingThings3D/optical_flow
    
    flyingthings3d_dir = '/nfs/STG/SemanticDenseMapping/changjiang/data/sceneflow/FlyingThings3D'
    driving_dir = '/nfs/STG/SemanticDenseMapping/changjiang/data'
    monkaa_dir = '/nfs/STG/SemanticDenseMapping/changjiang/data'
    
    sf_data_root = {
        'driving' : driving_dir,
        'monkaa' : monkaa_dir,
        'flyingthings3d' : flyingthings3d_dir,
    }
    main(sf_data_root, "splits/sceneflow_train_our.txt")
    #main(sf_data_root, "splits/sceneflow_test_our.txt")
