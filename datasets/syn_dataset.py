import torch
import random
import numpy as np
from keypoint_detection import load_detector, postprocess
from models import bird_model, load_regressor
import cv2
from utils.geometry import rot6d_to_rotmat, perspective_projection
from utils.renderer_p3d import RendererP3D

class synDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, n_samples=140, n_generation=14000):
        n_total = len(dataset)
        rand_idx = [random.choice(range(n_total)) for _ in range(n_samples)]

        self.device = 'cpu'

        self.bird = bird_model(device=self.device)
        regressor = load_regressor().to(self.device)
        predictor = load_detector().to(self.device)

        poses = []
        trans = []
        bones = []

        with torch.inference_mode():
            for i in rand_idx:
                imgs, target_kpts, target_masks, meta = dataset[i]
                imgs = imgs[None]
                # Prediction
                output = predictor(imgs.to(self.device))
                pred_kpts, pred_mask = postprocess(output)

                # Regression
                kpts_in = pred_kpts.reshape(pred_kpts.shape[0], -1)
                mask_in = pred_mask
                p_est, b_est = regressor(kpts_in, mask_in)
                pose, tran, bone = regressor.postprocess(p_est, b_est)
                
                poses.append(p_est.squeeze().cpu().numpy())
                bones.append(b_est.squeeze().cpu().numpy())

        poses = np.asarray(poses)
        bones = np.asarray(bones)

        # fit
        mu, cov = np.mean(poses, axis=0), np.cov(poses, rowvar=0)

        # sample
        self.poses = np.random.multivariate_normal(mu, cov, size=n_generation)
        self.bones = bones

        self.renderer = RendererP3D(faces=self.bird.dd['F'])
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        bone_idx = random.randint(0, len(self.bones)-1)
        p_est = self.poses[idx][None,:]
        b_est = (self.bones[bone_idx] + np.random.normal(loc=0, size=(24), scale=0.2))[None,:]

        p_est = torch.FloatTensor(p_est).to(self.device)
        b_est = torch.FloatTensor(b_est).to(self.device)

        pose, tran, bone = self.postprocess(p_est, b_est)
        global_t = tran.clone().to(self.device)
        bone_length = bone.clone().to(self.device)

        init_pose = self.transform_p(pose).to(self.device)
        global_orient = init_pose.clone()[:, :3]
        body_pose = init_pose.clone()[:, 3:]

        bird_output = self.bird(global_orient, body_pose, bone_length)
        global_txyz = self.transform_t(global_t)

        model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1).to(torch.float)
        kps = bird_output['keypoints'] + global_txyz.unsqueeze(1).to(torch.float)

        img, mask = self.renderer(model_mesh[0])
        proj_kps = self.renderer.cameras.get_projection_transform().transform_points(kps)
        proj_kps = proj_kps.reshape((36,))

        return img, mask.unsqueeze(0), proj_kps, p_est.squeeze(), b_est.squeeze()

    
    def transform_t(self, tran):
        """ From tran~[0,1] to tran_xyz of real unit.
            Simply intermediate layer. optimization objective is still tran.
        """
        tran_xyz = tran.clone()
        tran_xyz[:, 1] = tran_xyz[:, 1] - 1
        tran_xyz[:, 2] = tran_xyz[:, 2]*18 + 180

        return tran_xyz

    def transform_p(self, pose):
        """ From 9d rot pose to 3d axis-angle pose
            Fast enough for now as it is used only once.
        """
        batch_size = len(pose)
        
        pose = pose.to('cpu')
        pose = pose.detach().clone()
        pose = pose.reshape(batch_size, -1, 3, 3)
        new_pose = torch.zeros([batch_size, pose.shape[1]*3]).float()
        
        for i in range(batch_size):
            for j in range(pose.shape[1]):
                R = pose[i, j]
                aa, _ = cv2.Rodrigues(R.numpy())
                new_pose[i, 3*j:3*(j+1)] = torch.tensor(aa).squeeze()
            
        return new_pose
    
    def postprocess(self, p_est, b_est):
        """
        Convert 6d rotation to 9d rotation
        Input:
            p_est: pose_tran from forward()
            b_est: bone from forward()
        """
        pose_6d = p_est[:, :-3].contiguous()
        p_est_rot = rot6d_to_rotmat(pose_6d).view(-1, 25*9)
        
        pose = p_est_rot
        tran = p_est[:, -3:]
        bone = b_est

        return pose, tran, bone