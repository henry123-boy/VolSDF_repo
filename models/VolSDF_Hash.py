import copy
import importlib
import pdb
import sys
import cv2
import numpy as np
import os,sys,time
import torch.nn as nn
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__),"../external"))
import lpips
import tinycudann as tcnn
from utils.sfm_cal import SfMer
from external.pohsun_ssim import pytorch_ssim
from external.PointCould_vis.create_pointcloud import create_pointcloud as cret_ptc
import utils.utils as util
import utils.utils_vis as util_vis
from utils.utils import log,debug
from . import base
import utils.camera as camera
from utils.utils_vis import extract_mesh
import json
epsilon = 1e-6

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)
        self.lpips_loss = lpips.LPIPS(net="alex").to(opt.device)
    def build_networks(self,opt):
        super().build_networks(opt)
        return

    def load_dataset(self,opt,eval_split="val"):
        super().load_dataset(opt, eval_split=eval_split)
        # prefetch all training data
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(util.move_to_device(self.train_data.all, opt.device))
        return

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim, opt.optim.algo)
        self.optim = optimizer([dict(params=self.graph.parameters(), lr=opt.optim.lr)])
        ## check if beta is added into the net.parameters()
        # for para in self.graph.parameters():
        #     print(para.data)

        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
            if opt.optim.lr_end:
                assert (opt.optim.sched.type == "ExponentialLR")
                opt.optim.sched.gamma = (opt.optim.lr_end / opt.optim.lr) ** (1. / opt.max_iter)
            kwargs = {k: v for k, v in opt.optim.sched.items() if k != "type"}
            self.sched = scheduler(self.optim, **kwargs)
    def save_checkpoint(self,opt,ep=0,it=0,latest=False):
        self.graph.state_dict().update({"ln_beta":self.graph.beta})
        util.save_checkpoint(opt,self,ep=ep,it=it,latest=latest)
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group,opt.name,ep,it))
    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.graph.train()
        self.ep = 0  # dummy for timer
        # training
        #if self.iter_start == 0: self.validate(opt, 0)
        loader = tqdm.trange(opt.max_iter, desc="training", leave=False)
        num_scenes=len(self.train_data.all.idx)
        var = edict()
        #if self.iter_start== 0: self.validate(opt, 0)
        for self.it in loader:
            if self.it < self.iter_start: continue
            # set var to all available images
            # pdb.set_trace()
            # for var in self.train_data:
            #     #var = self.train_data.all
            #     var=edict(var)
            #     var.idx=[var.idx]
            #     var['image']=var['image'][None,:,:].to(opt.device)
            #     var['mask']=var['mask'][None,:].to(opt.device)
            #     var['intr']=var['intr'][None,:,:].to(opt.device)
            #     var['pose'] = var['pose'][None, :, :].to(opt.device)
            #     print("training the pic of idx{}".format(var.idx[0]))
            idx=torch.randperm(num_scenes,device=opt.device)[0]
            var.idx=[idx]
            var['image']=self.train_data.all['image'][idx,...].unsqueeze(0)
            var['mask'] = self.train_data.all['mask'][idx, ...].unsqueeze(0)
            var['intr'] = self.train_data.all['intr'][idx, ...].unsqueeze(0)
            var['pose'] = self.train_data.all['pose'][idx, ...].unsqueeze(0)
            self.train_iteration(opt, var, loader)
            if opt.optim.sched: self.sched.step()
            if self.it % opt.freq.val == 0: self.validate(opt, self.it)
            if self.it % opt.freq.ckpt == 0: self.save_checkpoint(opt, ep=None, it=self.it)
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        super().log_scalars(opt, var, loss, metric=metric, step=step, split=split)
        # log learning rate
        if split == "train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split, "lr"), lr, step)
            # compute PSNR
            psnrfr = -10 * (loss.render**2).log10()
            self.tb.add_scalar("{0}/{1}".format(split, "PSNR_render"), psnrfr, step)
            self.tb.add_scalar("{0}/{1}".format(split, "beta"), var.beta, step)
        else:
            psnrfr = -10 * (loss.render).log10()
            self.tb.add_scalar("{0}/{1}".format(split, "test_PSNR"), psnrfr, step)



    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train", eps=1e-10):
        if split=="val":
            H,W=opt.data.val_img_size
            self.tb.add_image("{0}/{1}".format(split, "ground_truth_rgb"), var.image.permute(0,2,1).view(3,H,W),step)
            self.tb.add_image("{0}/{1}".format(split, "predicted_rgb"), var.rgb.permute(0, 2, 1).view(3, H, W),step)
            if (step%opt.freq.vis_mesh==0)&(step>0):
                os.makedirs("{0}/mesh".format(opt.output_path),exist_ok=True)
                opt.mesh_dir="{0}/mesh".format(opt.output_path)
                extract_mesh(
                    self.graph.VolSDF,
                    filepath=os.path.join(opt.mesh_dir, '{:08d}.ply'.format(step)),
                    volume_size=2.0,
                    log=log,
                    show_progress=True)
        return

class VolSDF(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.define_network(opt)
        self.opt = opt
        self.rescale=4.0

    def define_network(self, opt):
        with open(opt.arch.mlp_surface_config) as config_file:
            self.config_surface = json.load(config_file)
        with open(opt.arch.mlp_radiance_config) as config_file:
            self.config_radiance = json.load(config_file)
        print("building the mlp for surface with hash encoding......")
        self.hash_ec=tcnn.Encoding(3,encoding_config=self.config_surface["encoding"])
        # self.mlp_surface = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=16,
        #                                       encoding_config=self.config_surface["encoding"], network_config=self.config_surface["network"])
        self.mlp_surface=nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,65),
        )

        print("building the mlp for radiance......")
        # sphere harmonics embedding
        self.sh_ec=tcnn.Encoding(3,encoding_config=self.config_radiance["encoding"])
        self.mlp_radiance = tcnn.Network(n_input_dims=87,n_output_dims=3,network_config=self.config_radiance["network"])


    def infer_sdf(self, p,mode="only_sdf"):
        # p_rescale=(p+self.rescale) / (2*self.rescale)
        p_rescale = p/self.rescale
        # print(p_rescale.max())
        # print(p_rescale.min())
        # if p_rescale.requires_grad==True:
        #     pdb.set_trace()
        #     x=self.hash_ec(p_rescale.view(-1,3))
        #     pdb.set_trace()
        feat=self.hash_ec(p_rescale.view(-1,3)).float()
        feat=self.mlp_surface(feat)
        if len(p_rescale.shape)==3:
            p_rescale=p_rescale.unsqueeze(0)
        # elif len(p_rescale.shape)==2:
        #     pdb.set_trace()
        if len(p_rescale.shape) != 2:
            batch,n_rays,n_pts,_=p_rescale.shape
            feat=feat.view(batch,n_rays,n_pts,65)
        if mode=="only_sdf":
            return feat[...,:1]
        else:
            return feat
    def infer_rad(self,points_3D, normals, ray_enc, feat):
        # p_rescale = (points_3D + self.rescale) / (2 * self.rescale)
        p_rescale = points_3D / self.rescale
        rendering_input = torch.cat([p_rescale.view(-1,3), ray_enc, normals.view(-1,3), feat.view(-1,65)], dim=-1)
        x=self.mlp_radiance(rendering_input)
        return x

    def gradient(self, p):
        with torch.enable_grad():
            p.requires_grad_(True)
            y = self.infer_sdf(p,mode="only_sdf")
            #y=self.hash_ec(p.view(-1,3))
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients
    def forward(self, opt,points_3D, ray_unit=None, only_sdf=False, return_addocc=False):
        x = self.infer_sdf(points_3D,mode="ret_feat")
        if only_sdf:
            sdf=x[..., :1]
            return torch.min(sdf, opt.VolSDF.obj_bounding_radius - points_3D.norm(dim=-1, keepdim=True))
        elif ray_unit is not None:
            batch, n_rays, n_pts, _ = ray_unit.shape
            ray_enc=self.sh_ec(ray_unit.reshape(-1,3))
            normals = self.gradient(points_3D)
            # normals = n / (torch.norm(n, dim=-1, keepdim=True)+1e-6)
            # the normals have to be the tensor without grad
            rgb = self.infer_rad(points_3D, normals, ray_enc, x)
            rgb=rgb.view(batch,n_rays,n_pts,3)
            if return_addocc:
                sdf = x[..., :1]
                return rgb, torch.min(sdf, opt.VolSDF.obj_bounding_radius - points_3D.norm(dim=-1, keepdim=True)),normals
            else:
                return rgb
    def forward_samples(self,opt,center,ray,depth_samples,mode=None):
        points_3D_samples = camera.get_3D_points_from_depth(opt,center,ray,depth_samples,multi_samples=True) # [B,HW,N,3]
        if opt.VolSDF.view_dep:
            ray_unit = torch_F.normalize(ray,dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[...,None,:].expand_as(points_3D_samples) # [B,HW,N,3]
        else: ray_unit_samples = None
        rgb_samples,sdf,normal = self.forward(opt,points_3D_samples,ray_unit=ray_unit_samples,return_addocc=True) # [B,HW,N],[B,HW,N,3]
        return rgb_samples,sdf,normal
    def composite(self,opt,ray,rgb_samples,density_samples,depth_samples):
        ray_length = ray.norm(dim=-1,keepdim=True) # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[...,1:,0]-depth_samples[...,:-1,0] # [B,HW,N-1]
        #depth_intv_samples = torch.cat([depth_intv_samples,torch.empty_like(depth_intv_samples[...,:1]).fill_(1e10)],dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples*ray_length # [B,HW,N-1]
        sigma_delta = density_samples[...,:-1]*dist_samples # [B,HW,N-1]
        alpha = 1-(-sigma_delta).exp_() # [B,HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta],dim=2).cumsum(dim=2)).exp_()[...,:-1] # [B,HW,N]
        prob = (T*alpha)[...,None] # [B,HW,N,1]
        # integrate RGB and depth weighted by probability
        rgb = (rgb_samples[...,:-1,:]*prob).sum(dim=2) # [B,HW,3]
        opacity = prob.sum(dim=2) # [B,HW,1]
        # pdb.set_trace()
        return rgb,prob

# def function(x: torch.tensor) 这样可以指定输入变量类型

# ============================ computation graph for forward/backprop ===============================

class Graph(base.Graph):
    def __init__(self,opt):
        super().__init__(opt)
        self.VolSDF=VolSDF(opt)                                           # add VolSDF into cal graph
        self.speed_factor = opt.VolSDF.speed_factor
        beta_init=np.log(opt.VolSDF.beta_init) / self.speed_factor
        beta_optim=torch.from_numpy(np.array([beta_init], dtype=np.float32)).to(opt.device)
        self.beta=nn.Parameter(data=beta_optim, requires_grad=True)       # when declare a variable of nn.Parameter(), it will be added to net.parameters()
        # self.parameters.update({"beta":nn.Parameter(data=beta_optim, requires_grad=True)})
        # self.state_dict().update({"beta":self.beta})
        # pdb.set_trace()
        self.obj_bounding_radius = opt.VolSDF.obj_bounding_radius         # see supple, the camera pose and object were normalized to a certain sphere
        self.use_sphere_bg = (opt.VolSDF.outside_scene=="builtin")        # if use the nerf++ to learn the bg
        if self.use_sphere_bg==False:                                     # chose nerf++ as the bg
            model=importlib.import_module("models.nerf")
            NeRF=model.NeRF
            self.nerf_outside = NeRF(input_ch=4, multires=10, multires_view=4, use_view_dirs=True)
    def sdf_to_sigma(self, sdf, alpha, beta):
        # alpha and beta are learnable parameters
        # sdf [B,HW,N,1]
        exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
        psi = torch.where(sdf > 0, exp, 1 - exp)
        return alpha * psi

    def error_bound(self,d_vals, sdf, alpha, beta):
        """
        @ Bound on the opacity approximation error
        mentioned in paper, bounds error is calculated between xi-1 and xi
        Args:
            d_vals: [..., N_pts]
            sdf:    [..., N_pts]
        Return:
            bounds: [..., N_pts-1]
        """
        device = sdf.device
        sigma = self.sdf_to_sigma(sdf, alpha, beta)     # [..., N_pts]
        sdf_abs_i = torch.abs(sdf)                      # [..., N_pts-1]
        # delta_i = (d_vals[..., 1:] - d_vals[..., :-1]) * rays_d.norm(dim=-1)[..., None]
        delta_i = d_vals[..., 1:] - d_vals[..., :-1]  # NOTE: already real depth
        # [..., N_pts-1]. R(t_k) of the starting point of the interval.
        R_t = torch.cat(
            [
                torch.zeros([*sdf.shape[:-1], 1], device=device),
                torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
            ], dim=-1)[..., :-1] # [..., N_pts-1]
        d_i_star = torch.clamp_min(0.5 * (sdf_abs_i[..., :-1] + sdf_abs_i[..., 1:] - delta_i), 0.) # [..., N_pts-1]
        errors = alpha / (4 * beta) * (delta_i ** 2) * torch.exp(-d_i_star / beta)  # [..., N_pts-1]. E(t_{k+1}) of the ending point of the interval.
        errors_t = torch.cumsum(errors, dim=-1) # [..., N_pts-1]
        bounds = torch.exp(-R_t) * (torch.exp(errors_t) - 1.)
        # TODO: better solution
        #     # NOTE: nan comes from 0 * inf
        #     # NOTE: every situation where nan appears will also appears c * inf = "true" inf, so below solution is acceptable
        bounds[torch.isnan(bounds)] = np.inf
        return bounds
    def forward_ab(self):
        beta = torch.exp(self.beta * self.speed_factor)
        return 1./beta, beta
    def forward_surface(self, pos):
        '''
        :param pos: the 3D space point [B,HW,N,3]
        :return:
        '''
        sdf = self.VolSDF.infer_sdf(pos,mode="only_sdf")
        if self.use_sphere_bg:              # which filter the pos outside the sphere
            return torch.min(sdf, self.obj_bounding_radius - pos.norm(dim=-1,keepdim=True))
        else:
            return sdf
    def forward(self,opt,var,mode=None):
        batch_size = len(var.idx)
        pose = self.get_pose(opt, var, mode=mode)
        # render images
        if opt.VolSDF.rand_rays and mode in ["train", "test-optim"]:
            # sample random rays for optimization
            #var.ray_idx= torch.arange(0,opt.VolSDF.rand_rays )
            var.ray_idx = torch.randperm(opt.H * opt.W, device=opt.device)[:opt.VolSDF.rand_rays // batch_size]
            ret = self.render(opt, pose, intr=var.intr, ray_idx=var.ray_idx, mode=mode)  # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            ret = self.render_by_slices(opt, pose, intr=var.intr, mode=mode)
        var.update(ret)
        return var
    def render_by_slices(self,opt,pose,intr=None,mode=None):
        ret_all = edict(rgb=[])
        H,W=opt.data.val_img_size
        for c in range(0,H*W,opt.VolSDF.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.VolSDF.rand_rays,H*W),device=opt.device)
            ret = self.render(opt,pose,intr=intr,ray_idx=ray_idx,mode=mode) # [B,R,3],[B,R,1]
            ret_all["rgb"].append(ret["rgb"])
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1)
        # t1=time.perf_counter()
        # print(str(t1-t0))
        return ret_all

    def get_pose(self,opt,var,mode=None):
        return var.pose
    def compute_loss(self,opt,var,ret,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image
        if opt.VolSDF.rand_rays and mode in ["train", "test-optim"]:
            image = image[:, var.ray_idx,:]
        # compute image losses
        if mode=="train":
            if opt.loss_weight.render is not None:
                loss.render = nn.functional.l1_loss(var.rgb, image)
            if opt.loss_weight.w_eikonal is not None:
                loss.w_eikonal = self.MSE_loss(torch.norm(var.normal,dim=-1),torch.ones_like(torch.norm(var.normal,dim=-1)).to(opt.device))
        else:
            loss.render=self.MSE_loss(var.rgb,var.image)

        return loss

    def sample_depth(self, opt, batch_size, num_rays=None):
        depth_min, depth_max = opt.VolSDF.depth.min_d,opt.VolSDF.depth.max_d
        num_rays = num_rays or opt.H * opt.W
        rand_samples = 0.5
        rand_samples += torch.arange(opt.VolSDF.sample_intvs, device=opt.device)[None, None, :,
                        None].float()  # [B,HW,N,1]
        depth_samples = rand_samples / opt.VolSDF.sample_intvs * (depth_max - depth_min) + depth_min  # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1 / (depth_samples + 1e-8),
        )[opt.VolSDF.depth.param]
        return depth_samples
    def volsdf_sampling(self,opt,center,ray):
        '''
        :param opt:
        :param unisamp:       uniform sample points [B,HW,N,3]
        :param center:   [B,HW,3]        the center of camera
        :param ray:      [B,HW,3]        the ray
        :return:
        '''
        with torch.no_grad():
            # init beta+
            max_d=torch.tensor([opt.VolSDF.depth.max_d]).to(opt.device)*torch.ones_like(center[:,:,0])
            eps_=torch.tensor([opt.VolSDF.eps]).to(opt.device)*torch.ones_like(center[:,:,0])
            beta = torch.sqrt((max_d ** 2) / (4 * (opt.VolSDF.sample_intvs - 1) * torch.log(1 + eps_)))
            alpha=1./beta
            # uniform sampling
            batch_size=center.shape[0]
            depth_samples = self.sample_depth(opt, batch_size, num_rays=ray.shape[1])  # [B,HW,N,1]
            unisamp = camera.get_3D_points_from_depth(opt, center, ray, depth_samples,multi_samples=True)    # [B,HW,N,3]
            sdf=self.forward_surface(unisamp)     # [B,HW,N,1]
            alpha_graph,beta_graph=self.forward_ab()           # [attain the alpha and beta
            sigma = self.sdf_to_sigma(sdf, alpha_graph, beta_graph)    # attain the sigma
            net_bounds_max = self.error_bound(depth_samples.squeeze(-1), sdf.squeeze(-1), alpha_graph, beta_graph).max(dim=-1).values
            mask = net_bounds_max > opt.VolSDF.eps

            # cal bounds_error by beta++
            bounds = self.error_bound(depth_samples.squeeze(-1), sdf.squeeze(-1), alpha[:,:,None], beta[:,:,None])
            bounds_masked = bounds[mask]

            final_fine_dvals = torch.zeros([*mask.shape, opt.VolSDF.final_sample_intvs]).to(opt.device)       # [B,HW,N_final]
            final_iter_usage = torch.zeros([*mask.shape]).to(opt.device)                                      # [B,HW]
            final_converge_flag = torch.zeros([*mask.shape],dtype=torch.bool).to(opt.device)                                   # [B,HW]
            # if the sampling is fine
            if (~mask).sum() > 0:
                final_fine_dvals[~mask] = self.opacity_to_sample(opt,depth_samples.repeat([*mask.shape,1,1]).squeeze(-1)[~mask], sdf.squeeze(-1)[~mask], alpha_graph, beta_graph,*mask.shape)[:,:,0]
                final_iter_usage[~mask] = 0
            final_converge_flag[~mask] = True

            current_samp = depth_samples.shape[-2]
            it_algo = 0
            depth_samples=depth_samples.squeeze(-1).repeat([*bounds.shape[:-1],1])
            sdf=sdf.squeeze(-1)
            while it_algo < opt.VolSDF.max_upsample_iter:
                print("sampling algo iteration{}".format(it_algo))
                it_algo += 1
                if mask.sum() > 0:
                    # intuitively, the bigger bounds error, the more weights in sampling
                    upsampled_d_vals_masked=self.sample_pdf(depth_samples[mask],bounds_masked,opt.VolSDF.sample_intvs+2,det=True)[..., 1:-1]
                    # upsample_depth
                    # depth_samples=depth_samples.expand(*mask.shape,depth_samples.shape[-1])
                    depth_samples = torch.cat([depth_samples, torch.zeros([*depth_samples.shape[:2], opt.VolSDF.sample_intvs]).to(opt.device)], dim=-1)
                    sdf = torch.cat([sdf, torch.zeros([*sdf.shape[:2], opt.VolSDF.sample_intvs]).to(opt.device)], dim=-1)

                    depth_samples_masked=depth_samples[mask]
                    sdf_masked=sdf[mask]
                    # add the hierachical sampled depth into the new one
                    depth_samples_masked[..., current_samp:current_samp + opt.VolSDF.sample_intvs] = upsampled_d_vals_masked
                    depth_samples_masked, sort_indices_masked = torch.sort(depth_samples_masked, dim=-1)
                    # add the hierachical sampled sdf into the new one
                    tem_samp_pts3D = center[mask][..., None, :] + ray[mask][..., None, :] * upsampled_d_vals_masked[..., :, None]
                    sdf_masked[..., current_samp:current_samp + opt.VolSDF.sample_intvs] = self.forward_surface(tem_samp_pts3D)[:,:,0]
                    sdf_masked = torch.gather(sdf_masked, dim=-1, index=sort_indices_masked)   # reorder the sdf by the corresponded orders of depth_sample
                    # update depth_samples and sdf
                    depth_samples[mask] = depth_samples_masked
                    sdf[mask] = sdf_masked
                    current_samp += opt.VolSDF.sample_intvs

                    # using the beta_graph to cal the new bound error
                    net_bounds_max = self.error_bound(depth_samples, sdf, alpha_graph,beta_graph).max(dim=-1).values
                    sub_mask_of_mask = net_bounds_max[mask] > opt.VolSDF.eps
                    converged_mask = mask.clone()
                    converged_mask[mask] = ~sub_mask_of_mask

                    if (converged_mask).sum() > 0:
                        final_fine_dvals[converged_mask] = self.opacity_to_sample(opt, depth_samples[converged_mask], sdf[converged_mask], alpha_graph, beta_graph, *converged_mask.shape).squeeze(-1)
                        final_iter_usage[converged_mask] = it_algo
                        final_converge_flag[converged_mask] = True
                    # using the bisection method to find the new beta++
                    if (sub_mask_of_mask).sum() > 0:
                        # mask-the-mask approach
                        new_mask = mask.clone()
                        new_mask[mask] = sub_mask_of_mask
                        # [Submasked, 1]
                        beta_right = beta[new_mask]
                        beta_left = beta_graph * torch.ones_like(beta_right, device=opt.device)
                        d_samp_tmp = depth_samples[new_mask]
                        sdf_tmp = sdf[new_mask]
                        # ----------------
                        # Bisection iterations
                        for _ in range(opt.VolSDF.max_bisection_itr):
                            beta_tmp = 0.5 * (beta_left + beta_right)
                            alpha_tmp = 1. / beta_tmp
                            # alpha_tmp = alpha_net
                            # [Submasked]
                            bounds_tmp_max = self.error_bound(d_samp_tmp, sdf_tmp, alpha_tmp[:,None], beta_tmp[:,None]).max(dim=-1).values
                            beta_right[bounds_tmp_max <= opt.VolSDF.eps] = beta_tmp[bounds_tmp_max <= opt.VolSDF.eps]
                            beta_left[bounds_tmp_max > opt.VolSDF.eps] = beta_tmp[bounds_tmp_max > opt.VolSDF.eps]
                        # updata beta++ and alpha++
                        beta[new_mask] = beta_right
                        alpha[new_mask] = 1. / beta[new_mask]

                        # ----------------
                        # after upsample, the remained rays that not yet converged.
                        # ----------------
                        bounds_masked = self.error_bound(d_samp_tmp, sdf_tmp, alpha[new_mask][:,None], beta[new_mask][:,None])
                        # bounds_masked = error_bound(d_vals_tmp, rays_d_tmp, sdf_tmp, alpha_net, beta[new_mask])
                        bounds_masked = torch.clamp(bounds_masked, 0, 1e5)  # NOTE: prevent INF caused NANs

                        # mask = net_bounds_max > eps   # NOTE: the same as the following
                        mask = new_mask
                    else:
                        break
                else:
                    break
            # ----------------
            # for rays that still not yet converged after max_iter, use the last beta+
            # ----------------
            if (~final_converge_flag).sum() > 0:
                print("existing rays which did not converge")
                beta_plus = beta[~final_converge_flag]
                alpha_plus = 1./beta_plus
                final_fine_dvals[~final_converge_flag] = self.opacity_to_sample(opt, depth_samples[~final_converge_flag],sdf[~final_converge_flag], alpha_plus[:,None], beta_plus[:,None],*final_converge_flag.shape).squeeze(-1)
                final_iter_usage[~final_converge_flag] = -1
            beta[final_converge_flag] = beta_graph
            depth_coarse = self.sample_depth(opt, batch_size, num_rays=ray.shape[1]).squeeze(-1).repeat(*final_fine_dvals.shape[:2],1)
            depth_sample_final=torch.cat([final_fine_dvals,depth_coarse],dim=-1)
            depth_sample_final, _ = torch.sort(depth_sample_final, dim=-1)
            return depth_sample_final, beta, final_iter_usage

    def sample_pdf(self,bins, weights, N_importance, det=False, eps=1e-5):
        # device = weights.get_device()
        device = weights.device
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat(
            [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
        )  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0.0, 1.0, steps=N_importance, device=device)
            u = u.expand(list(cdf.shape[:-1]) + [N_importance])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)
        u = u.contiguous()

        # Invert CDF
        inds = torch.searchsorted(cdf.detach(), u, right=False)

        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, cdf.shape[-1] - 1)
        # (batch, N_importance, 2) ==> (B, batch, N_importance, 2)
        inds_g = torch.stack([below, above], -1)

        matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]  # fix prefix shape

        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
        bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)  # fix prefix shape

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples
    def opacity_to_sample(self,opt,depth_samp,sdf,alpha,beta,B,HW):
        sigma = self.sdf_to_sigma(sdf, alpha, beta)
        delta_i = depth_samp[..., 1:] - depth_samp[..., :-1]  # NOTE: already real depth
        R_t = torch.cat([torch.zeros([*sdf.shape[:-1], 1], device=opt.device),torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)], dim=-1)[..., :-1]
        # -------------- a fresh set of \hat{O}
        opacity_approx = 1 - torch.exp(-R_t)
        fine_dvals = self.sample_depth_from_opacity(opt,depth_samp, opacity_approx)
        return fine_dvals
    def sample_depth_from_opacity(self,opt,depth_sample,opacity_approx):
        '''
        :param opt:
        :param opacity_approx:  [B,HW,N,1]
        :return:
        '''
        opacity_approx = torch.cat(
            [torch.zeros_like(opacity_approx[..., :1], device=opt.device), opacity_approx], -1
        )
        grid = torch.linspace(0, 1, opt.VolSDF.final_sample_intvs + 1, device=opt.device)  # [Nf+1]
        unif = 0.5 * (grid[:-1] + grid[1:]).repeat(*opacity_approx.shape[:-1], 1)  # [B,HW,Nf]
        idx = torch.searchsorted(opacity_approx, unif, right=False)  # [B,HW,Nf] \in {1...N}

        # inverse transform sampling from CDF
        depth_bin = depth_sample
        depth_low = depth_bin.gather(dim=-1, index=(idx - 1).clamp(min=0))  # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=-1, index=idx.clamp(max=opacity_approx.shape[-1]-1))  # [B,HW,Nf]
        cdf_low = opacity_approx.gather(dim=-1, index=(idx - 1).clamp(min=0))  # [B,HW,Nf]
        cdf_high = opacity_approx.gather(dim=-1, index=idx.clamp(max=opacity_approx.shape[-1]-1))  # [B,HW,Nf]
        # linear interpolation
        t = (unif - cdf_low) / (cdf_high - cdf_low + 1e-8)  # [B,HW,Nf]
        depth_samples = depth_low + t * (depth_high - depth_low)  # [B,HW,Nf]
        return depth_samples[..., None]  # [B,HW,Nf,1]
    def render(self,opt,pose,intr=None,ray_idx=None,mode=None):
        # ray_idx=torch.arange(0,100)
        batch_size = len(pose)
        center, ray = camera.get_center_and_ray(opt, torch.inverse(pose)[:,:3,:4], intr=intr,mode=mode)  # [B,HW,3]
        while ray.isnan().any():  # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center, ray = camera.get_center_and_ray(opt, pose, intr=intr)  # [B,HW,3]
        if ray_idx is not None:
            # consider only subset of rays
            center, ray = center[:, ray_idx], ray[:, ray_idx]
        # render with main MLP
        if (opt.VolSDF.volsdf_sampling)&(mode=="train"):
            depth_samples,beta, _ = self.volsdf_sampling(opt,center,ray)  # [B,HW,N,1]
        else:
            depth_samples=self.sample_depth(opt, batch_size, num_rays=ray.shape[1]).squeeze(-1).repeat(*center.shape[:2],1)
        # else:
        #     depth_samples=torch.linspace(opt.VolSDF.depth.min_d,opt.VolSDF.depth.max_d,opt.VolSDF.sample_intvs).to(opt.device)
        #     depth_samples=depth_samples[None,None,:].repeat(*center.shape[:2],1)
        #     pdb.set_trace()
        #     beta=self.beta.expand(*depth_samples.shape)
        alpha_render,beta_render=self.forward_ab()
        depth_samples=depth_samples[:,:,:,None]
        rgb_samples, sdf, normal = self.VolSDF.forward_samples(opt, center, ray, depth_samples, mode=mode)
        density_samples=self.sdf_to_sigma(sdf,alpha_render,beta_render)
        rgb,prob = self.VolSDF.composite(opt, ray, rgb_samples.squeeze(-1), density_samples.squeeze(-1), depth_samples)
        #_,ind=(prob[...,0]).max(dim=-1)
        #normal = torch.gather(normal, dim=-2, index=ind[..., None].repeat([*(len(normal.shape) - 1) * [1], 3]))
        #pts_3D_unif = torch.empty_like(normal).uniform_(-opt.VolSDF.obj_bounding_radius,opt.VolSDF.obj_bounding_radius).to(opt.device)
        #normal_extra = self.VolSDF.gradient(pts_3D_unif)
        ret = edict(rgb=rgb,normal=normal,beta=beta_render)  # [B,HW,K]
        return ret