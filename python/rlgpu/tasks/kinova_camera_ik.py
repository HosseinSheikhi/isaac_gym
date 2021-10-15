import numpy as np
import os
import torch
import math

from torch._C import device, dtype
import imageio

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

class KinovaCameraIK(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, enable_camera_sensors=False):
        # create directory for saved images
        # todo it is temporary
        self.img_dir = "kinova_camera_images"
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.parse_config(cfg, sim_params, physics_engine, device_type, device_id, headless)

        # the base class will call a few functin before returning the  program here
        super().__init__(cfg=self.cfg, enable_camera_sensors=enable_camera_sensors)

        self.set_tensors()

        self.reset(torch.arange(self.num_envs, device=self.device))
    
    def parse_config(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        """
            takes in some configuration parameters, sets and define a few variables
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        
        self.camera_width = self.cfg["env"]["camera"]["width"]
        self.camera_height = self.cfg["env"]["camera"]["height"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.dt = 1/60.

        num_obs = 13 # Dof pos (7) + to_target (3pos+3ori)
        num_acts = 6 # our action is 3 pose and 3 orientation error (look at the kinova_cube_it orientation error)

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

    def set_tensors(self):
        """
            creates the necessary tensors 
        """
        # get gym GPU dof tensors and create some wrapper and slices
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) # The shape of the tensor is (num_envs, num_dofs* 2).
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # contains the kinova+table+box dofs
        self.kinova_dof_state = self.dof_state.view(self.num_envs, -1, 2)[: , :self.num_kinova_dofs] # just get the kinova dofs
        self.kinova_dof_pos = self.kinova_dof_state[..., 0]
        self.kinova_dof_vel = self.kinova_dof_state[..., 1]
        
        # get gym GPU rigid body state tensors and create some wrappers
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) # The shape of the rigid body state tensor is (num_envs, num_rigid_bodies* 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13) # we need it to get the end efffector pose & orientation
        self.end_effector_pos = self.rigid_body_states[:, self.endefector_handle][:, 0:3]
        self.end_effector_rot = self.rigid_body_states[:, self.endefector_handle][:, 3:7]
        self.num_bodies = self.rigid_body_states.shape[1]
        
        # get gym GPU jacobian tensor and create some wrappers
        # for fixed-base kinova, tensor has shape (num envs, num_links=8, 6, num_dofs=7)
        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "kinova") # you must specify the actor name
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        # jacobian entries corresponding to end effector
        self.j_eef = self.jacobian[:, self.endeffector_index - 1, :]
        print("Jacobian shape: " ,self.jacobian.shape)
        print("Jacobian end effector entry shape: ", self.j_eef.shape)

        # refresh all the tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.kinova_default_dof_pos = to_torch([0.0, -1.0, 0.0, +2.6, -1.57, 0.0, 0.0], device=self.device)

        self.kinova_target_dof_pos = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device) # to call the set_dof_position_target
        self.global_indices = torch.arange(self.num_envs , dtype=torch.int32, device=self.device).view(self.num_envs, -1) # todo am not sure about dims
        
        self.goal = to_torch([0.7,-0.4,0.5,-0.6016797, 0, -0.6016797, 0.525322], device=self.device).repeat((self.num_envs, 1)) # Goal is the end effector position


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        asset_root = "../../assets"
        kinova_asset_file = "urdf/kinova_gen3/urdf/GEN3_URDF_V12.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            kinova_asset_file = self.cfg["env"]["asset"].get("assetFileNameKinova", kinova_asset_file)

        kinova_asset, kinova_dof_props = self.load_kinova_asset(asset_root, kinova_asset_file)
        
        kinova_start_pose = gymapi.Transform()
        kinova_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        kinova_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        # define camera properties
        cam_props = gymapi.CameraProperties()
        cam_props.width = self.camera_width
        cam_props.height = self.camera_height
        cam_props.enable_tensors = True

        self.kinovas = []
        self.sensors_tensors = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            kinova_actor = self.gym.create_actor(env_ptr, kinova_asset, kinova_start_pose, "kinova", i, 2)
            self.gym.set_actor_dof_properties(env_ptr, kinova_actor, kinova_dof_props)

            # add sensor
            sensor_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
            camera_offset = gymapi.Vec3(0, -0.1, -0.1)
            camera_rotation = gymapi.Quat.from_euler_zyx(math.pi/2.0,math.pi/2.0,0) 
            kinova_endeffector_handle = self.gym.find_actor_rigid_body_handle(env_ptr, kinova_actor,  "EndEffector_Link")
            self.gym.attach_camera_to_body(sensor_handle, env_ptr, kinova_endeffector_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            
            # obtain sensor tensor
            sensor_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, sensor_handle, gymapi.IMAGE_COLOR)
            print("Got sensor tensor with shape", sensor_tensor.shape)
            # wrap sensor tensor in a pytorch tensor
            torch_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor)
            self.sensors_tensors.append(torch_sensor_tensor)
            print("  Torch sensor tensor device:", torch_sensor_tensor.device)
            print("  Torch sensor tensor shape:", torch_sensor_tensor.shape)

            self.envs.append(env_ptr)
            self.kinovas.append(kinova_actor)

        self.endefector_handle = kinova_endeffector_handle
    
    def load_kinova_asset(self, asset_root, kinova_asset_file):
        # load kinova asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        #asset_options.collapse_fixed_joints = True # will colapse the end effector link
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        kinova_asset = self.gym.load_asset(self.sim, asset_root, kinova_asset_file, asset_options)

        #todo how should I choose damping and stifness?
        kinova_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        kinova_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80], dtype=torch.float, device=self.device)

        self.num_kinova_bodies = self.gym.get_asset_rigid_body_count(kinova_asset)
        self.num_kinova_dofs = self.gym.get_asset_dof_count(kinova_asset)

        print("num kinova bodies: ", self.num_kinova_bodies)
        print("num kinova dofs: ", self.num_kinova_dofs)

        # set kinova dof properties
        kinova_dof_props = self.gym.get_asset_dof_properties(kinova_asset)
        self.kinova_dof_lower_limits = []
        self.kinova_dof_upper_limits = []
        for i in range(self.num_kinova_dofs):
            kinova_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                kinova_dof_props['stiffness'][i] = kinova_dof_stiffness[i]
                kinova_dof_props['damping'][i] = kinova_dof_damping[i]
            else:
                kinova_dof_props['stiffness'][i] = 7000.0
                kinova_dof_props['damping'][i] = 50.0

            self.kinova_dof_lower_limits.append(kinova_dof_props['lower'][i])
            self.kinova_dof_upper_limits.append(kinova_dof_props['upper'][i])

        self.kinova_dof_lower_limits = to_torch(self.kinova_dof_lower_limits, device=self.device)
        self.kinova_dof_lower_limits = tensor_clamp(self.kinova_dof_lower_limits, -3.14* torch.ones_like(self.kinova_dof_lower_limits), +3.14* torch.ones_like(self.kinova_dof_lower_limits))
        self.kinova_dof_upper_limits = to_torch(self.kinova_dof_upper_limits, device=self.device)
        self.kinova_dof_upper_limits = tensor_clamp(self.kinova_dof_upper_limits, -3.14*torch.ones_like(self.kinova_dof_upper_limits), +3.14*torch.ones_like(self.kinova_dof_upper_limits))
        self.kinova_dof_speed_scales = torch.ones_like(self.kinova_dof_lower_limits)

         # get link index of endeffector hand, which we will use as end effector
        kinova_link_dict = self.gym.get_asset_rigid_body_dict(kinova_asset)
        self.endeffector_index = kinova_link_dict[ "EndEffector_Link"]
        return kinova_asset, kinova_dof_props

    def compute_reward(self, actions):
        
        
        self.rew_buf[:], self.reset_buf[:] = compute_kinova_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.end_effector_pos, self.end_effector_rot, self.goal,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale,
            self.action_penalty_scale, self.max_episode_length
        )

    def compute_observations(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # render sensors and refresh camera tensors
        frame_no = self.gym.get_frame_count(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)


        dof_pos_scaled = (2.0 * (self.kinova_dof_pos - self.kinova_dof_lower_limits)
                          / (self.kinova_dof_upper_limits - self.kinova_dof_lower_limits) - 1.0)
        to_target_pos =  self.goal[:,:3] - self.end_effector_pos # todo normalize pos
        to_target_pos = (2.0 * (to_target_pos - (-1))
                         /  (1 - (-1))-1) # goal is within max 1 meter
        to_target_err = orientation_error(self.goal[:,3:], self.end_effector_rot)
        self.obs_buf = torch.cat((dof_pos_scaled, to_target_pos, to_target_err), dim=-1)
        if frame_no < 10000 and frame_no%50==0:
            #for i in range(self.num_envs):
                # write tensor to image
            i=15
            fname = os.path.join(self.img_dir, "sensor-%04d-%04d.png" % (frame_no, i))
            sensor_img = self.sensors_tensors[i].cpu().numpy()
            imageio.imwrite(fname, sensor_img)

        self.gym.end_access_image_tensors(self.sim)
        return self.obs_buf

    def reset(self, env_ids):
        """
            reset kinova (or any other actor): go to a random position in a vaccinity of the default pose with 0 veliocities
        """
        pos = tensor_clamp(
            self.kinova_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_kinova_dofs), device=self.device) - 0.5),
            self.kinova_dof_lower_limits, self.kinova_dof_upper_limits)
        self.kinova_dof_pos[env_ids, :] = pos
        self.kinova_dof_vel[env_ids, :] = torch.zeros_like(self.kinova_dof_vel[env_ids])
        self.kinova_target_dof_pos[env_ids, :self.num_kinova_dofs] = pos

        multi_env_ids_int32 = self.global_indices[env_ids].flatten()

        # todo why set both pose and state (state itself has pos and vel)?
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.kinova_target_dof_pos),
                                                    gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        actions = actions.unsqueeze(-1)
        self.gym.refresh_jacobian_tensors(self.sim)
        # we assume actions are dpose = [pos_err, orn_err]
        self.actions = actions.clone().to(self.device)
        
        # solve damped least squares
        # https://www.daslhub.org/unlv/wiki/doku.php?id=robotic_manipulators_ik
        # https://www.tandfonline.com/doi/full/10.1080/01691864.2020.1780151 look at section 2
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        d = 0.05  # damping term
        lmbda = torch.eye(6).to(self.device) * (d ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ self.actions).view(self.num_envs, 7, 1)
        
        # update position targets
        dof_pos = self.kinova_dof_pos.view(self.num_envs, 7, 1)
        self.kinova_target_dof_pos[:, :self.num_kinova_dofs] = dof_pos.squeeze(-1) + u.squeeze(-1)

        self.gym.set_dof_position_target_tensor(self.sim,
                                                 gymtorch.unwrap_tensor(self.kinova_target_dof_pos))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            #self.gym.refresh_rigid_body_state_tensor(self.sim)
            goal_pos = self.goal[:,:3]
            goal_rot = self.goal[:,3:]
            for i in range(self.num_envs):
                px = (self.end_effector_pos[i] + quat_apply(self.end_effector_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.end_effector_pos[i] + quat_apply(self.end_effector_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.end_effector_pos[i] + quat_apply(self.end_effector_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.end_effector_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (goal_pos[i] + quat_apply(goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (goal_pos[i] + quat_apply(goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (goal_pos[i] + quat_apply(goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = goal_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])



@torch.jit.script
def compute_kinova_reward(
    reset_buf, progress_buf, actions,
    end_effector_pos, end_effector_rot, goal,
    num_envs, dist_reward_scale, rot_reward_scale,
            action_penalty_scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from hand to the drawer
    d = torch.norm(end_effector_pos - goal[:,:3], p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)
    
    rot_reward = torch.sum(end_effector_rot* goal[:,3:], dim=-1) 
    actions = actions.squeeze(-1)
    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward - action_penalty_scale * action_penalty

    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    return rewards, reset_buf



