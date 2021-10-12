import numpy as np
import os
import torch
import imageio

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class KinovaCamera(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, enable_camera_sensors=False):
        # create directory for saved images
        self.img_dir = "kinova_camera_images"
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        # todo read the camera parmas from config file
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        num_obs = 19 # todo 
        num_acts = 7

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg, enable_camera_sensors=enable_camera_sensors)

        # get gym GPU state tensors
        # todo do we need all of these three?
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) # The shape of this tensor is (num_actors, 13)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) # The shape of the tensor is (num_dofs, 2).
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) # The shape of the rigid body state tensor is (num_rigid_bodies, 13)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.kinova_default_dof_pos = to_torch([1.0, 1.0, -1.0, 2.0, -2.0, 0.0, 1.0], device=self.device) #todo
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.kinova_dof_state = self.dof_state.view(self.num_envs, -1, 2)[: , :self.num_kinova_dofs]
        self.kinova_dof_pos = self.kinova_dof_state[..., 0]
        self.kinova_dof_vel = self.kinova_dof_state[..., 1]
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_kinova_dofs:]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.kinova_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset(torch.arange(self.num_envs, device=self.device))

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
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            kinova_asset_file = self.cfg["env"]["asset"].get("assetFileNameKinova", kinova_asset_file)
            cabinet_asset_file = self.cfg["env"]["asset"].get("assetFileNameCabinet", cabinet_asset_file)

         # load kinova asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        kinova_asset = self.gym.load_asset(self.sim, asset_root, kinova_asset_file, asset_options)

         # load cabinet asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)

        kinova_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        kinova_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80], dtype=torch.float, device=self.device)

        self.num_kinova_bodies = self.gym.get_asset_rigid_body_count(kinova_asset)
        self.num_kinova_dofs = self.gym.get_asset_dof_count(kinova_asset)
        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        print("num kinova bodies: ", self.num_kinova_bodies)
        print("num kinova dofs: ", self.num_kinova_dofs)
        print("num cabinet bodies: ", self.num_cabinet_bodies)
        print("num cabinet dofs: ", self.num_cabinet_dofs)

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
        # todo
        # self.kinova_dof_speed_scales[[5, 6]] = 0.1
        # kinova_dof_props['effort'][5] = 9
        # kinova_dof_props['effort'][6] = 9

        # set cabinet dof properties
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        for i in range(self.num_cabinet_dofs):
            cabinet_dof_props['damping'][i] = 10.0

        kinova_start_pose = gymapi.Transform()
        kinova_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        kinova_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        # compute aggregate size
        num_kinova_bodies = self.gym.get_asset_rigid_body_count(kinova_asset)
        num_kinova_shapes = self.gym.get_asset_rigid_shape_count(kinova_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        max_agg_bodies = num_kinova_bodies + num_cabinet_bodies 
        max_agg_shapes = num_kinova_shapes + num_cabinet_shapes 

        # define camera properties
        cam_props = gymapi.CameraProperties()
        cam_props.width = 256
        cam_props.height = 256
        cam_props.enable_tensors = True

        self.kinovas = []
        self.sensors = []
        self.sensors_tensors = []
        self.cabinets = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            kinova_actor = self.gym.create_actor(env_ptr, kinova_asset, kinova_start_pose, "kinova", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, kinova_actor, kinova_dof_props)

            # add sensor
            sensor_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
            camera_offset = gymapi.Vec3(0, -0.1, -0.1)
            camera_rotation = gymapi.Quat.from_euler_zyx(0.78,1.57,0) #gymapi.Quat(0.0, 0.0, 1.0, 0.0) 
            kinova_bracklet_handle = self.gym.find_actor_rigid_body_handle(env_ptr, kinova_actor, "Bracelet_Link")
            self.gym.attach_camera_to_body(sensor_handle, env_ptr, kinova_bracklet_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            self.sensors.append(sensor_handle)

            #obtain sensor tensor
            sensor_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, sensor_handle, gymapi.IMAGE_COLOR)
            print("Got sensor tensor with shape", sensor_tensor.shape)

            # wrap sensor tensor in a pytorch tensor
            torch_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor)
            self.sensors_tensors.append(torch_sensor_tensor)
            print("  Torch sensor tensor device:", torch_sensor_tensor.device)
            print("  Torch sensor tensor shape:", torch_sensor_tensor.shape)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            
            cabinet_pose = cabinet_start_pose
            cabinet_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cabinet_pose.p.y += self.start_position_noise * dy
            cabinet_pose.p.z += self.start_position_noise * dz
            cabinet_actor = self.gym.create_actor(env_ptr, cabinet_asset, cabinet_pose, "cabinet", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cabinet_actor, cabinet_dof_props)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.kinovas.append(kinova_actor)
            self.cabinets.append(cabinet_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, kinova_actor, "Bracelet_Link")
        self.drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
        self.init_data()

    def init_data(self):
        
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.kinovas[0], "Bracelet_Link")
        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        
        # fake finger pose
        finger_pose = gymapi.Transform() 
        finger_pose.p = hand_pose.p + gymapi.Vec3(0.0, 0.0, 0.15)
        finger_pose.r = hand_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        kinova_local_grasp_pos = hand_pose_inv * finger_pose
        
        kinova_local_grasp_pos.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.kinova_local_grasp_pos = to_torch([kinova_local_grasp_pos.p.x, kinova_local_grasp_pos.p.y,
                                                kinova_local_grasp_pos.p.z], device=self.device).repeat((self.num_envs, 1))
        self.kinova_local_grasp_rot = to_torch([kinova_local_grasp_pos.r.x, kinova_local_grasp_pos.r.y,
                                                kinova_local_grasp_pos.r.z, kinova_local_grasp_pos.r.w], device=self.device).repeat((self.num_envs, 1))
        drawer_local_grasp_pose = gymapi.Transform()
        drawer_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.drawer_local_grasp_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
                                                drawer_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
                                                drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.drawer_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.drawer_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.kinova_grasp_pos = torch.zeros_like(self.kinova_local_grasp_pos)
        self.kinova_grasp_rot = torch.zeros_like(self.kinova_local_grasp_rot)
        self.kinova_grasp_rot[..., -1] = 1  # xyzw
        self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
        self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
        self.drawer_grasp_rot[..., -1] = 1
    
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_kinova_reward(
            self.reset_buf, self.progress_buf, self.actions, self.cabinet_dof_pos,
            self.kinova_grasp_pos, self.drawer_grasp_pos, self.kinova_grasp_rot, self.drawer_grasp_rot,
            self.gripper_forward_axis, self.drawer_inward_axis, self.gripper_up_axis, self.drawer_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # render sensors and refresh camera tensors
        frame_no = self.gym.get_frame_count(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        drawer_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]

        self.kinova_grasp_rot[:], self.kinova_grasp_pos[:], self.drawer_grasp_rot[:], self.drawer_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.kinova_local_grasp_rot, self.kinova_local_grasp_pos,
                                     drawer_rot, drawer_pos, self.drawer_local_grasp_rot, self.drawer_local_grasp_pos
                                     )

        dof_pos_scaled = (2.0 * (self.kinova_dof_pos - self.kinova_dof_lower_limits)
                          / (self.kinova_dof_upper_limits - self.kinova_dof_lower_limits) - 1.0)
        to_target = self.drawer_grasp_pos - self.kinova_grasp_pos

        self.obs_buf = torch.cat((dof_pos_scaled, self.kinova_dof_vel * self.dof_vel_scale, to_target,
                                  self.cabinet_dof_pos[:, 3].unsqueeze(-1), self.cabinet_dof_vel[:, 3].unsqueeze(-1)), dim=-1)
        
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
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset kinova
        pos = tensor_clamp(
            self.kinova_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_kinova_dofs), device=self.device) - 0.5),
            self.kinova_dof_lower_limits, self.kinova_dof_upper_limits)
        self.kinova_dof_pos[env_ids, :] = pos
        self.kinova_dof_vel[env_ids, :] = torch.zeros_like(self.kinova_dof_vel[env_ids])
        self.kinova_dof_targets[env_ids, :self.num_kinova_dofs] = pos

        # reset cabinet
        self.cabinet_dof_state[env_ids, :] = torch.zeros_like(self.cabinet_dof_state[env_ids])

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.kinova_dof_targets),
                                                    gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.kinova_dof_targets[:, :self.num_kinova_dofs] + self.kinova_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.kinova_dof_targets[:, :self.num_kinova_dofs] = tensor_clamp(
            targets, self.kinova_dof_lower_limits, self.kinova_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        # self.gym.set_dof_position_target_tensor(self.sim,
        #                                         gymtorch.unwrap_tensor(self.kinova_dof_targets))

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
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.kinova_grasp_pos[i] + quat_apply(self.kinova_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.kinova_grasp_pos[i] + quat_apply(self.kinova_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.kinova_grasp_pos[i] + quat_apply(self.kinova_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.kinova_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.drawer_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])



@torch.jit.script
def compute_kinova_reward(
    reset_buf, progress_buf, actions, cabinet_dof_pos,
    kinova_grasp_pos, drawer_grasp_pos, kinova_grasp_rot, drawer_grasp_rot,
    gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from hand to the drawer
    d = torch.norm(kinova_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(kinova_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
    axis3 = tf_vector(kinova_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward - action_penalty_scale * action_penalty
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    return rewards, reset_buf



@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, kinova_local_grasp_rot, kinova_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_kinova_rot, global_kinova_pos = tf_combine(
        hand_rot, hand_pos, kinova_local_grasp_rot, kinova_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_kinova_rot, global_kinova_pos, global_drawer_rot, global_drawer_pos
