from builtins import print
import numpy as np
import os
import torch
import math

from skimage import metrics

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
    print("q_r:", q_r)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

class AssetDesc:
    def __init__(self, file_name, asset_name, flip_visual_attachments=False, mesh_normal_mode=gymapi.FROM_ASSET):
        self.file_name = file_name
        self.asset_name = asset_name
        self.flip_visual_attachments = flip_visual_attachments
        self.mesh_normal_mode = mesh_normal_mode


asset_descriptors = [
AssetDesc("urdf/objects/cube_multicolor.urdf","cube", False)
]

class KinovaCameraIKEnv(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, enable_camera_sensors=False):
        # todo it is temporary create directory for saved images
        self.img_dir = "kinova_camera_images"
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.parse_config(cfg, sim_params, physics_engine, device_type, device_id, headless)

        # the base class will call a few functin before returning the program here
        super().__init__(cfg=self.cfg, enable_camera_sensors=enable_camera_sensors)

        self.set_tensors()
        self.respawn_rand_obj(torch.arange(self.num_envs, device=self.device))
        self.reach_target(torch.arange(self.num_envs, device=self.device))        
    
    def parse_config(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        """ initialize the some cfg parameters based on cfg file

        Args:
            cfg ([type]): [description]
            sim_params ([type]): [description]
            physics_engine ([type]): [description]
            device_type ([type]): [description]
            device_id ([type]): [description]
            headless ([type]): [description]
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.similarity_thr = self.cfg["env"]["similarityThreshold"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        
        self.camera_width = self.cfg["env"]["camera"]["width"]
        self.camera_height = self.cfg["env"]["camera"]["height"]
        
        self.up_axis = "z"
        self.up_axis_idx = 2
        self.dt = 1/60.

        self.cfg["env"]["numObservations"] = (2,self.camera_width, self.camera_height,4)
        self.cfg["env"]["numActions"] = 6 # our action is 3 pose and 3 orientation error (look at the kinova_cube_it orientation error)
        self.cfg["env"]["numStates"] = 13 # Dof pos (7) + to_target (3pos+3ori)
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
    
        self.asset_root = self.cfg["env"]["asset"].get("assetRoot")
        self.kinova_asset_file = self.cfg["env"]["asset"].get("assetFileNameKinova")

        
    def set_tensors(self):
        """creates, intilized and slice some usefull tensors
        """
        # get gym GPU dof tensors and create some wrapper and slices
        # Retrieves Degree-of-Freedom state buffer. Buffer has shape (num_environments* num_dofs , 2). Each DOF state contains position and velocity.
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # contains the kinova+table+box dofs

        self.kinova_dof_state = self.dof_state.view(self.num_envs, -1, 2)[: , :self.num_kinova_dofs] # just get the kinova dofs
        self.kinova_dof_pos = self.kinova_dof_state[..., 0]
        self.kinova_dof_vel = self.kinova_dof_state[..., 1]
        self.rand_obj_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_kinova_dofs:]

        
        # get gym GPU rigid body state tensors and create some wrappers
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) # The shape of the rigid body state tensor is (num_envs, num_rigid_bodies* 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13) # we need it to get the end efffector pose & orientation
        self.end_effector_pos = self.rigid_body_states[:, self.endefector_handle][:, 0:3]
        self.end_effector_rot = self.rigid_body_states[:, self.endefector_handle][:, 3:7]
        #self.num_bodies = self.rigid_body_states.shape[1]
        
        # necessary for reseting random objects
        # Retrieves buffer for Actor root states. Buffer has shape (num_environments, num_actors * 13). State for each actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13) # would be (self.num_env, kinova+rand obj, 13)
        self.rand_obj_states = self.root_state_tensor[:, 1:] 

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
        self.gym.refresh_actor_root_state_tensor(self.sim)

        #self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.kinova_home = to_torch([0.0, -1.0, 0.0, +2.6, -1.57, 0.0, 0.0], device=self.device) # pos of each joint
        self.home_joints = torch.zeros((self.num_envs, self.num_kinova_dofs), dtype=torch.float, device=self.device) # joints pose at home
        self.target_joints = torch.zeros((self.num_envs, self.num_kinova_dofs), dtype=torch.float, device=self.device) # joints pose at target
        # whenever you wana move kinova to a target you have to fill in this tensor and call set_dof_position_target
        self.kinova_target_dof_pos = torch.zeros((self.num_envs, self.num_kinova_dofs), dtype=torch.float, device=self.device)
        
        # 3 if manip is moved to a random conf, the goal image is taken, and the returned to home pose and home image is taken
        # 2 if moved to goal but not homed
        # 1 if rnd object is respawned
        # 0 if neither moved to goal nor homed nor rnd object is respawned
        self.reset_complete = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device) 
        self.goal_obs = torch.zeros((self.num_envs,1, self.camera_width, self.camera_height,4 ), device=self.device, dtype=torch.float)
        self.current_obs = torch.zeros((self.num_envs, 1, self.camera_width, self.camera_height,4 ), device=self.device, dtype=torch.float)
        self.valid = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device) # if 1 the outputs of step() are valid if zero are not valid

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs( self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        """creates the ground plane of simulator
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def load_rand_assets(self):
        """loads some random assets that will be used as targets
        """
        rand_assets = []
        for asset_desc in asset_descriptors:
                asset_file = asset_desc.file_name
                asset_options = gymapi.AssetOptions()
                asset_options.fix_base_link = True
                asset_options.flip_visual_attachments = asset_desc.flip_visual_attachments
                asset_options.mesh_normal_mode = asset_desc.mesh_normal_mode
                print("Loading asset '%s' from '%s'" % (asset_file, self.asset_root))
                rand_assets.append(self.gym.load_asset(self.sim, self.asset_root, asset_file, asset_options))
        return rand_assets

    def load_kinova_asset(self):
        """loads kinova assets and obtains some info based on asset

        Args:
            asset_root ([type]): [description]
            kinova_asset_file ([type]): [description]

        Returns:
            [type]: [description]
        """
        # load kinova asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        #asset_options.collapse_fixed_joints = True # will colapse the end effector link
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        kinova_asset = self.gym.load_asset(self.sim, self.asset_root, self.kinova_asset_file, asset_options)

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

    def _create_envs(self, spacing, num_per_row):
        """creates environments

        Args:
            spacing ([type]): [description]
            num_per_row ([type]): [description]
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.rand_assets = self.load_rand_assets()
        kinova_asset, kinova_dof_props = self.load_kinova_asset()
        
        kinova_start_pose = gymapi.Transform()
        kinova_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        kinova_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        # define camera properties
        cam_props = gymapi.CameraProperties()
        cam_props.width = self.camera_width
        cam_props.height = self.camera_height
        cam_props.enable_tensors = True

        self.cam_tensors = []
        self.envs = []
        self.kinova_indices = []
        self.obj_indices = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            kinova_actor = self.gym.create_actor(env_ptr, kinova_asset, kinova_start_pose, "kinova", i, 2)
            self.gym.set_actor_dof_properties(env_ptr, kinova_actor, kinova_dof_props)
            kinove_index = self.gym.get_actor_index(env_ptr,kinova_actor,gymapi.DOMAIN_SIM)
            self.kinova_indices.append(kinove_index)

            # add sensor
            sensor_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
            camera_offset = gymapi.Vec3(0, -0.1, -0.1)
            camera_rotation = gymapi.Quat.from_euler_zyx(math.pi/2,math.pi/2,0) 
            kinova_bracelet_handle = self.gym.find_actor_rigid_body_handle(env_ptr, kinova_actor,  "Bracelet_Link")
            self.gym.attach_camera_to_body(sensor_handle, env_ptr, kinova_bracelet_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            
            # obtain sensor tensor
            sensor_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, sensor_handle, gymapi.IMAGE_COLOR)
            print("Got sensor tensor with shape", sensor_tensor.shape)
            # wrap sensor tensor in a pytorch tensor
            torch_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor)
            self.cam_tensors.append(torch_sensor_tensor.unsqueeze(0))
            print("  Torch sensor tensor device:", torch_sensor_tensor.device)
            print("  Torch sensor tensor shape:", torch_sensor_tensor.shape)

            # # add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(np.random.rand()-2, np.random.rand()-0.5, 0.5)
            u = np.random.rand()
            v = np.random.rand()
            w = np.random.rand()
            pose.r = gymapi.Quat(math.sqrt(1-u)*math.sin(2*math.pi*v), math.sqrt(1-u)*math.cos(2*math.pi*v), math.sqrt(u)*math.sin(2*math.pi*w), math.sqrt(u)*math.cos(2*math.pi*w))
            object_actor = self.gym.create_actor(env_ptr, self.rand_assets[np.random.randint(len(self.rand_assets))], pose, "random_asset"+str(i), i,2)
            obj_index = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)
            self.obj_indices.append(obj_index)

            self.envs.append(env_ptr)

        self.kinova_indices = to_torch(self.kinova_indices, dtype=torch.long, device=self.device)
        self.obj_indices = to_torch(self.obj_indices, dtype=torch.long, device=self.device)
        self.endefector_handle = self.gym.find_actor_rigid_body_handle(env_ptr, kinova_actor,  "EndEffector_Link") 

    def compute_reward(self):
    
        similarities = []
        for i in range(self.num_envs):
            similarity = metrics.structural_similarity(self.obs_buf[i][1].cpu().numpy(), self.obs_buf[i][0].cpu().numpy(), multichannel=True)
            similarities.append(similarity)

        similarities_tensor = to_torch(similarities, device=self.device, dtype=torch.float)
        self.rew_buf = torch.where(similarities_tensor>=self.similarity_thr, torch.ones_like(self.rew_buf), torch.zeros_like(self.rew_buf) )
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        

    def compute_observations(self):
        """sets the obs_buff based on goal and current observations

        Returns:
            [tensor, tensor]: [observation buffer]
        """
        # render sensors and refresh camera tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        frame_no = self.gym.get_frame_count(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        reset_complete_2 = (self.reset_complete==2).nonzero(as_tuple=False).squeeze(-1) # already reached target
        reset_complete_3 = (self.reset_complete==3).nonzero(as_tuple=False).squeeze(-1)
        for i in reset_complete_2:
            self.goal_obs[i] = self.cam_tensors[i] # self.reset_complete==1 means already is reached to target but not hommed yet
        for i in reset_complete_3:
            self.current_obs[i] = self.cam_tensors[i]
        
        torch.cat((self.goal_obs, self.current_obs), 1 , out=self.obs_buf)
        if  frame_no%100 <=5 :
            for i in range(1):
                i=0
                fname = os.path.join(self.img_dir, "goal-%04d-%04d.png" % (frame_no, i))
                sensor_img = torch.squeeze(self.obs_buf[i][0]).cpu().numpy()
                imageio.imwrite(fname, sensor_img)
                
                fname = os.path.join(self.img_dir, "current-%04d-%04d.png" % (frame_no, i))
                sensor_img = torch.squeeze(self.obs_buf[i][1]).cpu().numpy()
                imageio.imwrite(fname, sensor_img)
        print("pose: ", self.end_effector_pos)
        print("ori: ", self.end_effector_rot)
        self.gym.end_access_image_tensors(self.sim)
        return self.obs_buf, self.states_buf #todo state buf is not changed

    def reach_home(self, env_ids):
        """sets the kinova target to home
        Note: it wont reach home until simulate step being called at base class 

        Args:
            env_ids ([list]): [index of envs that their kinova need to be homed]
        """
        kinova_indices = self.kinova_indices[env_ids].to(torch.int32)
        if(len(kinova_indices)>0):
            # pos = tensor_clamp(
            #     self.kinova_home.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_kinova_dofs), device=self.device) - 0.5),
            #     self.kinova_dof_lower_limits, self.kinova_dof_upper_limits)
            pos = self.kinova_home.unsqueeze(0)
            self.kinova_dof_pos[env_ids, :] = pos
            self.kinova_dof_vel[env_ids, :] = torch.zeros_like(self.kinova_dof_vel[env_ids])
            self.kinova_target_dof_pos[env_ids, :self.num_kinova_dofs] = pos
            self.home_joints[env_ids, :] = pos

            # todo why set both pose and state (state itself has pos and vel)?
            # When simulate is called, the actor joints will move based on their joint constraints and the effort DOF parameter to the target positions.
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.kinova_target_dof_pos),
                                                        gymtorch.unwrap_tensor(kinova_indices), len(kinova_indices))

            # Sets DOF state buffer to values provided for all DOFs in simulation. DOF state includes position in meters for prismatic DOF, or radians for revolute DOF, and velocity in m/s for prismatic DOF and rad/s for revolute DOF
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(kinova_indices), len(kinova_indices))



    def reach_target(self, env_ids):
        """generated some random dof (as a goal) for kinovas[env_ids] and sets the target to reach there 
            Note: it wont reach target until simulate step being called at base class 
            Note: if I just call set_dof_position_target_tensor_indexed if will reach target after a few steps but when calling set_dof_state_tensor_indexed together with that it reaches there immedietly
        Args:
            env_ids ([list]): [index of envs that their kinova need to be homed]
        """
        kinova_indices = self.kinova_indices[env_ids].to(torch.int32)

        if(len(kinova_indices)>0):            
           
            # set target for Kinova
            pos = tensor_clamp(
                self.kinova_home.unsqueeze(0) + 0.50 * (torch.rand((len(env_ids), self.num_kinova_dofs), device=self.device) - 0.25),
                self.kinova_dof_lower_limits, self.kinova_dof_upper_limits)
            self.kinova_dof_pos[env_ids, :] = pos
            self.kinova_dof_vel[env_ids, :] = torch.zeros_like(self.kinova_dof_vel[env_ids])
            self.kinova_target_dof_pos[env_ids, :self.num_kinova_dofs] = pos
            self.target_joints[env_ids, :] = pos
            # todo why set both pose and state (state itself has pos and vel)?
            # Sets DOF position targets to values provided for all DOFs in simulation. For presimatic DOF, target is in meters. For revolute DOF, target is in radians.
            # When simulate is called, the actor joints will move based on their joint constraints and the effort DOF parameter to the target positions.
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.kinova_target_dof_pos),
                                                            gymtorch.unwrap_tensor(kinova_indices), len(kinova_indices))
            #Sets DOF state buffer to values provided for all DOFs in simulation. DOF state includes position in meters for prismatic DOF, or radians for revolute DOF, and velocity in m/s for prismatic DOF and rad/s for revolute DOF
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(kinova_indices), len(kinova_indices))

    def respawn_rand_obj(self, env_ids):
        rand_obj_indices = self.kinova_indices[env_ids].to(torch.int32) + 1 # assuming each env has 1 kinova and 1 rand object
        if(len(rand_obj_indices)>0):
         # re-spawn random objects
            self.rand_obj_dof_state[env_ids, :] = torch.zeros_like(self.rand_obj_dof_state[env_ids])
            rand_poses = []
            for _ in range(self.num_envs):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(np.random.rand()-2, np.random.rand()-0.5, 0.5)
                u = np.random.rand()
                v = np.random.rand()
                w = np.random.rand()
                pose.r = gymapi.Quat(math.sqrt(1-u)*math.sin(2*math.pi*v), math.sqrt(1-u)*math.cos(2*math.pi*v), math.sqrt(u)*math.sin(2*math.pi*w), math.sqrt(u)*math.cos(2*math.pi*w))
                rand_poses.append([pose.p.x, pose.p.y, pose.p.z, pose.r.x, pose.r.y, pose.r.z, pose.r.w, 0.0, 0.0, 0.0 ,0.0 , 0.0 , 0.0 ])
            rand_poses_tensor= to_torch(rand_poses, device=self.device, dtype=torch.float).view(self.num_envs, 1, 13) # assuming just one random obj in each
            self.rand_obj_states[env_ids] = rand_poses_tensor[:]
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(rand_obj_indices), len(rand_obj_indices))

    def reset(self, env_ids):
        """
            reset kinova:
            1- this is the most complex function
            2- when a gym scenario resets we have to set a new goal and set the kinova to be at home
            3- to set the goal the kinova and the object has to MOVE to a random conf and kinova has to capture a rgb as goal and then get back to the home and take another rgb as current obs
            4- step 3 will happen in TWO CALLS TO THE STEP() AT BASE CLASS       

        Args:
            env_ids ([list]): [index of envs that their kinova need to be reset]
        """
        self.respawn_rand_obj((self.reset_complete==0).nonzero(as_tuple=False).squeeze(-1))
        self.reach_target((self.reset_complete==1).nonzero(as_tuple=False).squeeze(-1))
        self.reach_home((self.reset_complete==2).nonzero(as_tuple=False).squeeze(-1))
       
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.obs_buf[env_ids] = torch.zeros(*self.num_obs, device=self.device, dtype=torch.float32)
        self.states_buf[env_ids] = torch.zeros(( self.num_states), device=self.device, dtype=torch.float32)

    def pre_physics_step(self, desired):
        """
        takes the actions as input (actions are end effector pose not in join state) so an IK will be solved here and then set the target based on IK results

        Args:
            actions ([tensor]): [actions to be done]
        """
        kinova_indices = self.kinova_indices[(self.valid==1).nonzero(as_tuple=False).squeeze(-1)].to(torch.int32) # action will just be execute for those that are valid (are not in reset scenario)
        if(len(kinova_indices)>0):
            pos_err = desired[:, :3] - self.end_effector_pos
            orn_err = orientation_error( desired[:, 3:] , self.end_effector_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            self.gym.refresh_jacobian_tensors(self.sim)
            
            # solve damped least squares
            # https://www.daslhub.org/unlv/wiki/doku.php?id=robotic_manipulators_ik
            # https://www.tandfonline.com/doi/full/10.1080/01691864.2020.1780151 look at section 2
            j_eef_T = torch.transpose(self.j_eef, 1, 2)
            d = 0.05  # damping term
            lmbda = torch.eye(6).to(self.device) * (d ** 2)
            u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7, 1)
            # update position targets
            dof_pos = self.kinova_dof_pos.view(self.num_envs, self.num_kinova_dofs, 1)
            targets =   u.squeeze(-1)  + dof_pos.squeeze(-1) 
            # if reset is not complete dof targets are set either in reach_target ot reach_home
            self.kinova_target_dof_pos[(self.valid==1).nonzero(as_tuple=False).squeeze(-1), :self.num_kinova_dofs] = targets[(self.valid==1).nonzero(as_tuple=False).squeeze(-1), :self.num_kinova_dofs]
            
            
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.kinova_target_dof_pos),
                                                            gymtorch.unwrap_tensor(kinova_indices), len(kinova_indices))
            

    def post_physics_step(self):
        
        self.progress_buf += 1
        self.reset_buf[(self.progress_buf==self.max_episode_length-1).nonzero(as_tuple=False).squeeze(-1)]=1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0 or len((self.reset_complete!=3).nonzero(as_tuple=False).squeeze(-1)) > 0: # some already running scenarios need reset OR we already in the reset scenario
            self.reset_complete[env_ids] = 0 # 0 means neither moved to goal nor homed
            self.reset(env_ids)

        self.valid[(self.reset_complete==3).nonzero(as_tuple=False).squeeze(-1)]=1
        self.valid[(self.reset_complete!=3).nonzero(as_tuple=False).squeeze(-1)]=0
        self.compute_observations()
        self.compute_reward()

        self.reset_complete[(self.reset_complete==2).nonzero(as_tuple=False).squeeze(-1)] = 3
        self.reset_complete[(self.reset_complete==1).nonzero(as_tuple=False).squeeze(-1)] = 2
        self.reset_complete[(self.reset_complete==0).nonzero(as_tuple=False).squeeze(-1)] = 1 



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
    # to do is being reset just after max ep len? what about done?
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    return rewards, reset_buf



