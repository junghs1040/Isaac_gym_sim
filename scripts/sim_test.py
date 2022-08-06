
# 1 joint manipulator control test

import math
from isaacgym import gymapi
from isaacgym import gymutil

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description = "Manipulator Position Control Test")
        
# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0
#sim_params.up_axis = gymapi.UP_AXIS_Z

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    
if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 4
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)


# Load my Robot Asset
asset_root = "../assets"
asset_file = "urdf/hyper.urdf"
    
# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)    

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.5, 0.0)
initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole0 = gym.create_actor(env0, cartpole_asset, initial_pose, 'cartpole', 0, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cartpole0)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
props["stiffness"] = (5000.0, 5000.0, 5000.0,5000.0, 5000.0, 5000.0,5000.0, 5000.0, 5000.0,5000.0, 5000.0, 5000.0)
props["damping"] = (100.0, 100.0, 100.0,100.0, 100.0, 100.0,100.0, 100.0, 100.0,100.0, 100.0, 100.0)
gym.set_actor_dof_properties(env0, cartpole0, props)
# Set DOF drive targets
dof_handle1 = gym.find_actor_dof_handle(env0, cartpole0, 'LF_joint1')
dof_handle2 = gym.find_actor_dof_handle(env0, cartpole0, 'LF_joint2')
dof_handle3 = gym.find_actor_dof_handle(env0, cartpole0, 'LF_joint3')
dof_handle4 = gym.find_actor_dof_handle(env0, cartpole0, 'LB_joint1')
dof_handle5 = gym.find_actor_dof_handle(env0, cartpole0, 'LB_joint2')
dof_handle6 = gym.find_actor_dof_handle(env0, cartpole0, 'LB_joint3')
dof_handle7 = gym.find_actor_dof_handle(env0, cartpole0, 'RF_joint1')
dof_handle8 = gym.find_actor_dof_handle(env0, cartpole0, 'RF_joint2')
dof_handle9 = gym.find_actor_dof_handle(env0, cartpole0, 'RF_joint3')
dof_handle10 = gym.find_actor_dof_handle(env0, cartpole0, 'RB_joint1')
dof_handle11 = gym.find_actor_dof_handle(env0, cartpole0, 'RB_joint2')
dof_handle12 = gym.find_actor_dof_handle(env0, cartpole0, 'RB_joint3')
gym.set_dof_target_position(env0, dof_handle1, 0)
gym.set_dof_target_position(env0, dof_handle2, 0.392)
gym.set_dof_target_position(env0, dof_handle3, -0.35)
gym.set_dof_target_position(env0, dof_handle4, 0)
gym.set_dof_target_position(env0, dof_handle5, 0.392)
gym.set_dof_target_position(env0, dof_handle6, -0.35)
gym.set_dof_target_position(env0, dof_handle7, 0)
gym.set_dof_target_position(env0, dof_handle8, 0.392)
gym.set_dof_target_position(env0, dof_handle9, -0.35)
gym.set_dof_target_position(env0, dof_handle10, 0)
gym.set_dof_target_position(env0, dof_handle11, 0.392)
gym.set_dof_target_position(env0, dof_handle12, -0.35)



# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Nothing to be done for env 0

    # Nothing to be done for env 1
    
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)    
