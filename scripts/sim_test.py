import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()
        
# Configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 1
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu 
else:
    raise Exception("This example can only be used with PhysX")
    
#sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()
    


# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()
    
    
# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Load my Robot Asset
asset_root = "../assets"
robot_asset_file = "urdf/hyper.urdf"
    
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

# Mash를 Z-up left handed system에서 Y-up Right-Handed coordinate system으로 바꿈
asset_options.flip_visual_attachments = True  
asset_options.armature = 0.01

# asset에 대한 중력 무시하도록 설정
asset_options.disable_gravity = True


print("Loading Asset '%s' from '%s'" %(robot_asset_file, asset_root))
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, asset_options)

# Get joint limits and ranges for Robot
robot_dof_properties = gym.get_asset_dof_properties(robot_asset)
robot_lower_limits = robot_dof_properties['lower']
robot_upper_limits = robot_dof_properties['upper']
robot_ranges = robot_lower_limits - robot_upper_limits
robot_mids = 0.5*(robot_lower_limits + robot_upper_limits)
robot_num_dofs = len(robot_dof_properties)

# Set up the environment
# set up the env grid
num_envs = 64
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    height = 1.0

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, height, 0.0)

    actor_handle = gym.create_actor(env, robot_asset, pose, "MyActor", i, 1)
    actor_handles.append(actor_handle)
    
    
while True:
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    while not gym.query_viewer_has_closed(viewer):
    	# step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    	# update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

    	# Wait for dt to elapse in real time.
    	# This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
