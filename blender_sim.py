"""
conveyor belt similation script
to run: blender --background --python blender_sim.py
"""

import bpy
import math
import random
import json
import os
import sys
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
	sys.path.insert(0, script_dir)

from config import SimConfig
CONFIG = SimConfig()
CONFIG.ensure_dirs()
random.seed(CONFIG.seed)

def clear_scene():
	"""
	remove all objects and orphan data from enviornment.
	"""
	bpy.ops.object.select_all(action='SELECT')
	bpy.ops.object.delete()

	for block in bpy.data.meshes:
		if block.users == 0:
			bpy.data.meshes.remove(block)
	for block in bpy.data.materials:
		if block.users == 0:
			bpy.data.materials.remove(block)
	for block in bpy.data.cameras:
		if block.users == 0:
			bpy.data.cameras.remove(block)

def create_material(name, color, emission_strength = 1.0):
	"""
	create emission material for clean color segmentation with no shading for consistent coloring
	"""
	mat = bpy.data.materials.new(name=name)
	mat.use_nodes = True
	nodes = mat.node_tree.nodes
	links = mat.node_tree.links

	nodes.clear()

	emission = nodes.new('ShaderNodeEmission')
	emission.inputs['Color'].default_value = color
	emission.inputs['Strength'].default_value = emission_strength

	output = nodes.new('ShaderNodeOutputMaterial')
	links.new(emission.outputs['Emission'], output.inputs['Surface'])

	return mat

def poisson_sample(rate, dt):
	"""
	sample from poisson distribution for spawn counts in time interval = dt
	"""
	lam = rate * dt
	if lam <= 0:
		return 0
	elif lam < 30:
		L = math.exp(-lam)
		k = 0
		p = 1.0
		while p > L:
			k += 1
			p *= random.random()

		return k - 1
	else:
		return max(0, int(round(random.gauss(lam, math.sqrt(lam)))))
	
# --------------------------
# CREATE DAT SCENERY  
# --------------------------

def create_conveyor_belt():
	""" create the conveyor belt plane """
	bpy.ops.mesh.primitive_plane_add(
		size=1,
		location=(CONFIG.belt_length / 2, 0, 0)
	)

	belt = bpy.context.active_object
	belt.name = "ConveyorBelt"
	belt.scale = (CONFIG.belt_length, CONFIG.belt_width, 1)
	bpy.ops.object.transform_apply(scale=True)

	mat = create_material("BeltMaterial", CONFIG.belt_color, emission_strength=0.3)
	belt.data.materials.append(mat)

	return belt

def setup_camera():
	""" create / position camera at a specified angle """
	look_at = Vector((CONFIG.belt_length / 2, 0, 0))

	angle_rad = math.radians(CONFIG.camera_angle_deg)
	cam_x = look_at.x - CONFIG.camera_orbit_radius * math.cos(angle_rad)
	cam_z = CONFIG.camera_orbit_radius * math.sin(angle_rad)

	bpy.ops.object.camera_add(location=(cam_x, 0, cam_z))
	camera = bpy.context.active_object
	camera.name = "MainCamera"

	direction = look_at - camera.location
	rot_quat = direction.to_track_quat('-Z', 'Y')
	camera.rotation_euler = rot_quat.to_euler()
	camera.data.angle = math.radians(CONFIG.camera_fov_deg)
	camera.data.sensor_width = 36
	camera.data.sensor_height = 24
	bpy.context.scene.camera = camera
	
	return camera

def setup_lighting():
	""" basic dark background for contrast """
	world = bpy.context.scene.world
	if world is None:
		world = bpy.data.worlds.new("World")
		bpy.context.scene.world = world

	world.use_nodes = True
	nodes = world.node_tree.nodes
	nodes.clear()

	bg = nodes.new('ShaderNodeBackground')
	bg.inputs['Color'].default_value = (0.05, 0.05, 0.08, 1.0)
	bg.inputs['Strength'].default_value = 0.5
	
	output = nodes.new('ShaderNodeOutputWorld')
	world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

def setup_render_settings():
	""" configuring render settings for fast output? like """
	scene = bpy.context.scene

	scene.render.resolution_x = CONFIG.camera_resolution[0]
	scene.render.resolution_y = CONFIG.camera_resolution[1]
	scene.render.resolution_percentage = 100

	scene.render.engine = 'BLENDER_EEVEE'

	scene.render.image_settings.file_format = 'PNG'
	scene.render.image_settings.color_mode = 'RGB'
	scene.render.image_settings.compression = 50

	scene.render.fps = CONFIG.fps
	scene.frame_start = 1
	scene.frame_end = CONFIG.total_frames
	scene.render.use_motion_blur = False

# --------------------------
# CUBE MANAGEMENT 
# --------------------------

class CubeManager:
	"""
	manages cube spawning and movement. they move at their assigned speeds with flexible overlap of up to 50% to simulate object clustering.
	"""
	def __init__(self):
		self.cubes = {}
		self.cube_data = {}
		self.next_cube_id = 0

		self.cube_material = create_material("CubeMaterial", CONFIG.cube_color, emission_strength=1.0)
		self.noise_material = create_material("NoiseMaterial", CONFIG.noise_color, emission_strength=1.0)

	def spawn_cube(self, is_noise = False):
		"""
		cube spawns in. x->start of belt, y->random within belt width, z->belt surface
		"""
		cube_id = self.next_cube_id
		self.next_cube_id += 1

		if is_noise:
			base = CONFIG.cube_size * CONFIG.noise_size_factor
			variation = random.uniform(-0.1, 0.1)
			size = base * (1 + variation)
		else:
			variation = random.uniform(-CONFIG.cube_size_variation, CONFIG.cube_size_variation)
			size = CONFIG.cube_size * (1 + variation)

		speed_variation = random.uniform(-CONFIG.cube_speed_variation, CONFIG.cube_speed_variation)
		speed = CONFIG.cube_speed * (1 + speed_variation)

		margin = size
		spawn_x = size / 2 + random.uniform(0, 2)
		spawn_z = size / 2

		for _ in range(10):
			spawn_y = random.uniform(-CONFIG.belt_width / 2 + margin, CONFIG.belt_width / 2 - margin)

			valid = True
			for other_id, other_cube in self.cubes.items():
				if not self.cube_data[other_id]['active']:
					continue
				if other_cube is None:
					continue

				try:
					other_x = other_cube.location.x
					other_y = other_cube.location.y
					other_size = self.cube_data[other_id]['size']

					if other_x > 5:
						continue

					y_dist = abs(spawn_y - other_y)
					x_dist = abs(spawn_x - other_x)

					y_overlap = max(0, (size + other_size) / 2 - y_dist)
					x_overlap = max(0, (size + other_size) / 2 - x_dist)

					if y_overlap > 0 and x_overlap > 0:
						overlap_area = y_overlap * x_overlap
						smaller = min(size, other_size)
						max_allowed = 0.5 * smaller * smaller

						if overlap_area > max_allowed:
							valid = False
							break

				except:
					continue

			if valid:
				break

		bpy.ops.mesh.primitive_cube_add(
			size=size,
			location=(spawn_x, spawn_y, spawn_z)
		)
		cube = bpy.context.active_object
		cube.name = f"Cube_{cube_id}"

		cube.rotation_euler = (0, 0, random.uniform(-0.25, 0.25))

		mat = self.noise_material if is_noise else self.cube_material
		cube.data.materials.append(mat)

		self.cubes[cube_id] = cube
		self.cube_data[cube_id] = {
			'is_noise': is_noise,
			'size': size,
			'speed': speed,
			'spawn_frame': bpy.context.scene.frame_current,
			'active': True
		}

		return cube_id
	
	def update_positions(self, frame):
		"""
		move cubes forward on belt while respecting overlap constraints.
		any two cubes can overlap up to 50% of the smaller cube's area.
		if a faster cube would exceed this when catching up, it matches slower cubes speed.
		otherwise, the faster cube may pass.
		purpose is to simulate movement of living objects as accurately as possible.
		"""
		dt = 1.0 / CONFIG.fps

		active = []
		for cube_id in list(self.cubes.keys()):
			if not self.cube_data[cube_id]['active']:
				continue
			cube = self.cubes[cube_id]
			if cube is None:
				continue
			try:
				_ = cube.location
				active.append((cube_id, cube))
			except ReferenceError:
				self.cube_data[cube_id]['active'] = False
				continue

		active.sort(key=lambda x: -x[1].location.x)

		cubes_to_remove = []

		for i, (cube_id, cube) in enumerate(active):
			data = self.cube_data[cube_id]
			size = data['size']
			speed = data['speed']

			new_x = cube.location.x + speed * dt

			for j in range(i):
				other_id, other_cube = active[j]
				other_data = self.cube_data[other_id]
				other_size = other_data['size']

				y_dist = abs(cube.location.y - other_cube.location.y)
				y_overlap = max(0, (size + other_size) / 2 - y_dist)

				if y_overlap == 0:
					continue

				smaller = min(size, other_size)
				max_overlap_area = 0.5 * smaller * smaller
				max_x_overlap = max_overlap_area / y_overlap
				min_x_dist = (size + other_size) / 2 - max_x_overlap
				min_x_dist = max(0, min_x_dist)
				x_dist = other_cube.location.x - new_x

				if x_dist < min_x_dist:
					new_x = other_cube.location.x - min_x_dist

			new_x = max(new_x, cube.location.x)
			cube.location.x = new_x

			if cube.location.x > CONFIG.belt_length + 5:
				cubes_to_remove.append(cube_id)

		for cube_id in cubes_to_remove:
			self.remove_cube(cube_id)

	def remove_cube(self, cube_id):
		""" remove cube from the environment? """
		if cube_id in self.cubes:
			cube = self.cubes[cube_id]
			if cube:
				try:
					bpy.data.objects.remove(cube, do_unlink=True)
				except:
					pass
			del self.cubes[cube_id]
			if cube_id in self.cube_data:
				self.cube_data[cube_id]['active'] = False

	def get_active_cubes(self):
		""" get all active cube Ids and objects """
		return {cid: cube for cid, cube in self.cubes.items()
		  if cube is not None and self.cube_data.get(cid, {}).get('active', False)}
	
	def get_cube_positions_2d(self, camera, scene):
		""" project all cube 3d positions to 2d screen coordinates """
		positions = {}
		render = scene.render

		for cube_id, cube in self.cubes.items():
			if cube is None or not self.cube_data[cube_id]['active']:
				continue

			try:
				world_pos = cube.location.copy()
				co_2d = world_to_camera_view(scene, camera, world_pos)

				if co_2d.z <= 0:
					continue

				pixel_x = co_2d.x * render.resolution_x
				pixel_y = (1 - co_2d.y) * render.resolution_y

				margin = 50
				if not (-margin <= pixel_x <= render.resolution_x + margin and
						-margin <= pixel_y <= render.resolution_y + margin):
					continue

				positions[cube_id] = {
					'x': pixel_x,
					'y': pixel_y,
					'depth': co_2d.z,
					'world_pos': (world_pos.x, world_pos.y, world_pos.z),
					'is_noise': self.cube_data[cube_id]['is_noise'],
					'size': self.cube_data[cube_id]['size']
				}
			except:
				continue

		return positions
	
	def get_stats(self):
		""" self explanatory """
		active = self.get_active_cubes()
		normal = sum(1 for cid in active if not self.cube_data[cid]['is_noise'])
		noise = sum(1 for cid in active if self.cube_data[cid]['is_noise'])

		return {
			'total_spawned': self.next_cube_id,
			'active': len(active),
			'normal': normal,
			'noise': noise
		}
	
# --------------------------
# CUBE TRAFFIC CONTROLLER
# --------------------------

class TrafficController:
	"""
	controls cube spawn rate using poisson process with sine density waves.
	"""
	def __init__(self, cube_manager: CubeManager):
		self.cube_manager = cube_manager
		self.total_spawned = 0

	def update(self, frame):
		""" spawn cubes based on current density wave phase """
		dt = 1.0 / CONFIG.fps
		time_sec = frame / CONFIG.fps
		spawn_rate = CONFIG.get_spawn_rate(time_sec)
		num_spawns = poisson_sample(spawn_rate, dt)

		for _ in range(num_spawns):
			is_noise = (CONFIG.enable_noise_cubes and
				 random.random() < CONFIG.noise_cube_spawn_rate)
			self.cube_manager.spawn_cube(is_noise=is_noise)
			self.total_spawned += 1

	def get_current_rate(self, frame):
		""" get current spawn rate for logging """
		time_sec = frame / CONFIG.fps

		return CONFIG.get_spawn_rate(time_sec)
	
# --------------------------
# GROUND TRUTH EXPORTING
# --------------------------

def export_ground_truth(frame, cube_manager: CubeManager, camera):
	""" export cube positons for current frame """
	scene = bpy.context.scene
	positions = cube_manager.get_cube_positions_2d(camera, scene)
	stats = cube_manager.get_stats()

	data = {
		'frame': frame,
		'camera_angle': CONFIG.camera_angle_deg,
		'simulation_time': frame / CONFIG.fps,
		'stats': stats,
		'cubes': positions
	}

	filepath = os.path.join(CONFIG.ground_truth_dir, f"frame_{frame:05d}.json")
	with open(filepath, 'w') as f:
		json.dump(data, f, indent=2)

# --------------------------
# MAIN SIMULATION
# --------------------------

def run_simulation():
	""" where the magic happens """
	print(f"camera angle: {CONFIG.camera_angle_deg}°" )
	print(f"sim duration: {CONFIG.sim_duration} seconds ({CONFIG.total_frames} frames)")
	print(f"noise cubes: {'enabled' if CONFIG.enable_noise_cubes else 'disabled'}")
	print(f"traffic density range: {CONFIG.spawn_rate_light:.2f} - {CONFIG.spawn_rate_heavy:.2f} cubes/sec")
	print()

	print("setting up environment ...")
	clear_scene()
	belt = create_conveyor_belt()
	camera = setup_camera()
	setup_lighting()
	setup_render_settings()

	cube_manager = CubeManager()
	traffic_controller = TrafficController(cube_manager)
	scene = bpy.context.scene
	total_frames = CONFIG.total_frames

	print("running simulation ...")
	for frame in range(1, total_frames + 1):
		scene.frame_set(frame)

		traffic_controller.update(frame)
		cube_manager.update_positions(frame)
		bpy.context.view_layer.update()
		export_ground_truth(frame, cube_manager, camera)
		scene.render.filepath = os.path.join(CONFIG.frames_dir, f"frame_{frame:05d}")
		bpy.ops.render.render(write_still=True)

		if frame % 60 == 0:
			stats = cube_manager.get_stats()
			rate = traffic_controller.get_current_rate(frame)
			time_sec = frame / CONFIG.fps
			print(f"frame {frame}/{total_frames}")
			print(f"active: {stats['active']} (normal: {stats['normal']}, noise: {stats['noise']})")
			print(f"rate: {rate:.2f}/s")
			print()

	stats = cube_manager.get_stats()
	print("simulation completed.")
	print(f"total cubes (predicted ct): {stats['total_spawned']}")
	print(f"frames: {CONFIG.frames_dir}")
	print(f"ground truth: {CONFIG.ground_truth_dir}")

if __name__ == "__main__":
	run_simulation()