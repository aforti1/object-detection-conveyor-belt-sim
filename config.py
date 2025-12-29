"""
configuration file for simulation
"""

import os
import time
import math
import random
from dataclasses import dataclass, field
from typing import Tuple, Dict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class SimConfig:
	seed = int(time.time())

	output_dir = os.path.join(_SCRIPT_DIR, "output")
	frames_subdir = "frames"
	ground_truth_subdir = "ground_truth"
	results_subdir = "results"

	# ENVIRONMENT METRICS
	belt_width = 10.0
	belt_length = 120.0
	belt_color = (0.3, 0.3, 0.3, 1.0)

	cube_size = 1.0
	cube_size_variation = 0.20
	cube_color = (1.0, 0.9, 0.0, 1.0)
	cube_speed = 8.0 # units per second
	cube_speed_variation = 0.05

	enable_noise_cubes = True
	noise_size_factor = 0.50 # 50% of normal cube size
	noise_color = (1.0, 1.0, 1.0, 1.0)
	noise_cube_spawn_rate = 0.03

	spawn_rate_light = 0.15
	spawn_rate_heavy = 2.5
	wave_duration_min = 0.5 # seconds
	wave_duration_max = 45.0

	camera_angle_deg = 90.0
	camera_fov_deg = 90.0
	camera_resolution= (1280, 720)
	camera_orbit_radius = 60.0

	angle_start = 15.0
	angle_end = 90.0
	angle_step = 1.0

	fps = 30
	sim_duration = 200.0 # seconds
	
	counting_line_x_pos = 0.75 # located at 75% of the video width

	# lower / upper mask bounds in hsv
	yellow_lower = (15, 80, 80) 
	yellow_upper = (45, 255, 255)
	white_lower = (0, 0, 200)
	white_upper = (180, 50, 255)

	render_engine = "EEVEE"
	render_samples = 16

	def __post_init__(self):
		"""create output directories"""
		self.frames_dir = os.path.join(self.output_dir, self.frames_subdir)
		self.ground_truth_dir = os.path.join(self.output_dir, self.ground_truth_subdir)
		self.results_dir = os.path.join(self.output_dir, self.results_subdir)

		self.__generate_wave_schedule()

	def __generate_wave_schedule(self):
		"""
		creates random linear wavelengths for object density distribution
		"""
		rng = random.Random(self.seed + 1000)

		self.wave_schedule = []
		current_time = 0.0
		current_value = rng.uniform(0.0, 1.0)
		going_up = rng.choice([True, False])

		while current_time < self.sim_duration + 100:
			duration = rng.uniform(self.wave_duration_min, self.wave_duration_max)
			end_time = current_time + duration

			if going_up:
				end_value = 1.0
			else:
				end_value = 0.0

			self.wave_schedule.append((current_time, end_time, current_value, end_value))

			current_time = end_time
			current_value = end_value
			going_up = not going_up

	def ensure_dirs(self):
		""" create output directories if they don't exist? """
		for d in [self.frames_dir, self.ground_truth_dir, self.results_dir]:
			os.makedirs(d, exist_ok=True)

	@property
	def total_frames(self):
		return int(self.sim_duration * self.fps)
	
	@property
	def counting_line_x_pixels(self):
		return int(self.counting_line_x_pos * self.camera_resolution[0])
	
	@property
	def all_angles(self):
		""" generate list of all camera angles to test """
		angles = []
		angle = self.angle_start
		while angle <= self.angle_end:
			angles.append(angle)
			angle += self.angle_step

		return angles
	
	def get_spawn_rate(self, time_sec):
		""" get instantaneuous spawn rate based on density wave. 
		returns spawn rate at given simulation instance of time. """
		for (start_t, end_t, start_val, end_val) in self.wave_schedule:
			if start_t <= time_sec < end_t:
				phase = (time_sec - start_t) / (end_t - start_t)
				wave = start_val + phase * (end_val - start_val)

				return self.spawn_rate_light + wave * (self.spawn_rate_heavy - self.spawn_rate_light)
			
		return self.spawn_rate_light # fallback
	
CONFIG = SimConfig()