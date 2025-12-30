"""
detection and counting pipeline script that processes rendered frames to detect cubes, apply counting line and output CSV.
to run: python detect_and_count.py
"""

import os
import json
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
import cv2
import glob

from config import SimConfig

CONFIG = SimConfig()

# --------------------------
# DATA STRUCTURES 
# --------------------------

@dataclass
class BoundingBox:
	""" represents a detected bounding box """
	bbox_id: int
	x: int
	y: int
	width: int
	height: int
	area: int
	center_x: float
	center_y: float
	cube_ids: List[int] = field(default_factory=list)
	classification: str = "unknown"
	is_noise_detection: bool = False

	@property
	def x2(self):
		return self.x + self.width
	
	@property
	def y2(self):
		return self.y + self.height
	
@dataclass
class CountedDetection:
	""" a detection that has crossed the counting line """
	frame_index: int
	camera_angle: float
	bbox_id: int
	bbox_area_pixels: int
	classification: str
	num_cubes_in_bbox: int
	cube_ids: List[int]
	bbox_x: int
	bbox_y: int
	bbox_width: int
	bbox_height: int
	bbox_center_x: float
	bbox_center_y: float

# --------------------------
# COLOR DETECTION  
# --------------------------

def create_color_mask(frame, color_type = "yellow"):
	"""creates / returns a binary mask for detecting cubes of specified colors """
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if color_type == "yellow":
		lower = np.array(CONFIG.yellow_lower)
		upper = np.array(CONFIG.yellow_upper)
	elif color_type == "white":
		lower = np.array(CONFIG.white_lower)
		upper = np.array(CONFIG.white_upper)
	else:
		raise ValueError(f"Unknown color type: {color_type}")
	
	mask = cv2.inRange(hsv, lower, upper)

	return mask

def find_connected_components(mask, min_area = 50):
	"""
	find connected components in binary mask and compute bounding boxes.
	it returns a list of component info dicts with bbox coordinates
	"""
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
		mask, connectivity=8
	)

	components = []

	for label_id in range(1, num_labels):
		area = stats[label_id, cv2.CC_STAT_AREA]

		if area < min_area:
			continue

		x = stats[label_id, cv2.CC_STAT_LEFT]
		y = stats[label_id, cv2.CC_STAT_TOP]
		w = stats[label_id, cv2.CC_STAT_WIDTH]
		h = stats[label_id, cv2.CC_STAT_HEIGHT]
		cx, cy = centroids[label_id]

		components.append({
			'label_id': label_id,
			'x': x,
			'y': y,
			'width': w,
			'height': h,
			'area': area,
			'center_x': cx,
			'center_y': cy
		})

	return components

# --------------------------
# GROUND TRUTH LINKING
# --------------------------

def load_ground_truth(frame_index):
	""" loads ground truth json for a given frame """
	file = os.path.join(CONFIG.ground_truth_dir, f"frame_{frame_index:05d}.json")

	if not os.path.exists(file):
		return None
	
	with open(file, 'r') as f:
		return json.load(f)
	
def point_in_bbox(px, py, bbox, tolerance = 10):
	""" check if a point falls within a bounding box with tolerace """
	return (bbox.x - tolerance <= px <= bbox.x + bbox.width + tolerance and
			bbox.y - tolerance <= py <= bbox.y + bbox.height + tolerance)

def link_cubes_to_bboxes(components, ground_truth, is_noise = False):
	"""
	links ground truth cube positions to detected bounding boxes.
	for each bounding box, determine which cube IDs fall within it based on their projected 2d postions from ground truth.
	returns a list a BoundingBox objects with cube_ids populated
	"""
	bboxes = []
	cubes = ground_truth.get('cubes', {})

	for i, comp in enumerate(components):
		bbox = BoundingBox(
			bbox_id=i,
			x=comp['x'],
			y=comp['y'],
			width=comp['width'],
			height=comp['height'],
			area=comp['area'],
			center_x=comp['center_x'],
			center_y=comp['center_y'],
			is_noise_detection=is_noise
		)
		bboxes.append(bbox)

	# link cubes to bounding boxes
	for cube_id_str, cube_info in cubes.items():
		cube_id = int(cube_id_str)
		cube_x = cube_info['x']
		cube_y = cube_info['y']
		cube_is_noise = cube_info.get('is_noise', False)

		if cube_is_noise != is_noise:
			continue

		# find which bbox contains the cube
		for bbox in bboxes:
			if point_in_bbox(cube_x, cube_y, bbox):
				bbox.cube_ids.append(cube_id)
				break

	# classify based on cube count
	for bbox in bboxes:
		if is_noise:
			bbox.classification = "noise"
		elif len(bbox.cube_ids) == 0:
			bbox.classification = "unknown"
		elif len(bbox.cube_ids) == 1:
			bbox.classification = "accurate"
		else:
			bbox.classification = "cluster"

	return bboxes

# --------------------------
# COUNTING LINE LOGIC
# --------------------------

class CountingLineTracker:
	"""
	tracks detections crossing the counting line.
	a detection is counted once when its center crossed the line.
	uses cube set deduplication to handle same cubes appearing in consecutive frames.
	"""
	def __init__(self, line_x):
		self.line_x = line_x
		self.counted_detections = []
		self.counted_cube_sets = set()
		self.counted_cube_ids = set()

	def process_frame(self, frame_index, bboxes, camera_angle):
		""" process bounding boxes for a frame and count those that cross counting time """
		for bbox in bboxes:
			if bbox.center_x >= self.line_x:
				if bbox.classification == "unknown":
					continue

				if len(bbox.cube_ids) == 0:
					continue

				cube_set = frozenset(bbox.cube_ids)
				if cube_set in self.counted_cube_sets:
					continue

				detection = CountedDetection(
					frame_index=frame_index,
					camera_angle=camera_angle,
					bbox_id=bbox.bbox_id,
					bbox_area_pixels=bbox.area,
					classification=bbox.classification,
					num_cubes_in_bbox=len(bbox.cube_ids),
					cube_ids=sorted(bbox.cube_ids),
					bbox_x=bbox.x,
					bbox_y=bbox.y,
					bbox_width=bbox.width,
					bbox_height=bbox.height,
					bbox_center_x=bbox.center_x,
					bbox_center_y=bbox.center_y
				)

				self.counted_detections.append(detection)
				self.counted_cube_sets.add(cube_set)
				self.counted_cube_ids.update(bbox.cube_ids)

	def get_results(self):
		""" return all counted detections """
		return self.counted_detections
	
	def get_statistics(self):
		""" return counting statistics """
		accurate = sum(1 for d in self.counted_detections if d.classification == "accurate")
		cluster = sum(1 for d in self.counted_detections if d.classification == "cluster")
		noise = sum(1 for d in self.counted_detections if d.classification == "noise")

		cubes_in_clusters = sum(
			d.num_cubes_in_bbox
			for d in self.counted_detections
			if d.classification == "cluster"
		)

		return {
			'total_detections': len(self.counted_detections),
			'accurate': accurate,
			'cluster': cluster,
			'noise': noise,
			'cubes_in_clusters': cubes_in_clusters,
			'unique_cubes_counted': len(self.counted_cube_ids)
		}
	
# --------------------------
# CSV EXPORT
# --------------------------

def export_csv(detections, camera_angle, output_dir = None):
	""" export counted detections to CSV """
	if output_dir is None:
		output_dir = CONFIG.results_subdir

	os.makedirs(output_dir, exist_ok=True)

	file = os.path.join(
		output_dir,
		f"detections_angle_{int(camera_angle):02d}.csv"
	)

	fieldnames = [
		'frame_index',
		'camera_angle',
		'bbox_id',
		'bbox_area_pixels',
		'classification',
		'num_cubes_in_bbox',
		'cube_ids',
		'bbox_x',
		'bbox_y',
		'bbox_width',
		'bbox_height',
		'bbox_center_x',
		'bbox_center_y'
	]

	with open(file, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()

		for det in detections:
			writer.writerow({
				'frame_index': det.frame_index,
				'camera_angle': det.camera_angle,
				'bbox_id': det.bbox_id,
				'bbox_area_pixels': det.bbox_area_pixels,
				'classification': det.classification,
				'num_cubes_in_bbox': det.num_cubes_in_bbox,
				'cube_ids': json.dumps(det.cube_ids),
				'bbox_x': det.bbox_x,
				'bbox_y': det.bbox_y,
				'bbox_width': det.bbox_width,
				'bbox_height': det.bbox_height,
				'bbox_center_x': round(det.bbox_center_x, 2),
				'bbox_center_y': round(det.bbox_center_y, 2)
			})
	
	print(f"exported {len(detections)} detections to {file}")
	return file

# --------------------------
# VISUALIZATION
# --------------------------

def annotate_frame(frame, bboxes, counting_line_x, output_path, frame_index = 0, total_count = 0):
	""" update given frame with bounding box detections and counting line """
	vis = frame.copy()
	count_line_color = (255, 0, 255)
	cv2.line(vis, (counting_line_x, 0), (counting_line_x, frame.shape[0]), count_line_color, 2)
	cv2.putText(vis, "COUNT LINE", (counting_line_x + 10, 30),
			 cv2.FONT_HERSHEY_SIMPLEX, 0.6, count_line_color, 2)
	bbox_color = (0, 0, 255)
	
	for bbox in bboxes:
		cv2.rectangle(vis, (bbox.x, bbox.y), (bbox.x2, bbox.y2), bbox_color, 1)

	cv2.putText(vis, f"total count: {total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

	return vis

# --------------------------
# MAIN PIPELINE?
# --------------------------

def run_detection_pipeline(enableVisuals):
	""" where the magic happens (detection and counting) """
	if not os.path.exists(CONFIG.frames_dir):
		print("frames directory not found fr. did you run blender_sim.py first by chance?")
		return None
	
	tracker = CountingLineTracker(CONFIG.counting_line_x_pixels)
	
	video_writer = None
	if enableVisuals:
		video_path = os.path.join(CONFIG.output_dir, f"annotated_video_angle_{int(CONFIG.camera_angle_deg):02d}.mp4")
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		video_writer = cv2.VideoWriter(video_path, fourcc, CONFIG.fps, CONFIG.camera_resolution)

	total_frames = CONFIG.total_frames
	print(f"processing {total_frames} frames...")

	for frame_idx in range(1, total_frames+1):
		frame_path = os.path.join(CONFIG.frames_dir, f"frame_{frame_idx:05d}.png")

		if not os.path.exists(frame_path):
			print(f"frame {frame_idx} not found so we skipping that jawn")
			continue

		frame = cv2.imread(frame_path)
		if frame is None:
			print(f"coudn't read the {frame_idx}th frame, so skip")
			continue

		ground_truth = load_ground_truth(frame_idx)
		if ground_truth is None:
			print(f"ground truth for frame {frame_idx} not found, skipping ts")
			continue

		yellow_mask = create_color_mask(frame, "yellow")
		yellow_components = find_connected_components(yellow_mask)
		yellow_bboxes = link_cubes_to_bboxes(yellow_components, ground_truth, is_noise=False)
		all_bboxes = yellow_bboxes.copy()

		if CONFIG.enable_noise_cubes:
			white_mask = create_color_mask(frame, "white")
			white_components = find_connected_components(white_mask, min_area=0)
			white_bboxes = link_cubes_to_bboxes(white_components, ground_truth, is_noise=True)
			all_bboxes.extend(white_bboxes)

		tracker.process_frame(frame_idx, all_bboxes, CONFIG.camera_angle_deg)
		stats = tracker.get_statistics()

		if video_writer:
			vis = annotate_frame(frame, all_bboxes, CONFIG.counting_line_x_pixels, stats['total_detections'])
			video_writer.write(vis)

		if frame_idx % 120 == 0:
			print(f"frame {frame_idx:4d}/{total_frames}")
			print(f"counted: {stats['total_detections']}")
			print(f"accurate: {stats['accurate']} | cluster: {stats['cluster']} | noise: {stats['noise']}")
			print()

	if video_writer:
		video_writer.release()
		print(f"video saved to: {video_path}")

	detections = tracker.get_results()
	csv_path = export_csv(detections, CONFIG.camera_angle_deg)

	return csv_path
	
if __name__ == "__main__":
	run_detection_pipeline(enableVisuals=True)