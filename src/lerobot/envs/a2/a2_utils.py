"""A2 Environment utilities.

This module contains utility functions for object placement, mask generation,
and place instruction sampling used by the A2 simulation environment.
"""

import math
import random

import numpy as np

from .a2_constants import (
    IMAGE_SIZE,
    MAX_DIS,
    MIN_DIS,
    OBJ_DIRECTION_MAP,
    OBJ_UNSEEN_DIRECTION_MAP,
    PIXEL_SIZE,
    PLACE_LANG_TEMPLATES,
)


def generate_drop_positions(num_obj, workspace_limits, min_dist=0.15):
    """Generate random drop positions for objects ensuring minimum distance between them.

    Args:
        num_obj: Number of objects to generate positions for
        workspace_limits: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        min_dist: Minimum distance between objects

    Returns:
        List of (x, y) positions
    """
    def distance(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    drop_positions = []
    tried_times = 0
    while len(drop_positions) < num_obj:
        new_position = (
            random.uniform(workspace_limits[0][0] + 0.05, workspace_limits[0][1] - 0.05),
            random.uniform(workspace_limits[1][0] + 0.05, workspace_limits[1][1] - 0.05)
        )

        if all(distance(new_position, p) > min_dist for p in drop_positions):
            drop_positions.append(new_position)

        tried_times += 1
        if tried_times - num_obj >= 50:
            print("invalid position generations")
            break

    return drop_positions


def generate_mask_map(bbox_centers, bbox_sizes):
    """Generate occupancy mask map from bounding boxes.

    Args:
        bbox_centers: List of [cx, cy] centers
        bbox_sizes: List of [w, h] sizes

    Returns:
        Binary mask where 1 = available, 0 = occupied
    """
    # 0: being occupied -> 1: available
    mask_map = np.ones((IMAGE_SIZE, IMAGE_SIZE))
    for i in range(len(bbox_centers)):
        cx, cy = bbox_centers[i]
        w, h = bbox_sizes[i]
        mask_map[
            max(0, cx - h//2) : min(IMAGE_SIZE-1, cx + h//2),
            max(0, cy - w//2) : min(IMAGE_SIZE-1, cy + w//2)
        ] = 0
    return mask_map


def generate_sector_mask(shape, centre, angle_range, radius_min=None, radius_max=None, pixel_size=PIXEL_SIZE):
    """Generate a boolean mask for a circular sector.

    Args:
        shape: Image shape [h, w]
        centre: Center point [cx, cy]
        angle_range: [start_angle, stop_angle] in degrees (clockwise order)
        radius_min: Minimum radius in pixels (default: MIN_DIS/pixel_size)
        radius_max: Maximum radius in pixels (default: MAX_DIS/pixel_size)
        pixel_size: Pixel size for converting distance to pixels

    Returns:
        Boolean mask array
    """
    if radius_min is None:
        radius_min = MIN_DIS / pixel_size
    if radius_max is None:
        radius_max = MAX_DIS / pixel_size

    x, y = np.ogrid[:shape[0], :shape[1]]  # grid_index
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)

    # circular mask (ring)
    circmask_in = r2 >= radius_min * radius_min
    circmask_out = r2 <= radius_max * radius_max
    # angular mask
    anglemask = theta <= (tmax - tmin)

    return circmask_in * circmask_out * anglemask


def generate_unseen_place_inst(obj_labels, obj_dirs, bbox_ids, bbox_centers, bbox_sizes, pixel_size=PIXEL_SIZE):
    """Generate place instruction for unseen (test) objects.

    Args:
        obj_labels: Dict mapping obj_id to list of labels
        obj_dirs: Dict mapping obj_id to directory name
        bbox_ids: List of object IDs with bounding boxes
        bbox_centers: List of [cx, cy] centers
        bbox_sizes: List of [w, h] sizes
        pixel_size: Pixel size for mask generation

    Returns:
        Tuple of (language_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask)
    """
    ori_lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask = None, None, None, None, None
    mask_map = generate_mask_map(bbox_centers, bbox_sizes)

    choices = np.random.choice(a=range(len(obj_labels)), size=len(obj_labels), replace=False)
    for choice_idx in choices:  # object checking id
        obj_id = list(obj_labels.keys())[choice_idx]
        # if have unique label, use unique label
        obj_name = obj_labels[obj_id][0]
        obj_dir = obj_dirs[obj_id]

        if obj_id in bbox_ids:
            choice_bbox_idx = bbox_ids.index(obj_id)
        else:
            # the reference object is completely occluded
            continue

        dir_choices = []
        for direction in OBJ_UNSEEN_DIRECTION_MAP:
            if obj_dir in OBJ_UNSEEN_DIRECTION_MAP[direction]:
                dir_choices.append(direction)
        region_name = np.random.choice(a=dir_choices, size=1)[0]

        if region_name in ["on", "in", "to", "upside", "on top of"]:
            sector_mask = generate_sector_mask(
                (IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx],
                angle_range=[0, 359.99],
                radius_min=0, radius_max=min(bbox_sizes[choice_bbox_idx])//2
            )
            all_mask = sector_mask
        elif region_name in ["near", "around", "next to", "beside", "close to", "surrounding to"]:
            sector_mask = generate_sector_mask(
                (IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx],
                angle_range=[0, 359.99],
                radius_min=max(MIN_DIS/pixel_size, max(bbox_sizes[choice_bbox_idx])//2),
                radius_max=MAX_DIS/pixel_size
            )
            all_mask = mask_map * sector_mask

        if np.count_nonzero(all_mask) > np.count_nonzero(sector_mask) * 2 / 3 and np.count_nonzero(all_mask) > 800:
            ref_obj_ids = [obj_id]
            dir_phrase = region_name
            # diverse
            place_lang_goal = np.random.choice(a=PLACE_LANG_TEMPLATES, size=1)[0]
            ori_lang_goal = place_lang_goal.format(reference=obj_name, direction=dir_phrase)

            ref_obj_centers = [bbox_centers[choice_bbox_idx]]
            ref_regions = [region_name]

            if ori_lang_goal:
                break

    return ori_lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask


def generate_place_inst(obj_labels, obj_dirs, bbox_ids, bbox_centers, bbox_sizes, pixel_size=PIXEL_SIZE):
    """Generate place instruction for train objects.

    Args:
        obj_labels: Dict mapping obj_id to list of labels
        obj_dirs: Dict mapping obj_id to directory name
        bbox_ids: List of object IDs with bounding boxes
        bbox_centers: List of [cx, cy] centers
        bbox_sizes: List of [w, h] sizes
        pixel_size: Pixel size for mask generation

    Returns:
        Tuple of (language_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask)
    """
    ori_lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask = None, None, None, None, None
    mask_map = generate_mask_map(bbox_centers, bbox_sizes)

    choices = np.random.choice(a=range(len(obj_labels)), size=len(obj_labels), replace=False)
    for choice_idx in choices:  # object checking id
        obj_id = list(obj_labels.keys())[choice_idx]
        # if have unique label, use unique label
        obj_name = obj_labels[obj_id][0]
        obj_dir = obj_dirs[obj_id]

        if obj_id in bbox_ids:
            choice_bbox_idx = bbox_ids.index(obj_id)
        else:
            # the reference object is completely occluded
            continue

        dir_choices = []
        for direction in OBJ_DIRECTION_MAP:
            if obj_dir in OBJ_DIRECTION_MAP[direction]:
                dir_choices.append(direction)

        region_name = np.random.choice(a=dir_choices, size=1)[0]

        if region_name in ["on", "in", "to", "upside", "on top of"]:
            sector_mask = generate_sector_mask(
                (IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx],
                angle_range=[0, 359.99],
                radius_min=0, radius_max=min(bbox_sizes[choice_bbox_idx])//2
            )
            all_mask = sector_mask
        elif region_name in ["near", "around", "next to", "beside", "close to", "surrounding to"]:
            sector_mask = generate_sector_mask(
                (IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx],
                angle_range=[0, 359.99],
                radius_min=max(MIN_DIS/pixel_size, max(bbox_sizes[choice_bbox_idx])//2),
                radius_max=MAX_DIS/pixel_size
            )
            all_mask = mask_map * sector_mask

        if np.count_nonzero(all_mask) > np.count_nonzero(sector_mask) * 2 / 3 and np.count_nonzero(all_mask) > 800:
            ref_obj_ids = [obj_id]
            dir_phrase = region_name
            place_lang_goal = np.random.choice(a=PLACE_LANG_TEMPLATES, size=1)[0]
            ori_lang_goal = place_lang_goal.format(reference=obj_name, direction=dir_phrase)

            ref_obj_centers = [bbox_centers[choice_bbox_idx]]
            ref_regions = [region_name]

            if ori_lang_goal:
                break

    return ori_lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask
