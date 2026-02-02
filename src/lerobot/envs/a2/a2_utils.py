"""A2 Environment utilities.

This module contains utility functions for object placement, mask generation,
and place instruction sampling used by the A2 simulation environment.
"""

from __future__ import annotations

import logging
import math
import random

import numpy as np
import torch

from .a2_constants import (
    IMAGE_SIZE,
    IN_REGION,
    MAX_DIS,
    MIN_DIS,
    OBJ_DIRECTION_MAP,
    OBJ_UNSEEN_DIRECTION_MAP,
    OUT_REGION,
    PIXEL_SIZE,
    PLACE_LANG_TEMPLATES,
    PP_PIXEL_SIZE,
    WORKSPACE_LIMITS,
)

logger = logging.getLogger(__name__)

# Thresholds for place instruction validity
MIN_VALID_MASK_RATIO = 0.5  # Minimum ratio of available area to total sector area
MIN_VALID_MASK_PIXELS = 400  # Minimum number of available pixels for valid placement

# Direction categories
CONTAINMENT_DIRECTIONS = frozenset(["on", "in", "to", "upside", "on top of"])
PROXIMITY_DIRECTIONS = frozenset(["near", "around", "next to", "beside", "close to", "surrounding to"])


def generate_drop_positions(
    num_obj: int,
    workspace_limits: np.ndarray,
    min_dist: float = 0.15,
    margin: float = 0.05,
    max_retries: int = 50,
) -> list[tuple[float, float]]:
    """Generate random drop positions for objects ensuring minimum distance between them.

    Args:
        num_obj: Number of objects to generate positions for
        workspace_limits: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        min_dist: Minimum distance between objects
        margin: Margin from workspace boundaries
        max_retries: Maximum number of extra attempts beyond num_obj

    Returns:
        List of (x, y) positions. May contain fewer than num_obj if placement fails.
    """
    x_min, x_max = workspace_limits[0][0] + margin, workspace_limits[0][1] - margin
    y_min, y_max = workspace_limits[1][0] + margin, workspace_limits[1][1] - margin

    drop_positions: list[tuple[float, float]] = []
    attempts = 0

    while len(drop_positions) < num_obj and attempts < num_obj + max_retries:
        new_position = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))

        # Check distance to all existing positions
        if all(math.hypot(new_position[0] - p[0], new_position[1] - p[1]) > min_dist for p in drop_positions):
            drop_positions.append(new_position)

        attempts += 1

    if len(drop_positions) < num_obj:
        logger.warning(
            "Could only place %d of %d objects after %d attempts",
            len(drop_positions),
            num_obj,
            attempts,
        )

    return drop_positions


def generate_mask_map(bbox_centers: list, bbox_sizes: list) -> np.ndarray:
    """Generate occupancy mask map from bounding boxes.

    Args:
        bbox_centers: List of [cx, cy] centers in pixel coordinates
        bbox_sizes: List of [w, h] sizes in pixels

    Returns:
        Binary mask where 1 = available, 0 = occupied
    """
    mask_map = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for center, size in zip(bbox_centers, bbox_sizes, strict=False):
        cx, cy = center
        w, h = size
        # Note: cx corresponds to row (height), cy to column (width)
        x_start = max(0, cx - h // 2)
        x_end = min(IMAGE_SIZE, cx + h // 2)
        y_start = max(0, cy - w // 2)
        y_end = min(IMAGE_SIZE, cy + w // 2)
        mask_map[x_start:x_end, y_start:y_end] = 0

    return mask_map


def generate_sector_mask(
    shape: tuple[int, int],
    centre: tuple[int, int],
    angle_range: tuple[float, float],
    radius_min: float | None = None,
    radius_max: float | None = None,
    pixel_size: float = PIXEL_SIZE,
) -> np.ndarray:
    """Generate a boolean mask for a circular sector.

    Args:
        shape: Image shape (height, width)
        centre: Center point (cx, cy) in pixels
        angle_range: (start_angle, stop_angle) in degrees
        radius_min: Minimum radius in pixels (default: MIN_DIS/pixel_size)
        radius_max: Maximum radius in pixels (default: MAX_DIS/pixel_size)
        pixel_size: Pixel size for converting distance to pixels

    Returns:
        Boolean mask array of shape `shape`
    """
    if radius_min is None:
        radius_min = MIN_DIS / pixel_size
    if radius_max is None:
        radius_max = MAX_DIS / pixel_size

    # Create coordinate grids
    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # Ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # Convert cartesian to polar coordinates
    dx, dy = x - cx, y - cy
    r_squared = dx * dx + dy * dy
    theta = (np.arctan2(dx, dy) - tmin) % (2 * np.pi)

    # Create masks
    radius_mask = (r_squared >= radius_min**2) & (r_squared <= radius_max**2)
    angle_mask = theta <= (tmax - tmin)

    return radius_mask & angle_mask


def _generate_place_inst_impl(
    obj_labels: dict,
    obj_dirs: dict,
    bbox_ids: list,
    bbox_centers: list,
    bbox_sizes: list,
    direction_map: dict,
    pixel_size: float = PIXEL_SIZE,
) -> tuple[str | None, list | None, list | None, list | None, np.ndarray | None]:
    """Internal implementation for generating place instructions.

    Args:
        obj_labels: Dict mapping obj_id to list of labels
        obj_dirs: Dict mapping obj_id to directory name
        bbox_ids: List of object IDs with bounding boxes
        bbox_centers: List of [cx, cy] centers
        bbox_sizes: List of [w, h] sizes
        direction_map: Direction to object mapping (OBJ_DIRECTION_MAP or OBJ_UNSEEN_DIRECTION_MAP)
        pixel_size: Pixel size for mask generation

    Returns:
        Tuple of (language_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask)
        Returns (None, None, None, None, None) if no valid instruction could be generated.
    """
    mask_map = generate_mask_map(bbox_centers, bbox_sizes)
    obj_id_list = list(obj_labels.keys())

    # Try objects in random order
    for obj_id in np.random.permutation(obj_id_list):
        obj_name = obj_labels[obj_id][0]
        obj_dir = obj_dirs[obj_id]

        # Skip if object not visible (not in bbox_ids)
        if obj_id not in bbox_ids:
            continue

        bbox_idx = bbox_ids.index(obj_id)
        bbox_center = bbox_centers[bbox_idx]
        bbox_size = bbox_sizes[bbox_idx]

        # Find valid directions for this object
        valid_directions = [d for d in direction_map if obj_dir in direction_map[d]]
        np.random.shuffle(valid_directions)

        for direction in valid_directions:
            # Generate appropriate sector mask based on direction type
            if direction in CONTAINMENT_DIRECTIONS:
                # "on", "in" etc - place inside/on the object
                sector_mask = generate_sector_mask(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    bbox_center,
                    angle_range=(0, 359.99),
                    radius_min=0,
                    radius_max=min(bbox_size) // 2,
                )
                combined_mask = sector_mask
            elif direction in PROXIMITY_DIRECTIONS:
                # "near", "beside" etc - place around the object
                sector_mask = generate_sector_mask(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    bbox_center,
                    angle_range=(0, 359.99),
                    radius_min=max(MIN_DIS / pixel_size, max(bbox_size) // 2),
                    radius_max=MAX_DIS / pixel_size,
                )
                combined_mask = mask_map * sector_mask
            else:
                continue

            # Check if enough valid area exists
            sector_area = np.count_nonzero(sector_mask)
            valid_area = np.count_nonzero(combined_mask)

            if valid_area > sector_area * MIN_VALID_MASK_RATIO and valid_area > MIN_VALID_MASK_PIXELS:
                # Generate language goal
                template = np.random.choice(PLACE_LANG_TEMPLATES)
                lang_goal = template.format(reference=obj_name, direction=direction)

                return lang_goal, [obj_id], [bbox_center], [direction], combined_mask

    return None, None, None, None, None


def generate_place_inst(
    obj_labels: dict,
    obj_dirs: dict,
    bbox_ids: list,
    bbox_centers: list,
    bbox_sizes: list,
    pixel_size: float = PIXEL_SIZE,
) -> tuple[str | None, list | None, list | None, list | None, np.ndarray | None]:
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
    return _generate_place_inst_impl(
        obj_labels, obj_dirs, bbox_ids, bbox_centers, bbox_sizes, OBJ_DIRECTION_MAP, pixel_size
    )


def generate_unseen_place_inst(
    obj_labels: dict,
    obj_dirs: dict,
    bbox_ids: list,
    bbox_centers: list,
    bbox_sizes: list,
    pixel_size: float = PIXEL_SIZE,
) -> tuple[str | None, list | None, list | None, list | None, np.ndarray | None]:
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
    return _generate_place_inst_impl(
        obj_labels, obj_dirs, bbox_ids, bbox_centers, bbox_sizes, OBJ_UNSEEN_DIRECTION_MAP, pixel_size
    )


def generate_all_place_dist(
    ref_obj_id: int,
    ref_obj_name: str,
    ref_dir_phase: str,
    place_valid_mask: np.ndarray | None,
    obj_labels: dict,
    bbox_ids: list,
    bbox_centers: list,
    bbox_sizes: list,
    grasped_obj_size: list | None = None,
    pixel_size: float = PIXEL_SIZE,
) -> np.ndarray:
    """Generate valid place mask based on reference object and direction.

    This function creates a mask indicating all valid placement locations
    based on the reference object position, direction phrase (on/in/near/around),
    and collision avoidance with other objects.

    Args:
        ref_obj_id: ID of the reference object
        ref_obj_name: Name of the reference object
        ref_dir_phase: Direction phrase (on, in, near, around, etc.)
        place_valid_mask: Existing valid mask to combine with (or None)
        obj_labels: Dict mapping obj_id to list of labels
        bbox_ids: List of object IDs with bounding boxes
        bbox_centers: List of [cx, cy] centers in pixels
        bbox_sizes: List of [w, h] sizes in pixels
        grasped_obj_size: Size of grasped object [w, h] for collision margin
        pixel_size: Pixel size for distance calculations

    Returns:
        Binary mask where True = valid placement location
    """
    mask_map = generate_mask_map(bbox_centers, bbox_sizes)

    # Build list of candidate reference objects
    candidate_ref_obj_ids = []
    if place_valid_mask is None:
        candidate_ref_obj_ids.append(ref_obj_id)

    # Add objects with the same label as reference (for generalization)
    for obj_id in obj_labels:
        if obj_id != ref_obj_id:
            for obj_name in obj_labels[obj_id]:
                if obj_name == ref_obj_name:
                    candidate_ref_obj_ids.append(obj_id)

    # Initialize or use existing mask
    all_place_valid_mask = place_valid_mask if place_valid_mask is not None else np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE), dtype=bool
    )

    for candidate_ref_obj_id in candidate_ref_obj_ids:
        if candidate_ref_obj_id not in bbox_ids:
            # Reference object is not visible
            continue

        choice_bbox_idx = bbox_ids.index(candidate_ref_obj_id)
        bbox_center = bbox_centers[choice_bbox_idx]
        bbox_size = bbox_sizes[choice_bbox_idx]

        if ref_dir_phase in IN_REGION:
            # "on", "in", "to", "into", "upside", "on top of" - place inside/on the object
            sector_mask = generate_sector_mask(
                (IMAGE_SIZE, IMAGE_SIZE),
                bbox_center,
                angle_range=(0, 359.99),
                radius_min=0,
                radius_max=min(bbox_size) // 2,
                pixel_size=pixel_size,
            )
            all_mask = sector_mask
        elif ref_dir_phase in OUT_REGION:
            # "near", "around", "next to", "beside", "close to", "surrounding to"
            if grasped_obj_size is None:
                sector_mask = generate_sector_mask(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    bbox_center,
                    angle_range=(0, 359.99),
                    radius_min=max(MIN_DIS / pixel_size, max(bbox_size) // 2),
                    radius_max=max(MAX_DIS / pixel_size, max(bbox_size) // 2 + 0.1),
                    pixel_size=pixel_size,
                )
            else:
                # Account for grasped object size to prevent collision
                sector_mask = generate_sector_mask(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    bbox_center,
                    angle_range=(0, 359.99),
                    radius_min=max(MIN_DIS / pixel_size, max(bbox_size) // 2),
                    radius_max=max(
                        MAX_DIS / pixel_size,
                        max(bbox_size) // 2 + max(grasped_obj_size) // 2 + 0.1,
                    ),
                    pixel_size=pixel_size,
                )
            all_mask = mask_map * sector_mask
        else:
            continue

        # Check if enough valid area exists
        sector_area = np.count_nonzero(sector_mask)
        valid_area = np.count_nonzero(all_mask)

        if valid_area > sector_area * 2 / 3 and valid_area > 800:
            all_place_valid_mask = np.logical_or(all_place_valid_mask, all_mask)

    return all_place_valid_mask


class PlaceGenerator:
    """Generator for place action candidates from heightmap.

    This class generates place pose candidates using sector masks around objects
    in the scene, similar to the original A2 Placenet implementation.
    """

    def __init__(self):
        pass

    def place_generation_return_gt(
        self,
        depth_heightmap: np.ndarray,
        obj_centers: list,
        obj_sizes: list,
        ref_obj_centers: list,
        ref_regions: list,
        valid_place_mask: np.ndarray | None = None,
        grasped_obj_size: list | None = None,
        sample_num_each_object: int = 5,
        workspace_limits: np.ndarray = WORKSPACE_LIMITS,
        pixel_size: float = PIXEL_SIZE,
        topdown_place_rot: bool = False,
        action_variance: bool = False,
    ) -> tuple[np.ndarray | None, list, list]:
        """Generate place pose candidates from heightmap.

        Args:
            depth_heightmap: Depth heightmap of the scene
            obj_centers: List of [cx, cy] object centers in pixels
            obj_sizes: List of [w, h] object sizes in pixels
            ref_obj_centers: List of reference object centers
            ref_regions: List of direction phrases for each reference object
            valid_place_mask: Pre-computed valid placement mask
            grasped_obj_size: Size of grasped object for collision margin
            sample_num_each_object: Number of samples per object
            workspace_limits: Workspace bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            pixel_size: Size of each pixel in meters
            topdown_place_rot: Whether to use top-down rotation
            action_variance: Whether to use random sampling

        Returns:
            Tuple of (sample_indices, place_pose_list, valid_places_list)
        """
        mask_map = generate_mask_map(obj_centers, obj_sizes)

        sample_inds = None
        valid_places_list = []

        # Generate samples for each object (both IN and OUT regions)
        for i, obj_center in enumerate(obj_centers):
            obj_size = obj_sizes[i]

            # IN region samples (on/in the object)
            in_mask = generate_sector_mask(
                (IMAGE_SIZE, IMAGE_SIZE),
                obj_center,
                angle_range=(0, 359.99),
                radius_min=0,
                radius_max=min(obj_size) // 3,
                pixel_size=pixel_size,
            )
            in_mask_indices = np.argwhere(in_mask)

            if in_mask_indices.shape[0] > 0:
                num_samples = min(in_mask_indices.shape[0], sample_num_each_object)
                if action_variance:
                    in_sample_indices = in_mask_indices[
                        np.random.choice(in_mask_indices.shape[0], num_samples, replace=False)
                    ]
                else:
                    in_sample_indices = in_mask_indices[
                        np.linspace(0, in_mask_indices.shape[0] - 1, num_samples, dtype=int)
                    ]

                if sample_inds is None:
                    sample_inds = in_sample_indices
                else:
                    sample_inds = np.concatenate((sample_inds, in_sample_indices), axis=0)

                # Mark valid places if this is a reference object with IN region direction
                if valid_place_mask is None:
                    if list(obj_center) in [list(c) for c in ref_obj_centers] and ref_regions[0] in IN_REGION:
                        valid_places_list += list(
                            range(sample_inds.shape[0] - num_samples, sample_inds.shape[0])
                        )

            # OUT region samples (near/around the object)
            if grasped_obj_size is None:
                out_mask = generate_sector_mask(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    obj_center,
                    angle_range=(0, 359.99),
                    radius_min=max(MIN_DIS / pixel_size, max(obj_size) // 2),
                    radius_max=max(MAX_DIS / pixel_size, max(obj_size) // 2 + 0.1),
                    pixel_size=pixel_size,
                )
            else:
                out_mask = generate_sector_mask(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    obj_center,
                    angle_range=(0, 359.99),
                    radius_min=max(MIN_DIS / pixel_size, max(obj_size) // 2 + max(grasped_obj_size) // 2),
                    radius_max=max(
                        MAX_DIS / pixel_size, max(obj_size) // 2 + max(grasped_obj_size) // 2 + 0.1
                    ),
                    pixel_size=pixel_size,
                )

            out_mask = out_mask * mask_map
            out_mask_indices = np.argwhere(out_mask)

            if out_mask_indices.shape[0] > 0:
                num_samples = min(out_mask_indices.shape[0], sample_num_each_object)
                if action_variance:
                    out_sample_indices = out_mask_indices[
                        np.random.choice(out_mask_indices.shape[0], num_samples, replace=False)
                    ]
                else:
                    out_sample_indices = out_mask_indices[
                        np.linspace(0, out_mask_indices.shape[0] - 1, num_samples, dtype=int)
                    ]

                if sample_inds is None:
                    sample_inds = out_sample_indices
                else:
                    sample_inds = np.concatenate((sample_inds, out_sample_indices), axis=0)

                # Mark valid places if this is a reference object with OUT region direction
                if valid_place_mask is None:
                    if list(obj_center) in [list(c) for c in ref_obj_centers] and ref_regions[0] in OUT_REGION:
                        valid_places_list += list(
                            range(sample_inds.shape[0] - num_samples, sample_inds.shape[0])
                        )

        # If we have a valid_place_mask, use it to determine valid places
        if valid_place_mask is not None and sample_inds is not None:
            valid_places_list = list(
                np.where(valid_place_mask[sample_inds[:, 0], sample_inds[:, 1]])[0]
            )

        # Convert pixel indices to 3D poses
        place_pose_list = []
        if sample_inds is not None:
            sample_inds = sample_inds.transpose(1, 0)  # [2, N]
            pos_z = depth_heightmap[sample_inds[0, :], sample_inds[1, :]]

            pos_x = sample_inds[0, :] * pixel_size + workspace_limits[0][0]
            pos_y = sample_inds[1, :] * pixel_size + workspace_limits[1][0]
            place_points = np.float32([pos_x, pos_y, pos_z]).transpose(1, 0)

            # Add rotation (quaternion)
            if topdown_place_rot:
                # Top-down gripper rotation
                rot_quat = np.array([0.707, 0, -0.707, 0])
            else:
                # Identity rotation
                rot_quat = np.array([0, 0, 0, 1])

            place_poses = np.concatenate(
                (place_points, np.tile(rot_quat, (place_points.shape[0], 1))), axis=1
            )

            place_pose_list = [place_poses[i] for i in range(len(place_poses))]
            sample_inds = sample_inds.transpose(1, 0)  # Back to [N, 2]

        return sample_inds, place_pose_list, valid_places_list


def preprocess_pp_unified(
    pts: np.ndarray,
    clip_feats: torch.Tensor,
    clip_sims: torch.Tensor,
    action_set: list,
    sample_num: int = 1024,
    sample_action: bool = False,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Preprocess points and actions for the vision-language-action model.

    This function samples the top N points by CLIP similarity, normalizes
    the similarity scores, and converts actions to tensor format.

    Args:
        pts: Point cloud positions [N, 3]
        clip_feats: CLIP features for each point [N, feat_dim]
        clip_sims: CLIP similarities for each point [N, 1] or [N]
        action_set: List of action poses (7D: xyz + quaternion)
        sample_num: Number of points to sample
        sample_action: Whether to filter actions by distance to sampled points
        device: Device to place tensors on

    Returns:
        Tuple of:
            - sampled_pts: [1, sample_num, 3]
            - sampled_clip_feats: [1, sample_num, feat_dim]
            - sampled_clip_sims: [1, sample_num, 1] (normalized to [0, 1])
            - actions: [1, n_actions, action_dim]
            - action_set: Filtered list of actions
    """
    # Convert to torch tensors if needed
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts).to(device)
    if isinstance(clip_feats, np.ndarray):
        clip_feats = torch.from_numpy(clip_feats).to(device)
    if isinstance(clip_sims, np.ndarray):
        clip_sims = torch.from_numpy(clip_sims).to(device)

    # Ensure clip_sims is 1D for sorting
    if clip_sims.dim() == 2:
        clip_sims_1d = clip_sims[..., 0]
    else:
        clip_sims_1d = clip_sims

    # Sample top N points by CLIP similarity
    sample_num = min(sample_num, pts.shape[0])
    sample_indices = torch.argsort(clip_sims_1d, descending=True)[:sample_num]

    sampled_pts = pts[sample_indices]
    sampled_clip_feats = clip_feats[sample_indices]
    sampled_clip_sims = clip_sims_1d[sample_indices][..., None]

    # Normalize CLIP similarities to [0, 1]
    z = 1e-4
    sampled_clip_sims = (sampled_clip_sims - sampled_clip_sims.min() + z) / (
        sampled_clip_sims.max() - sampled_clip_sims.min() + z
    )

    # Add batch dimension
    sampled_pts = sampled_pts.unsqueeze(0).to(dtype=torch.float32)
    sampled_clip_feats = sampled_clip_feats.unsqueeze(0).to(dtype=torch.float32)
    sampled_clip_sims = sampled_clip_sims.unsqueeze(0).to(dtype=torch.float32)

    # Convert actions to tensor
    actions = None
    for action in action_set:
        action_tensor = torch.from_numpy(action) if isinstance(action, np.ndarray) else action
        action_tensor = action_tensor.unsqueeze(0).to(device)
        if actions is None:
            actions = action_tensor
        else:
            actions = torch.cat((actions, action_tensor), dim=0)

    if actions is not None:
        actions = actions.unsqueeze(0).to(dtype=torch.float32)

        # Optionally filter actions by distance to sampled points
        if sample_action and actions.shape[1] > 1:
            pos_actions = actions[..., :3][0].unsqueeze(1)  # [n_actions, 1, 3]
            dist_map = torch.norm(pos_actions - sampled_pts, dim=-1)  # [n_actions, sample_num]
            min_dist = torch.min(dist_map, dim=1)[0]  # [n_actions]
            sampled_indices = torch.where(min_dist < 0.05)[0]
            if len(sampled_indices) > 0:
                actions = actions[:, sampled_indices, :]
                action_set = [action_set[i] for i in sampled_indices.cpu().numpy()]
    else:
        actions = torch.zeros((1, 0, 7), dtype=torch.float32, device=device)

    return sampled_pts, sampled_clip_feats, sampled_clip_sims, actions, action_set


def normalize_pos(
    pos: torch.Tensor,
    workspace_limits: np.ndarray = WORKSPACE_LIMITS,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Normalize positions to [-1, 1] range based on workspace limits.

    Args:
        pos: Position tensor [..., 3]
        workspace_limits: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        device: Device for tensors

    Returns:
        Normalized positions in [-1, 1] range
    """
    pos_min = torch.tensor(
        [workspace_limits[0][0], workspace_limits[1][0], workspace_limits[2][0]],
        dtype=torch.float32,
        device=device,
    )
    pos_max = torch.tensor(
        [workspace_limits[0][1], workspace_limits[1][1], workspace_limits[2][1]],
        dtype=torch.float32,
        device=device,
    )
    return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
