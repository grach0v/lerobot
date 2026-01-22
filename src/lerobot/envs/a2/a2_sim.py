"""A2 PyBullet simulation environment.

This module provides the Environment class for tabletop manipulation simulation
with UR5e robot and Robotiq gripper, supporting both train and test object sets.
"""

import datetime
import glob
import os
import time

# EGL setup for headless GPU rendering
# Only set PYOPENGL_PLATFORM - let PyBullet/EGL handle device selection
# Can be disabled via A2_DISABLE_EGL=true environment variable
if os.environ.get('A2_DISABLE_EGL', '').lower() != 'true':
    if 'PYOPENGL_PLATFORM' not in os.environ:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import pybullet as pb
import pybullet_data
from scipy.spatial.transform import Rotation as R

from . import a2_cameras as cameras
from . import a2_utils as utils
from .a2_constants import (
    # Train object constants
    ALL_LABEL,
    ALL_LABEL_DIR_MAP,
    COLOR_SHAPE,
    FUNCTION,
    GENERAL_LABEL,
    KEYWORD_DIR_MAP,
    LABEL,
    LABEL_DIR_MAP,
    LANG_TEMPLATES,
    PIXEL_SIZE,
    # Test object constants
    UNSEEN_COLOR_SHAPE,
    UNSEEN_FUNCTION,
    UNSEEN_GENERAL_LABEL,
    UNSEEN_KEYWORD_DIR_MAP,
    UNSEEN_KEYWORD_DIR_MAP_PLACE,
    UNSEEN_LABEL,
    UNSEEN_LABEL_DIR_MAP,
    WORKSPACE_LIMITS,
)


class Environment:
    """PyBullet simulation environment for A2 tabletop manipulation.

    Supports both train (simplified_objects) and test (unseen_objects) object sets.
    """

    # Camera name to index mapping
    CAMERA_INDICES = {
        'front': 0,
        'left': 1,
        'right': 2,
        'top': 3,        # new_front - top-down view
        'side_left': 4,  # new_left
        'side_right': 5, # new_right
        'overview': 6,   # overview - sees robot and scene
        'gripper': -1,   # special: dynamic camera attached to gripper
    }

    def __init__(self, gui=True, time_step=1 / 240, object_set="train"):
        """Creates environment with PyBullet.

        Args:
            gui: Show environment with PyBullet's built-in display viewer.
            time_step: PyBullet physics simulation step speed. Default is 1/240.
            object_set: "train" for simplified_objects, "test" for unseen_objects.
        """
        self.time_step = time_step
        self.gui = gui
        self.object_set = object_set
        self.pixel_size = PIXEL_SIZE
        self.case_dir = "cases/"
        os.makedirs(self.case_dir, exist_ok=True)  # Ensure case directory exists
        self.obj_ids = {"fixed": [], "rigid": []}
        # Initialize task relevant object lists
        self.target_obj_ids = []
        self.reference_obj_ids = []
        self.obj_labels = {}
        self.obj_dirs = {}
        self.test_file_name = None
        self.agent_cams = cameras.RealSenseL515.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.bounds = WORKSPACE_LIMITS
        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.drop_joints0 = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.drop_joints1 = np.array([1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        # Get the actual asset path (downloaded from HuggingFace)
        from .a2 import get_a2_assets_path
        self._assets_root = str(get_a2_assets_path())

        # Set object-set-specific constants
        if self.object_set == "train":
            self._LABEL = LABEL
            self._GENERAL_LABEL = GENERAL_LABEL
            self._COLOR_SHAPE = COLOR_SHAPE
            self._FUNCTION = FUNCTION
            self._LABEL_DIR_MAP = LABEL_DIR_MAP
            self._KEYWORD_DIR_MAP = KEYWORD_DIR_MAP
            self._asset_dir = os.path.join(self._assets_root, "simplified_objects")
        else:  # test/unseen
            self._LABEL = UNSEEN_LABEL
            self._GENERAL_LABEL = UNSEEN_GENERAL_LABEL
            self._COLOR_SHAPE = UNSEEN_COLOR_SHAPE
            self._FUNCTION = UNSEEN_FUNCTION
            self._LABEL_DIR_MAP = UNSEEN_LABEL_DIR_MAP
            self._KEYWORD_DIR_MAP = UNSEEN_KEYWORD_DIR_MAP
            self._asset_dir = os.path.join(self._assets_root, "unseen_objects")

        # Start PyBullet
        self._client_id = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(time_step)

        # Load EGL plugin for GPU rendering in headless mode
        self._egl_plugin = -1
        if not gui and os.environ.get('A2_DISABLE_EGL', '').lower() != 'true':
            import sys
            pybullet_dir = os.path.dirname(pb.__file__)
            # Build expected filename based on current Python version
            py_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
            egl_candidates = [
                os.path.join(pybullet_dir, f'eglRenderer.{py_version}-x86_64-linux-gnu.so'),
                os.path.join(pybullet_dir, 'eglRenderer.cpython-310-x86_64-linux-gnu.so'),
            ]
            for egl_path in egl_candidates:
                if os.path.exists(egl_path):
                    self._egl_plugin = pb.loadPlugin(egl_path, '_eglRendererPlugin')
                    if self._egl_plugin >= 0:
                        print(f"GPU rendering enabled via EGL (plugin {self._egl_plugin})")
                        break

        if gui:
            target = pb.getDebugVisualizerCamera()[11]
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target,
            )

        # VLA frame recording state
        self._frame_recorder = None
        self._recording_fps = 30
        self._recording_cameras = None  # None means all cameras
        self._recording_render_size = None
        self._steps_per_frame = int((1 / time_step) / 30)  # Default: 240/30 = 8
        self._step_counter = 0
        self._current_action = None
        self._recorded_frame_count = 0  # Total frames in current episode
        self._max_frames_per_episode = 500  # ~17 sec at 30fps
        self._baseline_joints = None  # For progress detection
        self._no_progress_count = 0  # Frames without significant progress

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(pb.getBaseVelocity(i, physicsClientId=self._client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""
        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = pb.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self._client_id
                )
                dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    def obj_info(self, obj_id):
        """Get object pose and dimensions."""
        pos, rot = pb.getBasePositionAndOrientation(
            obj_id, physicsClientId=self._client_id
        )
        dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
        return (pos, rot, dim)

    def _load_texture_for_object(self, body_id, mesh_file):
        """Load and apply texture for an object from its URDF path.

        Only applies to train objects (simplified_objects).
        """
        if self.object_set != "train":
            return  # Unseen objects don't have textures

        obj_dir = mesh_file.replace('.urdf', '')
        texture_candidates = [
            os.path.join(obj_dir, 'texture_map.png'),
            os.path.join(obj_dir, 'textured.png'),
            os.path.join(obj_dir, 'dummy.png'),
            os.path.join(obj_dir, 'texture_map.jpg'),
            os.path.join(obj_dir, 'textured.jpg'),
            os.path.join(obj_dir, 'textured_map.jpg'),
        ]

        texture_path = None
        for candidate in texture_candidates:
            if os.path.exists(candidate):
                texture_path = candidate
                break

        if texture_path is not None:
            try:
                texture_id = pb.loadTexture(texture_path, physicsClientId=self._client_id)
                if texture_id >= 0:
                    pb.changeVisualShape(
                        body_id, -1,
                        textureUniqueId=texture_id,
                        physicsClientId=self._client_id
                    )
            except Exception as e:
                print(f"Warning: Failed to load texture {texture_path}: {e}")

    def generate_lang_goal(self):
        """Generate a language goal for grasp task."""
        from operator import itemgetter

        prob = np.array([0.4, 0.2, 0.1, 0.1, 0.2])
        template_id = np.random.choice(a=range(len(LANG_TEMPLATES)), size=1, p=prob)[0]

        if template_id == 0:
            id = np.random.choice(range(len(self._LABEL)), 1)[0]
            keyword = self._LABEL[id]
            self.target_obj_dir = [self._LABEL_DIR_MAP[id]]
            self.target_obj_lst = self.target_obj_dir
        else:
            if template_id == 1:
                id = np.random.choice(range(len(self._GENERAL_LABEL)), 1)[0]
                keyword = self._GENERAL_LABEL[id]
                self.target_obj_dir = self._KEYWORD_DIR_MAP[keyword]
            elif template_id in [2, 3]:
                id = np.random.choice(range(len(self._COLOR_SHAPE)), 1)[0]
                keyword = self._COLOR_SHAPE[id]
                self.target_obj_dir = self._KEYWORD_DIR_MAP[keyword]
            elif template_id == 4:
                id = np.random.choice(range(len(self._FUNCTION)), 1)[0]
                keyword = self._FUNCTION[id]
                self.target_obj_dir = self._KEYWORD_DIR_MAP[keyword]

            if len(self.target_obj_dir) > 3:
                batch = np.random.choice(range(len(self.target_obj_dir)), 2, replace=False)
                self.target_obj_lst = list(itemgetter(*batch)(self.target_obj_dir))
            else:
                self.target_obj_lst = self.target_obj_dir

        self.lang_goal = LANG_TEMPLATES[template_id].format(keyword=keyword)
        pb.addUserDebugText(text=self.lang_goal, textPosition=[0.8, -0.2, 0], textColorRGB=[0, 0, 1], textSize=2)

        return self.lang_goal

    def generate_place_lang_goal(self, obj_bbox_ids, obj_bbox_centers, obj_bbox_sizes, pp_file=False):
        """Generate a language goal for place task."""
        if self.object_set == "train":
            self.lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, valid_mask = utils.generate_place_inst(
                self.obj_labels, self.obj_dirs, obj_bbox_ids, obj_bbox_centers, obj_bbox_sizes, self.pixel_size
            )
        else:
            self.lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, valid_mask = utils.generate_unseen_place_inst(
                self.obj_labels, self.obj_dirs, obj_bbox_ids, obj_bbox_centers, obj_bbox_sizes, self.pixel_size
            )

        self.reference_obj_ids = ref_obj_ids

        if self.lang_goal is not None:
            obj_name = self.obj_labels[ref_obj_ids[0]][0]
            dir_phrase = ref_regions[0]

            if not pp_file:
                with open(self.case_dir + self.test_file_name, "r") as file:
                    current_content = file.read()
                new_content = self.lang_goal + "\n" + obj_name + "\n" + dir_phrase + "\n" + current_content
                with open(self.case_dir + self.test_file_name, "w") as file:
                    file.write(new_content)
            else:
                with open(self.case_dir + self.test_file_name, "r") as file:
                    current_content = file.readlines()
                insert_line_number = 17
                current_content.insert(insert_line_number, self.lang_goal + "\n" + obj_name + "\n" + dir_phrase + "\n")
                with open(self.case_dir + self.test_file_name, "w") as file:
                    file.writelines(current_content)

        return self.lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, valid_mask

    def get_target_id(self):
        return self.target_obj_ids

    def get_reference_id(self):
        return self.reference_obj_ids

    def add_object_id(self, obj_id, category="rigid"):
        """Add object ID to tracking list."""
        self.obj_ids[category].append(obj_id)

    def remove_object_id(self, obj_id, category="rigid"):
        """Remove object ID from tracking list."""
        self.obj_ids[category].remove(obj_id)

        if obj_id in self.target_obj_ids:
            self.target_obj_ids.remove(obj_id)
        if obj_id in self.reference_obj_ids:
            self.reference_obj_ids.remove(obj_id)
        if obj_id in self.obj_labels:
            del self.obj_labels[obj_id]

        pb.removeBody(obj_id)

    def save_objects(self):
        """Save states of all rigid objects."""
        success = False
        while not success:
            success = self.wait_static()
        object_states = []
        for obj in self.obj_ids["rigid"]:
            pos, orn = pb.getBasePositionAndOrientation(obj)
            linVel, angVel = pb.getBaseVelocity(obj)
            object_states.append((pos, orn, linVel, angVel))
        return object_states

    def restore_objects(self, object_states):
        """Restore states of all rigid objects."""
        for idx, obj in enumerate(self.obj_ids["rigid"]):
            pos, orn, linVel, angVel = object_states[idx]
            pb.resetBasePositionAndOrientation(obj, pos, orn)
            pb.resetBaseVelocity(obj, linVel, angVel)
        success = self.wait_static()
        return success

    def wait_static(self, timeout=3):
        """Step simulator asynchronously until objects settle."""
        pb.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            pb.stepSimulation()
        print(f"Warning: Wait static exceeded {timeout} second timeout. Skipping.")
        return False

    def reset(self, workspace="raw"):
        """Reset the simulation environment."""
        self.obj_ids = {"fixed": [], "rigid": []}
        self.target_obj_ids = []
        self.reference_obj_ids = []
        self.obj_labels = {}
        self.obj_dirs = {}

        # Temporarily disable VLA recording during reset
        saved_recorder = self._frame_recorder
        self._frame_recorder = None

        pb.resetSimulation()
        pb.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster
        if self.gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        # Load workspace
        self.plane = pb.loadURDF(
            "plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True,
        )
        if workspace == "raw":
            workspace_urdf = os.path.join(self._assets_root, "workspace/workspace.urdf")
            self.workspace = pb.loadURDF(
                workspace_urdf, basePosition=(0.5, 0, 0), useFixedBase=True,
            )
        elif workspace == "extend":
            workspace_urdf = os.path.join(self._assets_root, "workspace/pp_workspace.urdf")
            self.workspace = pb.loadURDF(
                workspace_urdf, basePosition=(0.5, 0, 0), useFixedBase=True,
            )

        pb.changeDynamics(
            self.plane, -1,
            lateralFriction=1.1, restitution=0.5, linearDamping=0.5, angularDamping=0.5,
        )
        pb.changeDynamics(
            self.workspace, -1,
            lateralFriction=1.1, restitution=0.5, linearDamping=0.5, angularDamping=0.5,
        )

        # Load UR5e
        ur5e_urdf = os.path.join(self._assets_root, "ur5e/ur5e.urdf")
        self.ur5e = pb.loadURDF(
            ur5e_urdf, basePosition=(0, 0, 0), useFixedBase=True,
        )
        self.ur5e_joints = []
        for i in range(pb.getNumJoints(self.ur5e)):
            info = pb.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)
        pb.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)

        self.setup_gripper()

        # Move robot to home joint configuration
        success = self.go_home()

        # Note: get eelink pose, NOT tcp pose
        home_pos, home_rot = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.initial_ee_pose = np.hstack((np.array(home_pos), np.array(home_rot)))

        self.close_gripper()
        self.open_gripper()

        if not success:
            print("Simulation is wrong!")
            exit()

        # Re-enable rendering
        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )

        # Restore VLA recording after reset is complete
        self._frame_recorder = saved_recorder

    def setup_gripper(self):
        """Load end-effector: gripper"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        gripper_urdf = os.path.join(self._assets_root, "ur5e/gripper/robotiq_2f_85.urdf")
        self.ee = pb.loadURDF(
            gripper_urdf,
            ee_position,
            pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self.ee_tip_z_offset = 0.1625
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.73
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(pb.getNumJoints(self.ee)):
            info = pb.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif "finger_pad_joint" in joint_name:
                pb.changeDynamics(self.ee, joint_id, lateralFriction=0.9)
                self.ee_finger_pad_id = joint_id
            elif joint_type == pb.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                pb.setJointMotorControl2(
                    self.ee, joint_id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0,
                )

        self.ee_constraint = pb.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(self.ee_constraint, maxForce=10000)
        pb.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper
        self._setup_gripper_mimic_joints()

    def _setup_gripper_mimic_joints(self):
        """Set up mimic joint constraints for Robotiq gripper."""
        # Left side
        c = pb.createConstraint(
            self.ee, self.gripper_main_joint,
            self.ee, self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=pb.JOINT_GEAR, jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)

        c = pb.createConstraint(
            self.ee, self.gripper_main_joint,
            self.ee, self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR, jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)

        # Right side
        c = pb.createConstraint(
            self.ee, self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee, self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=pb.JOINT_GEAR, jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)

        c = pb.createConstraint(
            self.ee, self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee, self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR, jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)

        # Connect left and right
        c = pb.createConstraint(
            self.ee, self.gripper_main_joint,
            self.ee, self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR, jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)

    def step(self, pose=None):
        """Execute action with specified primitive."""
        done = False
        if pose is not None:
            success, grasped_obj_id, pos_dist = self.grasp(pose)
            if grasped_obj_id in self.target_obj_ids:
                reward = 1
                done = True
            else:
                max_pos_dist = np.sqrt(
                    (self.bounds[0][1]-self.bounds[0][0]) ** 2 +
                    (self.bounds[1][1]-self.bounds[1][0]) ** 2
                )
                reward = - pos_dist / max_pos_dist

        # Step simulator until objects settle
        while not self.is_static:
            pb.stepSimulation()

        return reward, done

    def step_place(self, place_pose, grasp_pose=None):
        """Execute grasp-then-place for VLA recording."""
        graspable_ids = [oid for oid in self.obj_ids["rigid"]
                       if oid not in getattr(self, 'reference_obj_ids', [])]
        if not graspable_ids:
            print("Warning: No graspable objects available")
            return None, None

        # Disable recording during grasp phase
        saved_recorder = self._frame_recorder
        self._frame_recorder = None
        original_target_ids = self.target_obj_ids

        grasp_success = False
        grasped_obj_id = None

        if grasp_pose is not None:
            self.target_obj_ids = graspable_ids
            grasp_success, grasped_obj_id, _ = self.grasp(grasp_pose, follow_place=True, detect_force=False)
            if grasp_success and grasped_obj_id is not None:
                print(f"Successfully grasped object {grasped_obj_id} using graspnet pose")
        else:
            np.random.shuffle(graspable_ids)
            for obj_id in graspable_ids[:5]:
                obj_pos, obj_rot, obj_dim = self.obj_info(obj_id)
                self.target_obj_ids = [obj_id]

                grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])
                aabb_min, aabb_max = pb.getAABB(obj_id, physicsClientId=self._client_id)
                obj_center_z = (aabb_min[2] + aabb_max[2]) / 2
                grasp_z = obj_center_z + 0.02
                grasp_pos = np.array([obj_pos[0], obj_pos[1], grasp_z])
                simple_grasp_pose = np.concatenate([grasp_pos, grasp_quat])

                grasp_success, grasped_obj_id, _ = self.grasp(simple_grasp_pose, follow_place=True, detect_force=False)

                if grasp_success and grasped_obj_id is not None:
                    print(f"Successfully grasped object {grasped_obj_id}")
                    break

                self.open_gripper()
                self.go_home()

        self.target_obj_ids = original_target_ids

        if not grasp_success or grasped_obj_id is None:
            print("Warning: Grasp failed, skipping this episode")
            self._frame_recorder = saved_recorder
            self.go_home()
            return None, None

        print(f"Grasped object {grasped_obj_id}, now recording place...")

        # Re-enable recording for place phase
        self._frame_recorder = saved_recorder
        self.reset_recording_state()

        place_success = self.place(place_pose)

        if place_success and grasped_obj_id is not None:
            for _ in range(50):
                pb.stepSimulation(physicsClientId=self._client_id)

            target_xy = place_pose[:2]
            obj_pos, _, _ = self.obj_info(grasped_obj_id)
            obj_xy = np.array(obj_pos[:2])
            distance = np.linalg.norm(obj_xy - target_xy)

            actual_success = distance < 0.05
            print(f"  Place verification: dist={distance:.3f}m, success={actual_success}")

            if actual_success:
                return 2, True
            else:
                return 0, False
        else:
            return 0, False

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    # ========== VLA Frame Recording Methods ==========

    def set_frame_recorder(self, recorder_callback, fps=30, cameras=None, render_size=None):
        """Set callback for VLA frame recording during robot movements."""
        self._frame_recorder = recorder_callback
        self._recording_fps = fps
        self._recording_cameras = cameras
        self._recording_render_size = render_size
        self._steps_per_frame = max(1, int((1 / self.time_step) / fps))
        self._step_counter = 0
        self._recorded_frame_count = 0
        self._baseline_joints = None
        self._no_progress_count = 0

    def clear_frame_recorder(self):
        """Clear the frame recorder callback."""
        self._frame_recorder = None
        self._current_action = None
        self._step_counter = 0

    def reset_recording_state(self):
        """Reset recording state for a new episode."""
        self._step_counter = 0
        self._recorded_frame_count = 0
        self._baseline_joints = None
        self._no_progress_count = 0

    def set_current_action(self, action):
        """Set the current action being executed."""
        self._current_action = np.array(action, dtype=np.float32)

    def get_robot_state(self):
        """Get current robot state for recording."""
        ee_pos, ee_quat = self.get_link_pose(self.ur5e, self.ur5e_ee_id)

        joints = np.array([
            pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
            for i in self.ur5e_joints
        ], dtype=np.float32)

        gripper_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]

        return {
            'ee_pos': np.array(ee_pos, dtype=np.float32),
            'ee_quat': np.array(ee_quat, dtype=np.float32),
            'joints': joints,
            'gripper_angle': float(gripper_angle)
        }

    def _get_gripper_camera_config(self):
        """Get camera config for wrist-mounted camera."""
        ee_pos, ee_quat = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        ee_rotm = np.array(pb.getMatrixFromQuaternion(ee_quat)).reshape(3, 3)
        ee_tip_pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)

        ee_x = ee_rotm @ np.array([1, 0, 0])
        ee_z = ee_rotm @ np.array([0, 0, 1])

        camera_pos = np.array(ee_pos) + 0.08 * ee_x - 0.02 * ee_z
        target = np.array(ee_tip_pos)
        target[2] -= 0.03

        look_dir = target - camera_pos
        look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)

        world_up = np.array([0, 0, 1])
        if abs(np.dot(look_dir, world_up)) > 0.95:
            world_up = ee_x

        right = np.cross(world_up, look_dir)
        right = right / (np.linalg.norm(right) + 1e-8)
        cam_up = np.cross(look_dir, right)
        cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-8)

        rot_mat = np.column_stack([right, cam_up, look_dir])
        camera_quat = R.from_matrix(rot_mat).as_quat()

        return {
            "image_size": (480, 640),
            "intrinsics": np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]]),
            "position": tuple(camera_pos),
            "rotation": tuple(camera_quat),
            "zrange": (0.01, 2.0),
            "noise": False,
        }

    def render_camera_fast(self, config, render_size):
        """Render camera at lower resolution for speed."""
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        znear, zfar = config["zrange"]

        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        aspect = render_size[1] / render_size[0]
        projm = pb.computeProjectionMatrixFOV(60, aspect, znear, zfar)

        _, _, color, _, _ = pb.getCameraImage(
            width=render_size[1],
            height=render_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pb.ER_NO_SEGMENTATION_MASK,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        color = np.array(color, dtype=np.uint8).reshape(render_size[0], render_size[1], 4)
        return color[:, :, :3]

    def get_camera_images(self, cameras=None, render_size=None):
        """Get RGB images from RealSense cameras."""
        images = {}
        default_cams = ['front', 'left', 'right']
        cam_names = cameras if cameras else default_cams

        if render_size is None:
            render_size = getattr(self, '_recording_render_size', None)

        for name in cam_names:
            if name not in self.CAMERA_INDICES:
                raise ValueError(f"Unknown camera '{name}'. Available: {list(self.CAMERA_INDICES.keys())}")

            i = self.CAMERA_INDICES[name]

            if name == 'gripper':
                config = self._get_gripper_camera_config()
            else:
                config = self.agent_cams[i]

            if render_size is not None:
                color = self.render_camera_fast(config, render_size)
            else:
                color, _, _ = self.render_camera(config)
            images[name] = color
        return images

    def _maybe_record_frame(self):
        """Check if we should record a frame and do so if needed."""
        if self._frame_recorder is None:
            return

        if self._recorded_frame_count >= self._max_frames_per_episode:
            return

        self._step_counter += 1
        if self._step_counter >= self._steps_per_frame:
            self._step_counter = 0

            current_joints = np.array([
                pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                for i in self.ur5e_joints
            ])

            if self._baseline_joints is None:
                self._baseline_joints = current_joints.copy()
            else:
                self._no_progress_count += 1
                if self._no_progress_count >= 30:
                    joint_diff = np.max(np.abs(current_joints - self._baseline_joints))
                    if joint_diff < 0.05:
                        return
                    self._baseline_joints = current_joints.copy()
                    self._no_progress_count = 0

            self._recorded_frame_count += 1
            images = self.get_camera_images(self._recording_cameras)
            robot_state = self.get_robot_state()
            self._frame_recorder(images, robot_state)

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        _, _, color, depth, segm = pb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def get_multi_view_pointcloud(self, cameras=None, downsample=True, voxel_size=0.0015):
        """Get fused point cloud from multiple cameras.

        Args:
            cameras: List of camera indices or None for first 3 cameras
            downsample: Whether to voxel downsample the point cloud
            voxel_size: Voxel size for downsampling

        Returns:
            pcd: Open3D PointCloud with points and colors
            colors: Dict of camera_name -> RGB image
            depths: Dict of camera_name -> depth image
        """
        import open3d as o3d

        if cameras is None:
            cameras = [0, 1, 2]  # front, left, right

        pcds = []
        colors_dict = {}
        depths_dict = {}

        cam_names = ['front', 'left', 'right', 'new_front', 'new_left', 'new_right', 'overview']

        for cam_idx in cameras:
            if cam_idx >= len(self.agent_cams):
                continue

            config = self.agent_cams[cam_idx]
            color, depth, _ = self.render_camera(config)
            cam_name = cam_names[cam_idx] if cam_idx < len(cam_names) else f'cam{cam_idx}'
            colors_dict[cam_name] = color
            depths_dict[cam_name] = depth

            # Generate point cloud from depth using camera intrinsics
            K = config["intrinsics"]
            H, W = depth.shape

            # Valid depth mask
            mask = depth > 0.01

            # Create pixel coordinates
            xlin = np.linspace(0, W - 1, W)
            ylin = np.linspace(0, H - 1, H)
            px, py = np.meshgrid(xlin, ylin)
            px = px[mask]
            py = py[mask]
            pz = depth[mask]

            # Unproject to camera frame
            x = (px - K[0, 2]) * pz / K[0, 0]
            y = (py - K[1, 2]) * pz / K[1, 1]
            points_cam = np.stack([x, y, pz], axis=1)  # (N, 3)

            # Get colors for valid points
            color_normalized = color.astype(np.float64) / 255.0
            colors_cam = color_normalized[mask]

            # Compute camera extrinsics (world-to-camera transform)
            position = np.array(config["position"]).reshape(3, 1)
            rotation = pb.getMatrixFromQuaternion(config["rotation"])
            rotation = np.array(rotation).reshape(3, 3)

            # Build 4x4 transform matrix [R|t; 0 0 0 1]
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3:] = position

            # camera_extrinsics is the world-to-camera transform (inverted)
            camera_extrinsics = np.linalg.inv(transform)

            # To transform camera coords to world coords, invert again
            cam_to_world = np.linalg.inv(camera_extrinsics)

            # Transform points from camera to world coordinates
            points_homo = np.concatenate([points_cam.T, np.ones((1, points_cam.shape[0]))], axis=0)
            points_world = (cam_to_world @ points_homo)[:3, :].T

            # Clip to workspace bounds
            bounds_mask = (
                (points_world[:, 0] > self.bounds[0, 0]) &
                (points_world[:, 0] < self.bounds[0, 1]) &
                (points_world[:, 1] > self.bounds[1, 0]) &
                (points_world[:, 1] < self.bounds[1, 1]) &
                (points_world[:, 2] > self.bounds[2, 0]) &
                (points_world[:, 2] < self.bounds[2, 1])
            )

            points_clipped = points_world[bounds_mask]
            colors_clipped = colors_cam[bounds_mask]

            if len(points_clipped) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_clipped)
                pcd.colors = o3d.utility.Vector3dVector(colors_clipped)
                pcds.append(pcd)

        # Merge point clouds
        if len(pcds) == 0:
            merged_pcd = o3d.geometry.PointCloud()
        else:
            merged_pcd = pcds[0]
            for pcd in pcds[1:]:
                merged_pcd += pcd

            # Downsample
            if downsample and len(merged_pcd.points) > 0:
                merged_pcd = merged_pcd.voxel_down_sample(voxel_size)

        return merged_pcd, colors_dict, depths_dict

    def get_pointcloud_array(self, cameras=None, max_points=20000):
        """Get point cloud as numpy array.

        Args:
            cameras: List of camera indices
            max_points: Maximum number of points to return

        Returns:
            points: (N, 6) array with xyz and rgb
        """
        pcd, _, _ = self.get_multi_view_pointcloud(cameras)

        if len(pcd.points) == 0:
            return np.zeros((0, 6))

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255.0  # Convert to 0-255 range

        # Combine points and colors
        point_cloud = np.concatenate([points, colors], axis=1)

        # Subsample if needed
        if len(point_cloud) > max_points:
            indices = np.random.choice(len(point_cloud), max_points, replace=False)
            point_cloud = point_cloud[indices]

        return point_cloud.astype(np.float32)

    def close(self):
        """Close the environment and disconnect from PyBullet."""
        if self._client_id is not None and pb is not None:
            try:
                # Unload EGL plugin first
                if hasattr(self, '_egl_plugin') and self._egl_plugin >= 0:
                    try:
                        pb.unloadPlugin(self._egl_plugin, physicsClientId=self._client_id)
                    except Exception:
                        pass
                    self._egl_plugin = -1
                # Then disconnect
                if hasattr(pb, 'disconnect'):
                    pb.disconnect(physicsClientId=self._client_id)
            except Exception:
                pass
            finally:
                self._client_id = None

    def __del__(self):
        self.close()

    def get_link_pose(self, body, link):
        result = pb.getLinkState(body, link)
        return result[4], result[5]

    def add_objects(self, num_obj, workspace_limits, test_data=False):
        """Randomly drop objects to the workspace for grasp task."""
        self.num_obj = num_obj
        mesh_list = glob.glob(f"{self._asset_dir}/*.urdf")

        target_mesh_list = []
        for target_obj in self.target_obj_lst:
            target_mesh_file = f"{self._asset_dir}/{target_obj}.urdf"
            target_mesh_list.append(target_mesh_file)
        for obj in self.target_obj_dir:
            obj_mesh_file = f"{self._asset_dir}/{obj}.urdf"
            if obj_mesh_file in mesh_list:
                mesh_list.remove(obj_mesh_file)

        obj_mesh_ind = np.random.randint(0, len(mesh_list), size=num_obj-len(self.target_obj_lst))

        body_ids = []
        self.target_obj_ids = []

        if test_data:
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            self.test_file_name = timestamp_value.strftime("%Y-%m-%d-%H-%M-%S") + ".txt"
        else:
            self.test_file_name = "temp.txt"

        with open(self.case_dir + self.test_file_name, "w") as out_file:
            out_file.write("%s\n" % self.lang_goal)
            for i in range(len(target_mesh_list)):
                out_file.write(f"{i} ")
            out_file.write("\n")

            for idx, target_mesh_file in enumerate(target_mesh_list):
                drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
                drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
                object_position = [drop_x, drop_y, 0.2]
                object_orientation = [2 * np.pi * np.random.random_sample() for _ in range(3)]
                body_id = pb.loadURDF(
                    target_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.target_obj_ids.append(body_id)
                self.add_object_id(body_id)
                self._load_texture_for_object(body_id, target_mesh_file)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (target_mesh_file, *object_position, *object_orientation)
                )

            for object_idx in range(len(obj_mesh_ind)):
                curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
                drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
                drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
                object_position = [drop_x, drop_y, 0.2]
                object_orientation = [2 * np.pi * np.random.random_sample() for _ in range(3)]
                body_id = pb.loadURDF(
                    curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.add_object_id(body_id)
                self._load_texture_for_object(body_id, curr_mesh_file)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (curr_mesh_file, *object_position, *object_orientation)
                )

        return body_ids, True

    def add_objects_for_place(self, num_obj, workspace_limits, test_data=False):
        """Randomly drop objects to the workspace for place task."""
        self.num_obj = num_obj
        mesh_list = glob.glob(f"{self._asset_dir}/*.urdf")

        obj_mesh_ind = np.random.randint(0, len(mesh_list), size=num_obj)

        body_ids = []
        self.obj_labels = {}
        self.obj_dirs = {}

        if test_data:
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            self.test_file_name = timestamp_value.strftime("%Y-%m-%d-%H-%M-%S") + ".txt"
        else:
            self.test_file_name = "temp.txt"

        drop_xys = utils.generate_drop_positions(num_obj, workspace_limits, min_dist=0.05)
        if len(drop_xys) < num_obj:
            return body_ids, False

        with open(self.case_dir + self.test_file_name, "a") as out_file:
            for object_idx in range(len(obj_mesh_ind)):
                curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
                curr_dir_label = curr_mesh_file.split('/')[-1].split('.')[0]

                drop_x, drop_y = drop_xys[object_idx]
                object_position = [drop_x, drop_y, 0.1]
                object_orientation = [2 * np.pi * np.random.random_sample() for _ in range(3)]

                body_id = pb.loadURDF(
                    curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.add_object_id(body_id)
                self._load_texture_for_object(body_id, curr_mesh_file)
                self.wait_static()

                self.obj_dirs[body_id] = curr_dir_label
                self.obj_labels[body_id] = []

                # Assign labels based on object set
                if curr_dir_label in ALL_LABEL_DIR_MAP:
                    curr_ind = ALL_LABEL_DIR_MAP.index(curr_dir_label)
                    curr_obj_label = ALL_LABEL[curr_ind]
                    self.obj_labels[body_id].append(curr_obj_label)

                for key in self._KEYWORD_DIR_MAP:
                    if curr_dir_label in self._KEYWORD_DIR_MAP[key]:
                        self.obj_labels[body_id].append(key)
                        break

                if len(self.obj_labels[body_id]) == 0:
                    self.obj_labels.pop(body_id)

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (curr_mesh_file, *object_position, *object_orientation)
                )

        return body_ids, True

    def add_object_push_from_file(self, file_name, switch=None):
        """Load objects from a grasp test case file.

        File format:
            Line 1: Language goal
            Line 2: Target object index (space-separated if multiple)
            Line 3+: urdf_path x y z euler_x euler_y euler_z

        Args:
            file_name: Path to test case file
            switch: Unused, for compatibility

        Returns:
            Tuple of (success, lang_goal)
        """
        success = True
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            self.lang_goal = file_content[0].strip()
            target_obj = file_content[1].split()
            self.target_obj_ids = [4 + int(i) for i in target_obj]
            num_obj = len(file_content) - 2
            obj_files = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx + 2].split()
                obj_files.append(file_content_curr_object[0])
                obj_positions.append([
                    float(file_content_curr_object[1]),
                    float(file_content_curr_object[2]),
                    float(file_content_curr_object[3]),
                ])
                obj_orientations.append([
                    float(file_content_curr_object[4]),
                    float(file_content_curr_object[5]),
                    float(file_content_curr_object[6]),
                ])

        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            # Resolve asset path - test cases use paths like "assets/simplified_objects/015.urdf"
            # but we need the actual downloaded asset path
            if curr_mesh_file.startswith("assets/"):
                # Replace "assets/" with our actual assets root
                curr_mesh_file = os.path.join(self._assets_root, curr_mesh_file[7:])
            object_position = obj_positions[object_idx]
            object_orientation = obj_orientations[object_idx]
            body_id = pb.loadURDF(
                curr_mesh_file,
                object_position,
                pb.getQuaternionFromEuler(object_orientation),
                flags=pb.URDF_ENABLE_SLEEPING
            )
            self.add_object_id(body_id)
            success &= self.wait_static()
            success &= self.wait_static()

        # Give time to stop
        for _ in range(5):
            pb.stepSimulation()

        return success, self.lang_goal

    def add_object_push_from_place_file(self, file_name, switch=None, variance=False):
        """Load objects from a place test case file.

        File format:
            Line 1: Language goal
            Line 2: Target object label
            Line 3: Target direction
            Line 4+: urdf_path x y z euler_x euler_y euler_z

        Args:
            file_name: Path to test case file
            switch: Unused, for compatibility
            variance: Add random position variance

        Returns:
            Tuple of (success, lang_goal)
        """
        success = True
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            self.lang_goal = file_content[0].strip()
            target_obj_label = file_content[1].strip()
            target_dir = file_content[2].strip()
            num_obj = len(file_content) - 3
            obj_files = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx + 3].split()
                obj_files.append(file_content_curr_object[0])
                if not variance:
                    obj_positions.append([
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ])
                else:
                    obj_positions.append([
                        float(file_content_curr_object[1]) + 0.05 * np.random.uniform(-1, 1),
                        float(file_content_curr_object[2]) + 0.05 * np.random.uniform(-1, 1),
                        float(file_content_curr_object[3]),
                    ])
                obj_orientations.append([
                    float(file_content_curr_object[4]),
                    float(file_content_curr_object[5]),
                    float(file_content_curr_object[6]),
                ])

        # Import objects
        self.obj_labels = {}
        self.reference_obj_labels = []
        self.reference_obj_ids = []
        self.reference_obj_dirs = []
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            # Resolve asset path
            if curr_mesh_file.startswith("assets/"):
                curr_mesh_file = os.path.join(self._assets_root, curr_mesh_file[7:])
            object_position = obj_positions[object_idx]
            object_orientation = obj_orientations[object_idx]
            body_id = pb.loadURDF(
                curr_mesh_file,
                object_position,
                pb.getQuaternionFromEuler(object_orientation),
                flags=pb.URDF_ENABLE_SLEEPING
            )
            self.add_object_id(body_id)
            success &= self.wait_static()
            success &= self.wait_static()

            self.obj_labels[body_id] = []

            # Check if the object is target
            curr_dir_label = curr_mesh_file.split('/')[-1].split('.')[0]
            # For objects that have unique meaningful labels
            if curr_dir_label in ALL_LABEL_DIR_MAP:
                curr_ind = ALL_LABEL_DIR_MAP.index(curr_dir_label)
                curr_obj_label = ALL_LABEL[curr_ind]
                self.obj_labels[body_id].append(curr_obj_label)

                if curr_obj_label == target_obj_label:
                    self.reference_obj_ids.append(body_id)
                    self.reference_obj_dirs.append(target_dir)
                    self.reference_obj_labels.append(target_obj_label)

            # For objects that have general meaningful labels
            for key in self._KEYWORD_DIR_MAP.keys():
                if curr_dir_label in self._KEYWORD_DIR_MAP[key]:
                    self.obj_labels[body_id].append(key)
                    if len(self.obj_labels[body_id]) == 1 and key == target_obj_label:
                        self.reference_obj_ids.append(body_id)
                        self.reference_obj_dirs.append(target_dir)
                        self.reference_obj_labels.append(target_obj_label)
                    break

            if len(self.obj_labels[body_id]) == 0:
                self.obj_labels.pop(body_id)

        # Give time to stop
        for _ in range(5):
            pb.stepSimulation()

        return success, self.lang_goal

    def add_object_push_from_pickplace_file(self, file_name, mode="grasp"):
        """Load objects from a pick-and-place test case file.

        File format:
            Lines 1-2: Grasp language goal + target object index
            Lines 3-17: Grasp scene objects (15 objects)
            Lines 18-20: Place language goal + target label + direction
            Lines 21-28: Place scene objects (8 objects)

        Args:
            file_name: Path to test case file
            mode: "grasp" or "place"

        Returns:
            Tuple of (success, grasp_lang_goal, place_lang_goal)
        """
        success = True
        grasp_lang_goal = None
        place_lang_goal = None

        if mode == "grasp":
            with open(file_name, "r") as preset_file:
                file_content = preset_file.readlines()
                grasp_lang_goal = file_content[0].strip()
                target_obj = file_content[1].split()
                self.target_obj_ids = [4 + int(i) for i in target_obj]
                grasp_num_obj = 15
                obj_files = []
                obj_positions = []
                obj_orientations = []
                for object_idx in range(grasp_num_obj):
                    file_content_curr_object = file_content[object_idx + 2].split()
                    obj_files.append(file_content_curr_object[0])
                    obj_positions.append([
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ])
                    obj_orientations.append([
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ])

            # Import objects
            for object_idx in range(grasp_num_obj):
                curr_mesh_file = obj_files[object_idx]
                # Resolve asset path
                if curr_mesh_file.startswith("assets/"):
                    curr_mesh_file = os.path.join(self._assets_root, curr_mesh_file[7:])
                object_position = obj_positions[object_idx]
                object_orientation = obj_orientations[object_idx]
                body_id = pb.loadURDF(
                    curr_mesh_file,
                    object_position,
                    pb.getQuaternionFromEuler(object_orientation),
                    flags=pb.URDF_ENABLE_SLEEPING
                )
                self.add_object_id(body_id)
                success &= self.wait_static()
                success &= self.wait_static()

            for _ in range(5):
                pb.stepSimulation()

        elif mode == "place":
            with open(file_name, "r") as preset_file:
                file_content = preset_file.readlines()
                place_lang_goal = file_content[17].strip()
                target_obj_label = file_content[18].strip()
                target_dir = file_content[19].strip()
                place_num_obj = 8
                obj_files = []
                obj_positions = []
                obj_orientations = []
                for object_idx in range(place_num_obj):
                    file_content_curr_object = file_content[object_idx + 20].split()
                    obj_files.append(file_content_curr_object[0])
                    obj_positions.append([
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ])
                    obj_orientations.append([
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ])

            # Import objects
            self.obj_labels = {}
            self.reference_obj_labels = []
            self.reference_obj_ids = []
            self.reference_obj_dirs = []
            for object_idx in range(place_num_obj):
                curr_mesh_file = obj_files[object_idx]
                # Resolve asset path
                if curr_mesh_file.startswith("assets/"):
                    curr_mesh_file = os.path.join(self._assets_root, curr_mesh_file[7:])
                object_position = obj_positions[object_idx]
                object_orientation = obj_orientations[object_idx]
                body_id = pb.loadURDF(
                    curr_mesh_file,
                    object_position,
                    pb.getQuaternionFromEuler(object_orientation),
                    flags=pb.URDF_ENABLE_SLEEPING
                )
                self.add_object_id(body_id)
                success &= self.wait_static()
                success &= self.wait_static()

                self.obj_labels[body_id] = []

                # Check if the object is target
                curr_dir_label = curr_mesh_file.split('/')[-1].split('.')[0]
                # For objects that have unique meaningful labels
                if curr_dir_label in ALL_LABEL_DIR_MAP:
                    curr_ind = ALL_LABEL_DIR_MAP.index(curr_dir_label)
                    curr_obj_label = ALL_LABEL[curr_ind]
                    self.obj_labels[body_id].append(curr_obj_label)

                    if curr_obj_label == target_obj_label:
                        self.reference_obj_ids.append(body_id)
                        self.reference_obj_dirs.append(target_dir)
                        self.reference_obj_labels.append(target_obj_label)

                # For objects that have general meaningful labels
                for key in self._KEYWORD_DIR_MAP.keys():
                    if curr_dir_label in self._KEYWORD_DIR_MAP[key]:
                        self.obj_labels[body_id].append(key)
                        if len(self.obj_labels[body_id]) == 1 and key == target_obj_label:
                            self.reference_obj_ids.append(body_id)
                            self.reference_obj_dirs.append(target_dir)
                            self.reference_obj_labels.append(target_obj_label)
                        break

                if len(self.obj_labels[body_id]) == 0:
                    self.obj_labels.pop(body_id)

            for _ in range(5):
                pb.stepSimulation()

        return success, grasp_lang_goal, place_lang_goal

    def get_target_object_poses(self):
        transforms = {}
        for obj_id in self.target_obj_ids:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms

    def get_reference_object_poses(self):
        transforms = {}
        for obj_id in self.reference_obj_ids:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms

    def get_true_object_pose(self, obj_id):
        pos, ort = pb.getBasePositionAndOrientation(obj_id)
        position = np.array(pos).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(ort)
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        return transform

    def get_true_object_poses(self):
        transforms = {}
        for obj_id in self.obj_ids["rigid"]:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self):
        return self.move_joints(self.home_joints)

    def move_joints(self, target_joints, speed=0.01, timeout=3):
        """Move UR5e to target joint configuration."""
        # Longer timeout when recording (3x) and even longer with CPU rendering (10x)
        cpu_render = os.environ.get('A2_DISABLE_EGL', '').lower() == 'true'
        if self._frame_recorder is not None:
            effective_timeout = timeout * 10 if cpu_render else timeout * 3
        else:
            effective_timeout = timeout * 3 if cpu_render else timeout
        t0 = time.time()
        while (time.time() - t0) < effective_timeout:
            current_joints = np.array([
                pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                for i in self.ur5e_joints
            ])
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)

            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False

            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 0.05):
                for _ in range(5):
                    pb.stepSimulation()
                    self._maybe_record_frame()
                return True

            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            pb.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            pb.stepSimulation()
            self._maybe_record_frame()

        print(f"Warning: move_joints exceeded {effective_timeout} second timeout. Skipping.")
        return False

    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose (blocking)."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)

    def step_ee_pose(self, pose, num_sim_steps=8):
        """Single VLA-style step toward target EE pose (non-blocking).

        This method sets motor targets and steps simulation a fixed number of times.
        Used for 30Hz VLA control where we want incremental motion without blocking.

        Args:
            pose: Target (position, quaternion) tuple
            num_sim_steps: Number of physics steps (at 240Hz, 8 steps ≈ 30Hz control)

        Returns:
            True if step executed successfully
        """
        target_joints = self.solve_ik(pose)

        # Set motor targets
        pb.setJointMotorControlArray(
            bodyIndex=self.ur5e,
            jointIndices=self.ur5e_joints,
            controlMode=pb.POSITION_CONTROL,
            targetPositions=target_joints.tolist(),
            positionGains=np.ones(len(self.ur5e_joints)) * 0.5,
            velocityGains=np.ones(len(self.ur5e_joints)) * 0.1,
        )

        # Step simulation
        for _ in range(num_sim_steps):
            pb.stepSimulation(physicsClientId=self._client_id)
            self._maybe_record_frame()

        return True

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = pb.calculateInverseKinematics(
            bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ur5e_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
            restPoses=np.float32(self.ik_rest_joints).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        return joints

    def straight_move(self, pose0, pose1, rot, speed=0.01, max_force=300, detect_force=False, is_push=False):
        """Move in a straight line between two poses."""
        step_distance = 0.01
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance))
        success = True

        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if detect_force:
                force = np.sum(np.abs(np.array(pb.getJointState(self.ur5e, self.ur5e_ee_id)[2])))
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False

        if is_push:
            speed /= 5
        success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def grasp(self, pose, speed=0.005, follow_place=False, detect_force=True):
        """Execute grasping primitive."""
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        pose = np.array(pose, dtype=np.float32)
        rot = pose[-4:]
        pos = pose[:3]
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = pos

        ee_tip_transform = np.array([
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, -self.ee_tip_z_offset],
            [0, 0, 0, 1]
        ])

        ee_transform = transform @ ee_tip_transform
        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        grasped_obj_id = None
        min_pos_dist = None
        self.open_gripper()
        success = self.move_joints(self.ik_rest_joints)

        if success:
            success = self.move_ee_pose((over, rot))
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=detect_force)
        if success:
            self.close_gripper()
            success = self.straight_move(pos, over, rot, speed)
            success &= self.is_gripper_closed

            if success:
                max_height = -0.0001
                for i in self.obj_ids["rigid"]:
                    height = self.info[i][0][2]
                    if height >= max_height:
                        grasped_obj_id = i
                        max_height = height

                pos_dists = []
                for target_obj_id in self.target_obj_ids:
                    pos_dist = np.linalg.norm(
                        np.array(self.info[grasped_obj_id][0][:2]) -
                        np.array(self.info[target_obj_id][0][:2])
                    )
                    pos_dists.append(pos_dist)
                if pos_dists:
                    min_pos_dist = min(pos_dists)

        if not success:
            pos_dists = []
            for target_obj_id in self.target_obj_ids:
                pos_dist = np.linalg.norm(
                    np.array(pos[:2]) - np.array(self.info[target_obj_id][0][:2])
                )
                pos_dists.append(pos_dist)
            if pos_dists:
                min_pos_dist = min(pos_dists)

        if not follow_place:
            if success:
                success = self.move_joints(self.drop_joints1)
                self.open_gripper(is_slow=True)
            self.go_home()

        print(f"Grasp at {pose[:3]}, success={success}")

        pb.changeDynamics(self.ee, self.ee_finger_pad_id, lateralFriction=0.9)

        return success, grasped_obj_id, min_pos_dist

    def place_out_of_workspace(self):
        """Place object out of workspace (trash)."""
        success = self.move_joints(self.drop_joints0)
        success &= self.is_gripper_closed
        self.open_gripper(is_slow=True)
        self.go_home()

        print(f"Move the object to the trash bin, success={success}")
        pb.changeDynamics(self.ee, self.ee_finger_pad_id, lateralFriction=0.9)
        return success

    def place(self, pose, speed=0.005):
        """Execute placing primitive."""
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        pose = np.array(pose, dtype=np.float32)
        rot = pose[-4:]
        pos = pose[:3]
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = pos

        ee_tip_transform = np.array([
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, -self.ee_tip_z_offset],
            [0, 0, 0, 1]
        ])

        ee_transform = transform @ ee_tip_transform
        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        self.close_gripper()
        success = self.move_ee_pose((over, rot))

        if success:
            success = self.straight_move(over, pos, rot, speed/4, detect_force=True)
            self.open_gripper(is_slow=True)
            self.go_home()

        print(f"Place at {pose[:3]}, success={success}")

        pb.changeDynamics(self.ee, self.ee_finger_pad_id, lateralFriction=0.9)

        return success

    def open_gripper(self, is_slow=False, blocking=True):
        """Open the gripper.

        Args:
            is_slow: Use slower movement
            blocking: If True, wait until gripper reaches target. If False, just set target.
        """
        if blocking:
            self._move_gripper(self.gripper_angle_open, is_slow=is_slow)
        else:
            self._set_gripper_target(self.gripper_angle_open)

    def close_gripper(self, is_slow=True, blocking=True):
        """Close the gripper.

        Args:
            is_slow: Use slower movement
            blocking: If True, wait until gripper reaches target. If False, just set target.
        """
        if blocking:
            self._move_gripper(self.gripper_angle_close, is_slow=is_slow)
        else:
            self._set_gripper_target(self.gripper_angle_close)

    def _set_gripper_target(self, target_angle):
        """Set gripper target without waiting (non-blocking)."""
        pb.setJointMotorControl2(
            bodyIndex=self.ee,
            jointIndex=self.gripper_main_joint,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
            maxVelocity=1.0,
            physicsClientId=self._client_id,
        )
        # Also set mimic joint for proper gripper sync
        pb.setJointMotorControl2(
            bodyIndex=self.ee,
            jointIndex=self.gripper_mimic_joints["right_outer_knuckle_joint"],
            controlMode=pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
            maxVelocity=1.0,
            physicsClientId=self._client_id,
        )

    @property
    def is_gripper_closed(self):
        gripper_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]
        return gripper_angle < self.gripper_angle_close_threshold

    def _move_gripper(self, target_angle, timeout=3, is_slow=False):
        """Move gripper to target angle."""
        # Longer timeout when recording (3x) and even longer with CPU rendering (10x)
        cpu_render = os.environ.get('A2_DISABLE_EGL', '').lower() == 'true'
        if self._frame_recorder is not None:
            effective_timeout = timeout * 10 if cpu_render else timeout * 3
        else:
            effective_timeout = timeout * 3 if cpu_render else timeout
        t0 = time.time()
        prev_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]

        if is_slow:
            pb.setJointMotorControl2(
                self.ee, self.gripper_main_joint,
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            pb.setJointMotorControl2(
                self.ee, self.gripper_mimic_joints["right_outer_knuckle_joint"],
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            for _ in range(10):
                pb.stepSimulation()
                self._maybe_record_frame()

            while (time.time() - t0) < effective_timeout:
                current_angle = pb.getJointState(self.ee, self.gripper_main_joint)[0]
                diff_angle = abs(current_angle - prev_angle)
                if diff_angle < 1e-4:
                    break
                prev_angle = current_angle
                for _ in range(10):
                    pb.stepSimulation()
                    self._maybe_record_frame()

        pb.setJointMotorControl2(
            self.ee, self.gripper_main_joint,
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        pb.setJointMotorControl2(
            self.ee, self.gripper_mimic_joints["right_outer_knuckle_joint"],
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        for _ in range(10):
            pb.stepSimulation()
            self._maybe_record_frame()


if __name__ == "__main__":
    env = Environment(gui=True, object_set="train")
    env.reset()
    print(pb.getPhysicsEngineParameters(env._client_id))
    time.sleep(5)
