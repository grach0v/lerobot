#!/usr/bin/env python
"""Render all A2 assets and save images for texture verification.

Usage:
    uv run python scripts/render_a2_assets.py --output_dir ./asset_renders
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_data
from PIL import Image


def get_assets_path():
    """Get path to A2 assets."""
    cache_dir = Path.home() / ".cache" / "a2_assets"
    if not cache_dir.exists():
        raise FileNotFoundError(f"A2 assets not found at {cache_dir}. Run A2Env first to download.")
    return cache_dir


def setup_pybullet():
    """Initialize PyBullet in direct mode with EGL rendering."""
    client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Try to load EGL plugin for GPU rendering
    try:
        egl = pb.loadPlugin("eglRendererPlugin")
        if egl >= 0:
            print("EGL GPU rendering enabled")
    except Exception:
        print("EGL not available, using CPU rendering")

    pb.setGravity(0, 0, -9.8)
    return client


def render_object(urdf_path, output_path, client, image_size=(512, 512)):
    """Render a single object and save the image."""
    # Reset simulation
    pb.resetSimulation(physicsClientId=client)
    pb.setGravity(0, 0, -9.8, physicsClientId=client)

    # Load ground plane
    pb.loadURDF("plane.urdf", [0, 0, -0.05], physicsClientId=client)

    # Load the object
    try:
        obj_id = pb.loadURDF(
            str(urdf_path),
            basePosition=[0, 0, 0.1],
            useFixedBase=False,
            physicsClientId=client
        )
    except Exception as e:
        print(f"  Failed to load {urdf_path}: {e}")
        return False

    # Let object settle
    for _ in range(50):
        pb.stepSimulation(physicsClientId=client)

    # Get object position for camera focus
    pos, _ = pb.getBasePositionAndOrientation(obj_id, physicsClientId=client)

    # Setup camera - orbit around object
    views = [
        ("front", [0.3, 0, 0.15], [0, 0, 0.05]),
        ("side", [0, 0.3, 0.15], [0, 0, 0.05]),
        ("top", [0, 0, 0.4], [0, 0, 0.05]),
        ("angle", [0.2, 0.2, 0.2], [0, 0, 0.05]),
    ]

    images = []
    for view_name, cam_pos, target in views:
        view_matrix = pb.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=client
        )
        proj_matrix = pb.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.01,
            farVal=2.0,
            physicsClientId=client
        )

        _, _, rgba, _, _ = pb.getCameraImage(
            width=image_size[0],
            height=image_size[1],
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client
        )

        img = np.array(rgba, dtype=np.uint8).reshape(image_size[1], image_size[0], 4)
        images.append(img[:, :, :3])  # RGB only

    # Create 2x2 grid of views
    top_row = np.concatenate([images[0], images[1]], axis=1)
    bottom_row = np.concatenate([images[2], images[3]], axis=1)
    grid = np.concatenate([top_row, bottom_row], axis=0)

    # Save image
    Image.fromarray(grid).save(output_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Render A2 assets for texture verification")
    parser.add_argument("--output_dir", type=str, default="./asset_renders", help="Output directory")
    parser.add_argument("--object_set", type=str, default="all", choices=["train", "test", "all"],
                        help="Which object set to render")
    parser.add_argument("--image_size", type=int, default=256, help="Image size per view")
    args = parser.parse_args()

    assets_path = get_assets_path()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = setup_pybullet()

    object_sets = []
    if args.object_set in ["train", "all"]:
        object_sets.append(("simplified_objects", assets_path / "simplified_objects"))
    if args.object_set in ["test", "all"]:
        object_sets.append(("unseen_objects", assets_path / "unseen_objects"))

    total_success = 0
    total_failed = 0
    failed_objects = []

    for set_name, set_path in object_sets:
        print(f"\n{'=' * 50}")
        print(f"Rendering {set_name}")
        print(f"{'=' * 50}")

        set_output = output_dir / set_name
        set_output.mkdir(parents=True, exist_ok=True)

        # Find all URDF files
        urdf_files = sorted(set_path.glob("*.urdf"))

        for urdf_path in urdf_files:
            obj_name = urdf_path.stem
            output_path = set_output / f"{obj_name}.png"

            print(f"Rendering {obj_name}...", end=" ")

            success = render_object(
                urdf_path,
                output_path,
                client,
                image_size=(args.image_size, args.image_size)
            )

            if success:
                print("OK")
                total_success += 1
            else:
                print("FAILED")
                total_failed += 1
                failed_objects.append(f"{set_name}/{obj_name}")

    pb.disconnect(client)

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total rendered: {total_success}")
    print(f"Total failed: {total_failed}")
    print(f"Output directory: {output_dir}")

    if failed_objects:
        print(f"\nFailed objects:")
        for obj in failed_objects:
            print(f"  - {obj}")

    # Create index HTML for easy viewing
    html_path = output_dir / "index.html"
    with open(html_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>A2 Asset Renders</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        h1, h2 { color: #333; }
        .gallery { display: flex; flex-wrap: wrap; gap: 10px; }
        .item { background: white; padding: 10px; border-radius: 8px; text-align: center; }
        .item img { max-width: 300px; border: 1px solid #ddd; }
        .item p { margin: 5px 0 0 0; font-size: 14px; color: #666; }
    </style>
</head>
<body>
    <h1>A2 Asset Renders</h1>
""")

        for set_name, _ in object_sets:
            set_output = output_dir / set_name
            images = sorted(set_output.glob("*.png"))

            f.write(f"<h2>{set_name} ({len(images)} objects)</h2>\n")
            f.write('<div class="gallery">\n')

            for img_path in images:
                rel_path = f"{set_name}/{img_path.name}"
                f.write(f'''    <div class="item">
        <img src="{rel_path}" alt="{img_path.stem}">
        <p>{img_path.stem}</p>
    </div>\n''')

            f.write('</div>\n')

        f.write("</body>\n</html>")

    print(f"\nView results: file://{html_path.absolute()}")


if __name__ == "__main__":
    main()
