"""
Isaac Sim / Replicator sanity dataset generator.

Run this script inside Isaac Sim's Python environment, not inside the normal
project .venv.

Purpose:
- Generate a small raw synthetic dataset for sanity checking.
- Export RGB, semantic segmentation, instance segmentation if available,
  and 2D tight bounding boxes.
- Use this raw output later with scripts/convert_isaac_export_to_sim_index.py.

Notes:
- This script is intentionally a skeleton.
- Asset URLs and object spawning should be adapted after checking which Isaac
  assets are installed locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small Isaac Sim Replicator sanity dataset."
    )

    parser.add_argument(
        "--output-dir",
        default="raw_isaac_exports/sanity_run",
        help="Raw Replicator output directory.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=25,
        help="Number of frames to generate.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Rendered image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Rendered image height.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim headless.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Isaac Sim modules must be imported after SimulationApp is launched.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": "RayTracedLighting",
        }
    )

    import omni.replicator.core as rep

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[COGAR] Isaac Sim sanity dataset generation")
    print("[COGAR] Output dir:", output_dir)
    print("[COGAR] Frames:", args.num_frames)
    print("[COGAR] Resolution:", args.width, "x", args.height)

    # ---------------------------------------------------------------------
    # Scene setup
    # ---------------------------------------------------------------------
    # This skeleton uses primitive shapes first, because they are available
    # without downloading extra USD assets. Later, replace these primitives
    # with real robotic/tabletop assets.
    #
    # Required semantic classes for the benchmark:
    # - metal_tool
    # - glass_cup
    # - connector
    # - screw
    # - cable
    # - robot_gripper_or_hand
    # - table
    # - distractor_object
    # ---------------------------------------------------------------------

    with rep.new_layer():
        # Camera
        camera = rep.create.camera(
            position=(0.0, -4.0, 3.0),
            rotation=(60.0, 0.0, 0.0),
            focal_length=35.0,
        )

        render_product = rep.create.render_product(
            camera,
            resolution=(args.width, args.height),
        )

        # Table / workspace
        table = rep.create.cube(
            position=(0.0, 0.0, -0.05),
            scale=(3.0, 2.0, 0.1),
            semantics=[("class", "table")],
        )

        # Reflective-metal placeholder object.
        metal_tool = rep.create.cube(
            position=(-0.8, 0.0, 0.15),
            scale=(0.45, 0.12, 0.08),
            semantics=[("class", "metal_tool")],
        )

        # Transparent-object placeholder.
        transparent_box = rep.create.cube(
            position=(0.0, 0.2, 0.18),
            scale=(0.28, 0.28, 0.25),
            semantics=[("class", "transparent_box")],
        )

        # Small-part placeholder.
        screw = rep.create.cylinder(
            position=(0.6, -0.25, 0.08),
            scale=(0.06, 0.06, 0.04),
            semantics=[("class", "screw")],
        )

        # Cable / occluder placeholder.
        cable = rep.create.torus(
            position=(0.35, 0.35, 0.08),
            scale=(0.18, 0.18, 0.03),
            semantics=[("class", "cable")],
        )

        # Robot/gripper placeholder occluder.
        robot_part = rep.create.cube(
            position=(-0.2, -0.35, 0.35),
            scale=(0.18, 0.55, 0.12),
            semantics=[("class", "robot_gripper_or_hand")],
        )

        # Lighting
        rep.create.light(
            light_type="Distant",
            rotation=(315, 0, 0),
            intensity=3000,
        )

        # Randomization trigger.
        with rep.trigger.on_frame(num_frames=args.num_frames):
            with metal_tool:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (-1.0, -0.4, 0.12),
                        (-0.3, 0.4, 0.25),
                    ),
                    rotation=rep.distribution.uniform(
                        (0, 0, 0),
                        (0, 0, 180),
                    ),
                )

            with transparent_box:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (-0.2, -0.1, 0.12),
                        (0.5, 0.5, 0.28),
                    ),
                    rotation=rep.distribution.uniform(
                        (0, 0, 0),
                        (0, 0, 180),
                    ),
                )

            with screw:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (0.3, -0.5, 0.06),
                        (1.0, 0.2, 0.10),
                    ),
                )

            with robot_part:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (-0.5, -0.55, 0.25),
                        (0.3, -0.2, 0.5),
                    ),
                    rotation=rep.distribution.uniform(
                        (0, 0, -30),
                        (0, 0, 30),
                    ),
                )

            with camera:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (-0.2, -4.2, 2.5),
                        (0.2, -3.5, 3.3),
                    ),
                    rotation=rep.distribution.uniform(
                        (55, -5, -5),
                        (70, 5, 5),
                    ),
                )

        # Writer setup.
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(output_dir),
            rgb=True,
            bounding_box_2d_tight=True,
            semantic_segmentation=True,
            instance_segmentation=True,
            colorize_semantic_segmentation=False,
            colorize_instance_segmentation=False,
        )
        writer.attach([render_product])

        rep.orchestrator.run()

    print("[COGAR] Generation requested.")
    print("[COGAR] Check raw output folder:", output_dir)

    simulation_app.close()


if __name__ == "__main__":
    main()
