#!/usr/bin/env python3
"""Inspect the composed official Unitree G1 USD inside Isaac Sim."""

from __future__ import annotations

from isaacsim import SimulationApp


def main() -> None:
    app = SimulationApp({"headless": True})

    import omni.usd
    from pxr import UsdGeom

    asset = "/workspace/Isacc_dataset/assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World/RobotRig").GetPrim()
    asset_root = UsdGeom.Xform.Define(stage, "/World/RobotRig/RobotAsset").GetPrim()
    asset_root.GetReferences().AddReference(asset)
    app.update()

    robot_prims = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/RobotRig"):
            robot_prims.append((path, prim.GetTypeName(), prim.IsInstance(), prim.IsInstanceable()))

    print(f"robot_prim_count={len(robot_prims)}")
    for item in robot_prims[:120]:
        print("PRIM", item)

    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    box = cache.ComputeWorldBound(root).ComputeAlignedBox()
    box_min = box.GetMin()
    box_max = box.GetMax()
    print(f"bbox_min={box_min}")
    print(f"bbox_max={box_max}")
    print(f"bbox_size=({box_max[0] - box_min[0]}, {box_max[1] - box_min[1]}, {box_max[2] - box_min[2]})")

    pose_targets = [
        "left_shoulder_pitch",
        "right_shoulder_pitch",
        "left_elbow",
        "right_elbow",
        "waist_yaw",
        "left_knee",
        "right_knee",
    ]
    for target in pose_targets:
        matches = [str(prim.GetPath()) for prim in stage.Traverse() if target in prim.GetName().lower()]
        print(f"target={target} matches={matches[:20]}")

    app.close()


if __name__ == "__main__":
    main()
