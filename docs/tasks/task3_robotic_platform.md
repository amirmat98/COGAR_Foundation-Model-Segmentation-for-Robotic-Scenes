# Task 3: Robotic Platform

Task 3 requires a simulated robotic platform such as Unitree G1 EDU or Unitree
AS2 EDU.

## Platform Used

The main dataset uses the official Unitree G1 USD asset in Isaac Sim.

The final Isaac generator is configured to require the official USD asset and
not fall back to a surrogate robot.

## Dataset Output

The generated dataset is:

```text
/mnt/Info/COGAR_DATASETs/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000
```

Configured repo-default path:

```text
Datasets/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000
```

The BlenderProc dataset uses a robot-gripper proxy as an additional tabletop
occluder. It is secondary and does not replace the official Unitree G1 dataset.
