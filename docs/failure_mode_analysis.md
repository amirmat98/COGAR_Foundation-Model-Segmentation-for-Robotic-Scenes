# Failure Mode Analysis

## Transparent glass objects

Several of the lowest-IoU examples are glass objects. These objects have weak visible boundaries, partial transparency, and overlap with background geometry. Box prompts help but do not fully solve this issue.

Representative rows:
- Box: row 334, object 383, glass_object, IoU 0.0193
- Box: row 309, object 355, glass_object, IoU 0.2425
- Point: row 309, object 355, glass_object, IoU 0.0091
- Auto: row 309, object 355, glass_object, IoU 0.0000

## Thin cables

Cables are elongated, thin, and often partially occluded. SAM sometimes segments only part of the cable or merges it with nearby dark structures.

Representative rows:
- Box: row 273, object 311, cable, IoU 0.2182
- Box: row 311, object 357, cable, IoU 0.2654
- Auto: row 339, object 388, cable, IoU 0.0000

## Small screws and connectors

Point and automatic prompting fail more often on small parts than box prompting. Small objects are sensitive to prompt location and can be missed by automatic mask proposal generation.

Representative rows:
- Point: row 37, object 46, screw, IoU 0.0262
- Point: row 230, object 262, connector, IoU 0.0224
- Auto: row 34, object 43, connector, IoU 0.0000
- Auto: row 43, object 52, screw, IoU 0.0000

## Robot gripper and occluded dark parts

Robot gripper parts sometimes fail when they are dark, partially occluded, or visually merged with other dark structures.

Representative rows:
- Box: row 186, object 213, robot_gripper, IoU 0.0915
- Point: row 186, object 213, robot_gripper, IoU 0.0881
- Point: row 359, object 413, robot_gripper, IoU 0.0667

## Prompt-mode comparison

Box prompts are the most reliable prompt type in this sanity benchmark. Point prompts are competitive on easy objects but have more severe failures. Automatic mask generation has high median IoU but many near-zero failures and is too slow for real-time robotic perception in the current setup.
