# AgiBot X2-Ultra asset (SuperMimic adaptation)

- `build_x2_mjcf.py`: regenerates `agibot_x2ultra.xml` from the team URDF
  (meshes are NOT committed; the XML's `meshdir` points at the canonical
  asset tree — regenerate with `--meshdir` when relocating).
- `agibot_x2ultra.xml`: hand-built MJCF (freejoint pelvis, URDF-tree joint
  order == `X2_JOINT_NAMES`, mesh collisions + fitted sole boxes, home keyframe).
- Runtime constants (actuators/collisions/action scale): `src/supermimic/assets/x2_constants.py`.
- Robot registry: `src/supermimic/robots/`.

Standing geometry: pelvis 0.92 m straight / 0.70 m home; sole box fitted to
the foot mesh (10.6 x 13.0 cm); total mass 44.8 kg.
