# visuals/arm_renderer.py
"""MuJoCo-based 3D arm renderer for Phase-13 clip frame generation.

Uses EGL headless rendering to produce 1920×1080 RGB frames of the 2-link
planar arm.  PIL composites text and glow-arc overlays onto each frame.

Pixel mapping (calibrated via test spheres at known world positions,
camera azimuth=270, fovy=40°, distance=2.8, lookat=(0,0,0.45)):
  screen_x = 728 − world_x × 355   (world +X → screen LEFT)
  screen_y = 568 − world_z × 354   (world +Z → screen UP)
where world_x = sim_x, world_z = sim_y (2D trajectory → 3D MuJoCo XZ plane).
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Arm geometry (matches trajectory_gen.VISUAL_REACHER_CONFIG) ---
_L1 = 0.6
_L2 = 0.6
_SHOULDER_Z = -0.1   # shoulder Z in MuJoCo world (= sim y = -0.1)

# Natural bent-arm rest position used for hold/MISS frames.
# At sim (0, 0.8): theta1≈41°, symmetric inverted-V shape pointing up.
REST_POSITION: list[float] = [0.0, 0.8]

# --- Calibrated pixel mapping ---
_W, _H = 1920, 1080
_CAL_SX    =  728.0   # screen x for world (0, z=any)
_CAL_SY    =  568.0   # screen y for world z=0
_SCALE_X   = -355.0   # px per world_x unit (negative: +X goes LEFT)
_SCALE_Z   =  354.0   # px per world_z unit (Z up = screen Y down)

# --- Fonts ---
_FONT_PATH = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FONT_PATH, size)


# --- MuJoCo MJCF XML ---
_ARM_XML = """<mujoco model="arm">
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="2048" offsamples="8"/>
    <rgba haze="0.051 0.067 0.09 1"/>
    <map fogstart="4" fogend="12"/>
  </visual>
  <option gravity="0 0 0"/>
  <asset>
    <texture name="sky" type="skybox" builtin="flat"
             rgb1="0.051 0.067 0.09" rgb2="0.051 0.067 0.09"
             width="64" height="64"/>
    <material name="arm_mat"    rgba="0.35 0.65 1.0 1.0" shininess="0.7" specular="0.5"/>
    <material name="joint_mat"  rgba="0.55 0.82 1.0 1.0" shininess="1.0" specular="0.8"/>
    <material name="target_mat" rgba="1.0 1.0 0.85 1.0"  shininess="0.4"
              specular="0.3" emission="0.4"/>
    <material name="bg_mat"     rgba="0.051 0.067 0.09 1.0"/>
    <material name="floor_mat"  rgba="0.07 0.08 0.12 1.0" shininess="0.05"/>
  </asset>
  <worldbody>
    <light name="key"  directional="true" pos="1 -4 5"  dir="-0.15 0.55 -0.8"
           diffuse="0.85 0.88 1.0" specular="0.25 0.25 0.35" castshadow="true"/>
    <light name="fill" directional="true" pos="-2 -3 3" dir="0.3 0.5 -0.6"
           diffuse="0.18 0.22 0.38" specular="0.03 0.03 0.08"/>
    <light name="rim"  directional="true" pos="0 3 2"   dir="0 -0.7 -0.3"
           diffuse="0.1 0.15 0.25" specular="0.02 0.02 0.05"/>
    <!-- Background panel fills the backdrop for the dark theme. -->
    <geom name="bg"    type="plane" pos="0 0.35 0" zaxis="0 -1 0"
          size="4 4 0.01" material="bg_mat" contype="0" conaffinity="0"/>
    <geom name="floor" type="plane" pos="0 0 -1.3" size="4 4 0.01"
          material="floor_mat" contype="0" conaffinity="0"/>
    <!-- Target: glowing sphere with faint halo ring. -->
    <geom name="target"      type="sphere"   pos="0 0 1.0" size="0.07"  material="target_mat"/>
    <geom name="target_ring" type="cylinder" pos="0 0 1.0" size="0.20 0.005"
          material="target_mat" rgba="1.0 1.0 0.7 0.18" euler="90 0 0"/>
    <!-- 2-link planar arm in the XZ plane. -->
    <body name="shoulder_body" pos="0 0 -0.1">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <geom name="upper_arm"    type="capsule" fromto="0 0 0 0 0 0.6"
            size="0.048" material="arm_mat"/>
      <geom name="shoulder_jnt" type="sphere" pos="0 0 0"   size="0.068" material="joint_mat"/>
      <geom name="elbow_jnt"    type="sphere" pos="0 0 0.6" size="0.058" material="joint_mat"/>
      <body name="elbow_body" pos="0 0 0.6">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="forearm"  type="capsule" fromto="0 0 0 0 0 0.6"
              size="0.040" material="arm_mat"/>
        <geom name="hand_jnt" type="sphere" pos="0 0 0.6" size="0.055" material="joint_mat"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

# Geom names that belong to the arm (for bulk colour changes).
_ARM_GEOM_NAMES = ["upper_arm", "forearm", "shoulder_jnt", "elbow_jnt", "hand_jnt"]


# --- Inverse kinematics ---

def _pos_to_joints(pos_xy: list[float]) -> tuple[float, float]:
    """Convert 2D hand endpoint → (shoulder_angle, elbow_angle) for MuJoCo.

    Uses elbow-right convention (phi + alpha).  theta2 is always negative
    (arm folds so the joint range [-3.14, 3.14] is never violated).
    """
    hx, hz = float(pos_xy[0]), float(pos_xy[1])
    dx = hx  # shoulder is at world X = 0
    dz = hz - _SHOULDER_Z  # vector from shoulder to hand
    d = float(np.sqrt(dx * dx + dz * dz))
    # Clamp to reachable band so law-of-cosines stays numerically stable.
    d = float(np.clip(d, 0.001, _L1 + _L2 - 0.001))

    cos_alpha = (d * d + _L1 * _L1 - _L2 * _L2) / (2.0 * d * _L1)
    alpha = float(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
    phi = float(np.arctan2(dx, dz))   # angle from world +Z toward +X
    theta1 = phi + alpha               # elbow-right: elbow toward +X world

    cos_beta = (_L1 * _L1 + _L2 * _L2 - d * d) / (2.0 * _L1 * _L2)
    beta = float(np.arccos(np.clip(cos_beta, -1.0, 1.0)))
    # Joint 0 = fully extended; -(π-β) gives the folded angle.
    theta2 = -(np.pi - beta)
    return theta1, theta2


# --- Screen-space helpers ---

def sim_to_screen(x: float, y: float) -> tuple[int, int]:
    """Convert 2D simulation coords to screen pixel coords (calibrated)."""
    sx = _CAL_SX + x * _SCALE_X
    sy = _CAL_SY - y * _SCALE_Z    # Z up → screen Y decreases
    return int(round(sx)), int(round(sy))


def _hex_to_int_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _draw_glow_arc(
    draw: ImageDraw.ImageDraw,
    overlay: Image.Image,
    screen_pts: list[tuple[int, int]],
    color_hex: str,
    alpha_scale: float = 1.0,
) -> None:
    """Three-pass glow arc (wide+faint, medium, sharp core) on an RGBA overlay."""
    r, g, b = _hex_to_int_rgb(color_hex)
    for width_px, alpha in [(28, int(18 * alpha_scale)),
                             (12, int(48 * alpha_scale)),
                             (3,  int(200 * alpha_scale))]:
        if len(screen_pts) >= 2:
            ImageDraw.Draw(overlay).line(screen_pts, fill=(r, g, b, alpha), width=width_px)


# --- Main renderer class ---

class ArmRenderer:
    """MuJoCo EGL renderer for the 2-link planar arm.

    Lifetime: create once per clip, call render() per frame, call close() when done.
    """

    def __init__(self) -> None:
        self._model = mujoco.MjModel.from_xml_string(_ARM_XML)
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=_H, width=_W)

        # Free camera: azimuth=270 (from -Y), elevation=0, distance=2.8.
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.lookat[:] = [0.0, 0.0, 0.45]
        self._cam.distance = 2.8
        self._cam.azimuth = 270.0
        self._cam.elevation = 0.0

        self._arm_geom_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in _ARM_GEOM_NAMES
        ]

    def _set_arm_rgba(self, rgba: tuple[float, float, float, float]) -> None:
        for gid in self._arm_geom_ids:
            self._model.geom_rgba[gid] = list(rgba)

    def render_raw(
        self,
        pos_xy: list[float],
        color_rgba: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Render the arm at pos_xy with arm colour; return (H, W, 3) uint8."""
        theta1, theta2 = _pos_to_joints(pos_xy)
        self._data.qpos[0] = theta1
        self._data.qpos[1] = theta2
        self._set_arm_rgba(color_rgba)
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=self._cam)
        return self._renderer.render().copy()

    def make_frame(
        self,
        pixels: np.ndarray,
        *,
        labels: list[tuple[tuple[int, int], str, int, str]] | None = None,
        arc_pts: list[tuple[int, int]] | None = None,
        arc_color: str = "#ffffff",
        arc_alpha: float = 1.0,
    ) -> Image.Image:
        """Composite text labels and optional glow arc onto an RGB pixel array.

        Args:
            pixels:     Raw MuJoCo render, (H, W, 3) uint8.
            labels:     List of (screen_xy, text, font_size, hex_color).
            arc_pts:    Screen-space polyline for the glow arc (or None).
            arc_color:  Hex color for the arc.
            arc_alpha:  Overall alpha scaling for the arc.
        """
        img = Image.fromarray(pixels, "RGB").convert("RGBA")

        # --- Glow arc overlay ---
        if arc_pts and len(arc_pts) >= 2:
            arc_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            _draw_glow_arc(
                ImageDraw.Draw(arc_layer), arc_layer,
                arc_pts, arc_color, alpha_scale=arc_alpha,
            )
            img = Image.alpha_composite(img, arc_layer)

        # --- Text labels ---
        if labels:
            txt_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(txt_layer)
            for (sx, sy), text, size, hex_col in labels:
                r, g, b = _hex_to_int_rgb(hex_col)
                draw.text((sx, sy), text, fill=(r, g, b, 255), font=_font(size))
            img = Image.alpha_composite(img, txt_layer)

        return img.convert("RGB")

    def save_frame(
        self,
        pixels: np.ndarray,
        frames_dir: Path,
        index: int,
        *,
        labels: list[tuple[tuple[int, int], str, int, str]] | None = None,
        arc_pts: list[tuple[int, int]] | None = None,
        arc_color: str = "#ffffff",
        arc_alpha: float = 1.0,
    ) -> None:
        """Render, composite, and save frame PNG."""
        frames_dir.mkdir(parents=True, exist_ok=True)
        img = self.make_frame(
            pixels,
            labels=labels,
            arc_pts=arc_pts,
            arc_color=arc_color,
            arc_alpha=arc_alpha,
        )
        img.save(frames_dir / f"{index:04d}.png")

    def close(self) -> None:
        self._renderer.close()
