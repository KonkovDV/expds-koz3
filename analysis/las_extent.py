"""Шаг DEM, который получит LAS: как в accel.grid_spec, без чтения точек."""
import math
import laspy

path = r"F:\expds3\train-expds-3\101_Гнёздово_Днепровская\01_Гнёздово_Днепровская_La_точки\01_Гнёздово_Днепровская_La_точки.las"
with laspy.open(path) as f:
    h = f.header
    res = 0.1
    max_pixels = 60_000_000
    left = math.floor(h.mins[0] / res) * res
    bottom = math.floor(h.mins[1] / res) * res
    width = int(math.ceil((h.maxs[0] - left) / res)) + 1
    height = int(math.ceil((h.maxs[1] - bottom) / res)) + 1
    factor = 1
    if width * height > max_pixels:
        factor = int(math.ceil(math.sqrt(width * height / max_pixels)))
    print(f"points {h.point_count}  {width}x{height} @ 0.1m  factor {factor}  step {0.1 * factor:.2f}m")
