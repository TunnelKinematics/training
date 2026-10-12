import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg


def rough_terrains(max_step_height: float) -> TerrainGeneratorCfg:
    """Curriculum terrains scaled to the robot: rows get harder up to max_step_height."""
    h = max_step_height
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            "rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.15, noise_range=(0.01, 0.4 * h), noise_step=0.01, border_width=0.25
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.1, grid_width=0.45, grid_height_range=(0.02, 0.6 * h), platform_width=2.0
            ),
            "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.02, h),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.02, h),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        },
    )
