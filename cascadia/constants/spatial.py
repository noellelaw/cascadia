# To overlay on rasters and have better idea on how 
# each class will cause a grid square to flood
NLCD = [
    "Open Water",
    "Perennial Ice/Snow",
    "Developed, Open Space",
    "Developed, Low Intensity",
    "Developed, Medium Intensity",
    "Developed, High Intensity",
    "Barren Land (Rock/Sand/Clay)",
    "Deciduous Forest",
    "Evergreen Forest",
    "Mixed Forest",
    "Shrub/Scrub",
    "Grassland/Herbaceous",
    "Pasture/Hay",
    "Cultivated Crops",
    "Woody Wetlands",
    "Emergent Herbaceous Wetlands"
]

nlcd_classes = [
    (11, "Open Water"),
    (12, "Perennial Ice/Snow"),
    (21, "Developed, Open Space"),
    (22, "Developed, Low Intensity"),
    (23, "Developed, Medium Intensity"),
    (24, "Developed, High Intensity"),
    (31, "Barren Land (Rock/Sand/Clay)"),
    (41, "Deciduous Forest"),
    (42, "Evergreen Forest"),
    (43, "Mixed Forest"),
    (52, "Shrub/Scrub"),
    (71, "Grassland/Herbaceous"),
    (81, "Pasture/Hay"),
    (82, "Cultivated Crops"),
    (90, "Woody Wetlands"),
    (95, "Emergent Herbaceous Wetlands")
]

land_cover_classes = ['urban', 'vegetation', 'water', 'bare_soil', 'asphalt']
zone_classes = ['residential', 'infrastructure', 'coastal_buffer', 'wetland', 'elevated']
