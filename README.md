# Project Zomboid - Real Life Map Generator

This project provides a simple tool to generate simplified maps and vegetation maps from real life terrain with OpenStreetMap data using Project Zomboid official color mapping palette.

<div align="center">
  <img src="./readmeData/interface.png" width="400" alt="Interface">
</div>
<table>
  <tr>
    <td><img src="./readmeData/kerfany_satellite.png" width="250" alt="Satellite view"></td>
    <td><img src="./readmeData/kerfany_simplified.png" width="250" alt="Simplified map"></td>
    <td><img src="./readmeData/kerfany_vegetation.png" width="250" alt="Vegetation map"></td>
  </tr>
</table>

## Installation

1. Clone this repository or download the files.

2. Install the required dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```
osmnx
matplotlib
pillow
numpy
shapely
pyproj
rtree
geopandas
```
Tkinter is usually included with Python by default.

## Usage

Run the application with:

```bash
python map_generator.py
```

### In the GUI:

* Enter the **latitude** and **longitude** of the map center.

* Choose the **number of cells** per side (e.g., 2 for a 2x2 grid).

* Adjust the **download margin** as a percentage of map size. It is clamped between 300 m and 2.5 km so large grids do not pull enormous OSM areas into memory. Only terrain and water features are downloaded (not buildings or amenities).

* Adjust the **road width scale** to make roads more or less visible on the generated maps.

* Click **"Generate Maps + Vegetation"** to download data and create both the simplified map grid and vegetation maps in one operation.

## Output

All generated files are saved under `output/`, in a subfolder named from the coordinates and grid size (e.g. `output/47.803288_-3.720999_2x2/`). Running the tool again with the same parameters overwrites that folder; different locations or grid sizes get their own folder.

### Complete Maps (Full Resolution)
* `complete_map.png` — Full resolution simplified map of the entire downloaded area
* `complete_map_veg.png` — Full resolution vegetation map of the entire downloaded area
* `complete_roads.png` — Vehicle roads only (excludes footways, paths, pedestrian ways, steps, etc.)
* `complete_roads_with_sidewalk_both.png` — Roads tagged `sidewalk=both` or `sidewalk=yes`
* `complete_roads_with_sidewalk_left.png` — Roads tagged `sidewalk=left`, `both`, or `yes`
* `complete_roads_with_sidewalk_right.png` — Roads tagged `sidewalk=right`, `both`, or `yes`
* `complete_footpaths.png` — Pedestrian walkways only (`footway`, `path`, `pedestrian`, `steps`, etc.)
* `complete_canals.png` — Canals from OSM `waterway=canal` features

These complete maps are useful for:
- Verifying that all data was correctly downloaded and rendered
- Getting an overview of the entire area before it's split into cells
- Debugging any issues with the generation process

### Map Tiles (Cell Grid)
* `map_cells/` — Contains simplified map tiles in a grid format (e.g., `0,0.png`, `0,1.png`, etc.) plus tool layers per cell:
  * `0,0_roads.png`, `0,0_roads_with_sidewalk_both.png`, `0,0_roads_with_sidewalk_left.png`, `0,0_roads_with_sidewalk_right.png`, `0,0_footpaths.png`, `0,0_canals.png`
* `map_vegetation/` — Contains corresponding vegetation map tiles (`0,0_veg.png`, etc.)

Each cell is 300x300 pixels and represents a 300x300 meter area in the real world.

## Workflow

1. **Data Download**: Downloads OpenStreetMap data for roads, natural features, landuse, water bodies, etc.
2. **Cell Rendering**: Renders each 300×300 m tile individually using Project Zomboid's color palette (keeps memory use low on large grids)
3. **Vegetation Processing**: Classifies vegetation per tile from the simplified map
4. **Preview Stitching**: Assembles full-area preview images when the grid is small enough; very large grids keep only the per-cell tiles

Large grids (roughly 66×66 cells and above) skip the full `complete_map.png` preview to avoid loading the entire image into RAM. The individual tiles in `map_cells/` and `map_vegetation/` are always generated.

## Known issues

The coastline is often badly generated because it's a line in OSM, so it's 'impossible' to color. I've drawn it anyway, to make it easier to modify later (paint bucket).

## About

This project uses [OSMnx](https://osmnx.readthedocs.io/) to download OpenStreetMap data.

OpenStreetMap data is made available under the Open Database License (ODbL). You are free to copy, distribute, transmit and adapt the data, as long as you credit OpenStreetMap and its contributors. If you alter or build upon the data, you may distribute the result only under the same license.

Please make sure your usage complies with OpenStreetMap's usage policy, especially if using this tool at scale or in an automated fashion.

If you use OSMnx in your work, please cite the [paper](https://onlinelibrary.wiley.com/doi/10.1111/gean.70009):

Boeing, G. (2025). Modeling and Analyzing Urban Networks and Amenities with OSMnx. Geographical Analysis, published online ahead of print.

Developed by [SadPeanut](https://github.com/SadPeanut).
