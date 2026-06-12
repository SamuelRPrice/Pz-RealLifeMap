import gc
import io
import math
import os
import re
import shutil
import threading
import tkinter as tk
from tkinter.messagebox import showerror
from urllib.error import URLError
from urllib.request import Request, urlopen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import osmnx as ox
from shapely.geometry import box




# ========== PALETTES ==========
PALETTE = {
    'dark_grass': (90/255, 100/255, 35/255),
    'medium_grass': (117/255, 117/255, 47/255),
    'light_grass': (145/255, 135/255, 60/255),
    'sand': (210/255, 200/255, 160/255),
    'water': (0/255, 138/255, 255/255),
    'dark_asphalt': (100/255, 100/255, 100/255),
    'medium_asphalt': (120/255, 120/255, 120/255),
    'light_asphalt': (165/255, 160/255, 140/255),
    'gravel_dirt': (140/255, 70/255, 15/255),
    'dirt': (120/255, 70/255, 20/255),
}

PALETTE_ORIG = {
    'dark_grass': (90, 100, 35),
    'medium_grass': (117, 117, 47),
    'light_grass': (145, 135, 60),
}

VEGETATION_COLORS = {
    'dense_trees_and_dark_grass': (127, 0, 0),
    'trees_and_grass': (64, 0, 0),
    'light_long_grass': (0, 255, 0),
}

CELL_SIZE_M = 300
CELL_PX = 300
RENDER_DPI = 100
# Skip stitching full preview images above this side length (~1.2 GB RGB at 20k px).
MAX_COMPLETE_MAP_PX = 20000
MIN_DOWNLOAD_MARGIN_M = 300
MAX_DOWNLOAD_MARGIN_M = 2500
FEATURE_TAGS = {'natural': True, 'landuse': True, 'waterway': True}
FEATURE_COLUMNS = ('natural', 'landuse', 'waterway')
DRAWABLE_GEOM_TYPES = frozenset({
    'Polygon', 'MultiPolygon', 'LineString', 'MultiLineString',
})
PREVIEW_WIDTH = 400
PREVIEW_HEIGHT = 260
OSM_TILE_SIZE = 256
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_USER_AGENT = "Pz-RealLifeMap/1.0 (map generator preview)"




# ========== UTILS ==========
def get_natural_color(natural_value):
    if natural_value is None:
        return None
    nv = str(natural_value).lower()
    if nv in ['tree', 'wood', 'shrubbery', 'tree_row', 'forest']:
        return PALETTE['dark_grass']
    elif nv in ['grassland', 'heath', 'scrub', 'meadow']:
        return PALETTE['light_grass']
    elif nv in ['fell', 'tundra']:
        return PALETTE['medium_grass']
    elif nv in ['sand', 'beach']:
        return PALETTE['sand']
    elif nv in ['water', 'wetland', 'bay', 'coastline']:
        return PALETTE['water']
    else:
        return None

def get_landuse_color(landuse_value):
    if landuse_value is None:
        return None
    lv = str(landuse_value).lower()
    if lv in ['forest', 'wood']:
        return PALETTE['dark_grass']
    elif lv in ['grass', 'meadow', 'farmland', 'recreation_ground']:
        return PALETTE['light_grass']
    return None

def get_road_color(highway, surface=None):
    if isinstance(highway, list): highway = highway[0]
    if isinstance(surface, list): surface = surface[0]
    if highway in ['motorway', 'primary', 'trunk']:
        return PALETTE['dark_asphalt']
    elif highway in ['secondary', 'tertiary', 'residential', 'service', 'unclassified']:
        return PALETTE['medium_asphalt']
    elif highway in ['path', 'track', 'bridleway', 'cycleway', 'footway']:
        if surface == 'sand':
            return PALETTE['sand']
        elif surface in ['gravel', 'dirt', 'earth']:
            return PALETTE['gravel_dirt']
        else:
            return PALETTE['dirt']
    return PALETTE['medium_asphalt']

def get_road_width_m(highway):
    if isinstance(highway, list): highway = highway[0]
    return {
        'motorway': 20,
        'primary': 15,
        'trunk': 15,
        'secondary': 10,
        'tertiary': 8,
        'residential': 7,
        'service': 7,
        'unclassified': 7,
        'path': 2,
        'track': 2,
        'bridleway': 2,
        'cycleway': 2,
        'footway': 2
    }.get(highway, 8)

def get_road_priority(highway):
    """Retourne la priorité de dessin d'une route (plus le chiffre est élevé, plus elle est dessinée tard)"""
    if isinstance(highway, list): 
        highway = highway[0]
    
    priority_order = {
        'path': 1,
        'track': 2,
        'footway': 3,
        'cycleway': 4,
        'bridleway': 5,
        'service': 6,
        'unclassified': 7,
        'residential': 8,
        'tertiary': 9,
        'secondary': 10,
        'primary': 11,
        'trunk': 12,
        'motorway': 13
    }
    
    return priority_order.get(highway, 5)  # valeur par défaut pour les types inconnus

# ========== VEGETATION MAP ==========
def classify_vegetation_color_vectorized(img_array):
    ref_colors = np.array(list(PALETTE_ORIG.values()), dtype=np.int16)
    veg_colors = np.array(list(VEGETATION_COLORS.values()), dtype=np.uint8)
    h, w, _ = img_array.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    threshold_sq = 17 * 17
    chunk_rows = 64
    for y0 in range(0, h, chunk_rows):
        y1 = min(y0 + chunk_rows, h)
        chunk = img_array[y0:y1].astype(np.int16)
        diff = chunk[:, :, np.newaxis, :] - ref_colors[np.newaxis, np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=3)
        closest = np.argmin(dist_sq, axis=2)
        valid = np.min(dist_sq, axis=2) < threshold_sq
        chunk_result = np.zeros((y1 - y0, w, 3), dtype=np.uint8)
        chunk_result[valid] = veg_colors[closest[valid]]
        result[y0:y1] = chunk_result
    return result

def compute_download_margin_m(total_zone_m, margin_factor):
    margin_m = total_zone_m * margin_factor
    return max(MIN_DOWNLOAD_MARGIN_M, min(margin_m, MAX_DOWNLOAD_MARGIN_M))

def meters_to_degree_offsets(lat, half_size_m):
    dlat = half_size_m / 111_320
    cos_lat = math.cos(math.radians(lat))
    dlon = half_size_m / (111_320 * max(abs(cos_lat), 1e-6))
    return dlat, dlon

def compute_render_bbox_deg(lat, lon, nb_cells):
    half_m = (CELL_SIZE_M * nb_cells) / 2
    dlat, dlon = meters_to_degree_offsets(lat, half_m)
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon

def compute_download_bbox_deg(lat, lon, nb_cells, margin_factor):
    total_zone_m = CELL_SIZE_M * nb_cells
    margin_m = compute_download_margin_m(total_zone_m, margin_factor)
    half_m = total_zone_m / 2 + margin_m
    dlat, dlon = meters_to_degree_offsets(lat, half_m)
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon

def lat_lon_to_tile_xy(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n
    return x, y

def latlon_to_world_px(lat, lon, zoom):
    tile_x, tile_y = lat_lon_to_tile_xy(lat, lon, zoom)
    return tile_x * OSM_TILE_SIZE, tile_y * OSM_TILE_SIZE

def bbox_pixel_size(south, west, north, east, zoom):
    x_w, y_n = lat_lon_to_tile_xy(north, west, zoom)
    x_e, y_s = lat_lon_to_tile_xy(south, east, zoom)
    return abs(x_e - x_w) * OSM_TILE_SIZE, abs(y_s - y_n) * OSM_TILE_SIZE

def choose_preview_zoom(south, west, north, east):
    for zoom in range(17, 2, -1):
        width_px, height_px = bbox_pixel_size(south, west, north, east, zoom)
        if width_px <= PREVIEW_WIDTH * 0.9 and height_px <= PREVIEW_HEIGHT * 0.9:
            return zoom
    return 3

def fetch_osm_tile(zoom, tile_x, tile_y):
    tile_count = 2 ** zoom
    tile_x = int(tile_x) % tile_count
    if tile_x < 0:
        tile_x += tile_count
    tile_y = max(0, min(int(tile_y), tile_count - 1))
    url = OSM_TILE_URL.format(z=zoom, x=tile_x, y=tile_y)
    request = Request(url, headers={'User-Agent': OSM_USER_AGENT})
    with urlopen(request, timeout=10) as response:
        return Image.open(io.BytesIO(response.read())).convert('RGB')

def bbox_to_preview_rect(south, west, north, east, zoom, view_left, view_top):
    x1, y1 = latlon_to_world_px(north, west, zoom)
    x2, y2 = latlon_to_world_px(south, east, zoom)
    return (
        x1 - view_left, y1 - view_top,
        x2 - view_left, y2 - view_top,
    )

def build_osm_preview(lat, lon, nb_cells, margin_factor):
    render_bbox = compute_render_bbox_deg(lat, lon, nb_cells)
    download_bbox = compute_download_bbox_deg(lat, lon, nb_cells, margin_factor)
    south, west, north, east = download_bbox
    zoom = choose_preview_zoom(south, west, north, east)

    center_x, center_y = latlon_to_world_px(lat, lon, zoom)
    view_left = center_x - PREVIEW_WIDTH / 2
    view_top = center_y - PREVIEW_HEIGHT / 2

    tile_x_min = int(math.floor(view_left / OSM_TILE_SIZE))
    tile_y_min = int(math.floor(view_top / OSM_TILE_SIZE))
    tile_x_max = int(math.floor((view_left + PREVIEW_WIDTH - 1) / OSM_TILE_SIZE))
    tile_y_max = int(math.floor((view_top + PREVIEW_HEIGHT - 1) / OSM_TILE_SIZE))

    composite = Image.new('RGB', (PREVIEW_WIDTH, PREVIEW_HEIGHT), (232, 232, 232))
    for tile_x in range(tile_x_min, tile_x_max + 1):
        for tile_y in range(tile_y_min, tile_y_max + 1):
            tile_img = fetch_osm_tile(zoom, tile_x, tile_y)
            paste_x = int(tile_x * OSM_TILE_SIZE - view_left)
            paste_y = int(tile_y * OSM_TILE_SIZE - view_top)
            composite.paste(tile_img, (paste_x, paste_y))

    draw = ImageDraw.Draw(composite)
    download_rect = bbox_to_preview_rect(*download_bbox, zoom, view_left, view_top)
    render_rect = bbox_to_preview_rect(*render_bbox, zoom, view_left, view_top)
    draw.rectangle(download_rect, outline=(0, 120, 255), width=2)
    draw.rectangle(render_rect, outline=(220, 40, 40), width=3)

    return composite

def slim_feature_gdf(gdf):
    if gdf.empty:
        return gdf
    keep_cols = ['geometry'] + [c for c in FEATURE_COLUMNS if c in gdf.columns]
    gdf = gdf[keep_cols].copy()
    geom_mask = gdf.geometry.type.isin(DRAWABLE_GEOM_TYPES)
    return gdf.loc[geom_mask].reset_index(drop=True)

def filter_gdf_by_box(gdf, bbox):
    if gdf.empty:
        return gdf
    indices = gdf.sindex.query(bbox, predicate='intersects')
    return gdf.iloc[list(indices)]

def get_roads_for_cell(gdf_edges_utm, bbox):
    if gdf_edges_utm.empty:
        return []
    indices = gdf_edges_utm.sindex.query(bbox, predicate='intersects')
    if len(indices) == 0:
        return []
    roads = []
    for idx, row in gdf_edges_utm.iloc[list(indices)].iterrows():
        highway = row.get('highway')
        if highway:
            roads.append((get_road_priority(highway), idx, row))
    roads.sort(key=lambda x: x[0])
    return roads

def prepare_feature_layers(gdf_features_utm):
    polys = gdf_features_utm[gdf_features_utm.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    lines = gdf_features_utm[gdf_features_utm.geometry.type.isin(['LineString', 'MultiLineString'])].copy()

    if not polys.empty:
        polys['is_water'] = False
        polys['is_sand'] = False
        for col in ['natural', 'waterway', 'landuse']:
            if col in polys.columns:
                water_mask = polys[col].astype(str).str.lower().isin(['water', 'wetland', 'bay', 'reservoir'])
                polys.loc[water_mask, 'is_water'] = True
        for col in ['natural', 'landuse']:
            if col in polys.columns:
                sand_mask = polys[col].astype(str).str.lower().isin(['sand', 'beach'])
                polys.loc[sand_mask, 'is_sand'] = True

    if not lines.empty:
        lines['is_water_line'] = False
        for col in ['natural', 'waterway']:
            if col in lines.columns:
                water_line_mask = lines[col].astype(str).str.lower().isin([
                    'coastline', 'river', 'stream', 'canal', 'ditch'
                ])
                lines.loc[water_line_mask, 'is_water_line'] = True

    return polys, lines

def draw_map_cell(ax, xmin, ymin, xmax, ymax, cell_polys, cell_lines, cell_roads,
                  meters_per_pixel, road_width_scale):
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           facecolor=PALETTE['light_grass'], edgecolor='none', zorder=0))

    if not cell_polys.empty:
        for col, func in [('natural', get_natural_color), ('landuse', get_landuse_color)]:
            if col in cell_polys.columns:
                colors = cell_polys[col].apply(func)
                mask = colors.notna()
                if col == 'natural':
                    water_mask = cell_polys[col].astype(str).str.lower().isin(['water', 'wetland', 'bay', 'coastline'])
                    mask = mask & ~water_mask
                if mask.any():
                    cell_polys[mask].plot(ax=ax, color=colors[mask], linewidth=0, zorder=1)

        if 'is_water' in cell_polys.columns:
            water_polys = cell_polys[cell_polys['is_water']]
            if not water_polys.empty:
                water_polys.plot(ax=ax, color=PALETTE['water'], linewidth=0, zorder=2)

        if 'is_sand' in cell_polys.columns:
            sand_polys = cell_polys[cell_polys['is_sand']]
            if not sand_polys.empty:
                sand_polys.plot(ax=ax, color=PALETTE['sand'], linewidth=0, zorder=3)

    if not cell_lines.empty and 'is_water_line' in cell_lines.columns:
        water_line_features = cell_lines[cell_lines['is_water_line']]
        for _, row in water_line_features.iterrows():
            waterway_type = row.get('waterway', '')
            natural_type = row.get('natural', '')
            if natural_type == 'coastline':
                lw = 4
            elif waterway_type in ['river', 'canal']:
                lw = 3
            else:
                lw = 2
            try:
                if hasattr(row.geometry, 'xy'):
                    x, y = row.geometry.xy
                    ax.plot(x, y, color=PALETTE['water'], linewidth=lw, solid_capstyle='round', zorder=2)
            except Exception as e:
                print(f"Error drawing a water line: {e}")

    for _, _, row in cell_roads:
        highway = row.get('highway')
        surface = row.get('surface')
        color = get_road_color(highway, surface)
        width_m = get_road_width_m(highway)
        lw = max((width_m / meters_per_pixel) * 0.01 * road_width_scale, 0.1)
        try:
            x, y = row.geometry.xy
            ax.plot(x, y, color=color, linewidth=lw, solid_capstyle='round', zorder=4)
        except Exception as e:
            print(f"Error drawing a road: {e}")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    ax.set_aspect('equal')

def render_cell(output_path, xmin, ymin, xmax, ymax, cell_polys, cell_lines, cell_roads,
                meters_per_pixel, road_width_scale):
    fig, ax = plt.subplots(figsize=(CELL_PX / RENDER_DPI, CELL_PX / RENDER_DPI), dpi=RENDER_DPI)
    fig.subplots_adjust(0, 0, 1, 1)
    draw_map_cell(ax, xmin, ymin, xmax, ymax, cell_polys, cell_lines, cell_roads,
                  meters_per_pixel, road_width_scale)
    for artist in ax.get_children():
        if hasattr(artist, 'set_antialiased'):
            artist.set_antialiased(False)
    plt.savefig(output_path, dpi=RENDER_DPI, pad_inches=0, bbox_inches='tight')
    plt.close(fig)

def stitch_cells(cells_dir, nb_cells, filename_fn, output_path):
    total_px = CELL_PX * nb_cells
    complete = Image.new('RGB', (total_px, total_px))
    for row in range(nb_cells):
        for col in range(nb_cells):
            cell_path = os.path.join(cells_dir, filename_fn(col, row))
            with Image.open(cell_path) as cell_img:
                complete.paste(cell_img.convert('RGB'), (col * CELL_PX, row * CELL_PX))
    complete.save(output_path)
    complete.close()

def cleanup_cache():
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache directory '{cache_dir}' has been removed.")

def sanitize_output_name(name):
    name = name.strip()
    if not name:
        return None
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.strip('. ')
    if not name:
        raise ValueError("Output folder name is invalid")
    return name

def get_output_dir(lat, lon, nb_cells, output_name=None):
    custom_name = sanitize_output_name(output_name or "")
    if custom_name:
        return os.path.join("output", custom_name)
    lat_str = f"{lat:.6f}".rstrip('0').rstrip('.')
    lon_str = f"{lon:.6f}".rstrip('0').rstrip('.')
    return os.path.join("output", f"{lat_str}_{lon_str}_{nb_cells}x{nb_cells}")

def cell_paths(output_dir, veg_output_dir, col, row):
    map_path = os.path.join(output_dir, f"{col},{row}.png")
    veg_path = os.path.join(veg_output_dir, f"{col},{row}_veg.png")
    return map_path, veg_path

def cell_is_complete(map_path, veg_path):
    return (
        os.path.isfile(map_path) and os.path.getsize(map_path) > 0
        and os.path.isfile(veg_path) and os.path.getsize(veg_path) > 0
    )

def count_complete_cells(output_dir, veg_output_dir, nb_cells):
    complete = 0
    for row in range(nb_cells):
        for col in range(nb_cells):
            map_path, veg_path = cell_paths(output_dir, veg_output_dir, col, row)
            if cell_is_complete(map_path, veg_path):
                complete += 1
    return complete

def generate_map_grid(lat, lon, nb_cells, road_width_scale, margin_factor, status_label,
                      resume=False, output_name=None):
    try:
        output_base = get_output_dir(lat, lon, nb_cells, output_name)
        os.makedirs(output_base, exist_ok=True)
        print(f"Output directory: {output_base}")

        status_label.config(text="Downloading OSM data... Please wait.", fg="orange")
        print("Step 1: Downloading OSM data...")
        root.update()

        total_zone_m = CELL_SIZE_M * nb_cells
        margin_m = compute_download_margin_m(total_zone_m, margin_factor)
        download_zone_m = total_zone_m + 2 * margin_m
        dist = download_zone_m / 2

        print(f"Rendering zone: {total_zone_m}m x {total_zone_m}m")
        print(f"Download margin: {margin_m:.0f}m (capped at {MAX_DOWNLOAD_MARGIN_M}m for large maps)")
        print(f"Download zone: {download_zone_m:.0f}m x {download_zone_m:.0f}m")

        print("Downloading road network...")
        simplify_roads = dist > 5000
        G = ox.graph_from_point((lat, lon), dist=dist, network_type='all',
                                simplify=simplify_roads, retain_all=True, truncate_by_edge=True)
        gdf_edges = ox.graph_to_gdfs(G, nodes=False)
        del G
        gc.collect()
        print(f"Roads downloaded: {len(gdf_edges)}")

        status_label.config(text="Downloading map features...", fg="orange")
        print("Step 2: Downloading map features (terrain/water only)...")
        root.update()

        gdf_features = ox.features_from_point((lat, lon), tags=FEATURE_TAGS, dist=dist)
        print(f"Features downloaded: {len(gdf_features)}")
        gdf_features = slim_feature_gdf(gdf_features)
        print(f"Features after filtering to drawable terrain/water: {len(gdf_features)}")
        gc.collect()

        status_label.config(text="Projecting geometries...", fg="orange")
        print("Step 3: Projecting geometries...")
        root.update()

        gdf_edges_utm = ox.projection.project_gdf(gdf_edges)
        utm_crs = gdf_edges_utm.crs
        gdf_features_utm = gdf_features.to_crs(utm_crs)

        center_x = gdf_edges_utm.geometry.centroid.x.mean()
        center_y = gdf_edges_utm.geometry.centroid.y.mean()
        print(f"Map center: ({center_x:.0f}, {center_y:.0f})")

        xmin = center_x - total_zone_m / 2
        xmax = center_x + total_zone_m / 2
        ymin = center_y - total_zone_m / 2
        ymax = center_y + total_zone_m / 2
        print(f"Rendering zone: x[{xmin:.0f}, {xmax:.0f}], y[{ymin:.0f}, {ymax:.0f}]")

        total_bbox = box(xmin, ymin, xmax, ymax)
        gdf_features_utm = gdf_features_utm.clip(total_bbox)
        gdf_edges_utm = gdf_edges_utm.clip(total_bbox)
        del gdf_edges, gdf_features
        gc.collect()

        total_map_px = CELL_PX * nb_cells
        meters_per_pixel = total_zone_m / total_map_px

        plt.rcParams['path.simplify'] = False
        plt.rcParams['agg.path.chunksize'] = 0
        plt.rcParams['lines.antialiased'] = False

        polys, lines = prepare_feature_layers(gdf_features_utm)
        print(f"Prepared {len(polys)} polygons, {len(lines)} lines, {len(gdf_edges_utm)} roads")

        output_dir = os.path.join(output_base, "map_cells")
        veg_output_dir = os.path.join(output_base, "map_vegetation")
        if resume:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(veg_output_dir, exist_ok=True)
        else:
            for dir_name in [output_dir, veg_output_dir]:
                if os.path.exists(dir_name):
                    shutil.rmtree(dir_name)
                os.makedirs(dir_name, exist_ok=True)

        total_cells = nb_cells * nb_cells
        if resume:
            existing_cells = count_complete_cells(output_dir, veg_output_dir, nb_cells)
            print(f"Resume mode: {existing_cells}/{total_cells} cells already complete")

        status_label.config(text="Rendering map cells...", fg="orange")
        print(f"Step 4: Rendering {total_cells} cells one at a time...")
        root.update()

        for row in range(nb_cells):
            for col in range(nb_cells):
                cell_index = row * nb_cells + col + 1
                map_cell_path, veg_cell_path = cell_paths(output_dir, veg_output_dir, col, row)

                if resume and cell_is_complete(map_cell_path, veg_cell_path):
                    status_label.config(
                        text=f"Skipped cell {cell_index}/{total_cells} ({col},{row}) — already done",
                        fg="orange"
                    )
                    if cell_index % 10 == 0 or cell_index == total_cells:
                        print(f"Skipped cell {cell_index}/{total_cells}: ({col},{row})")
                    root.update()
                    continue

                cell_xmin = xmin + col * CELL_SIZE_M
                cell_xmax = cell_xmin + CELL_SIZE_M
                cell_ymax = ymax - row * CELL_SIZE_M
                cell_ymin = cell_ymax - CELL_SIZE_M
                cell_bbox = box(cell_xmin, cell_ymin, cell_xmax, cell_ymax)

                cell_polys = filter_gdf_by_box(polys, cell_bbox)
                cell_lines = filter_gdf_by_box(lines, cell_bbox)
                cell_roads = get_roads_for_cell(gdf_edges_utm, cell_bbox)

                if resume and os.path.isfile(map_cell_path) and os.path.getsize(map_cell_path) > 0:
                    print(f"Reusing map tile for ({col},{row}), generating vegetation only")
                else:
                    render_cell(
                        map_cell_path, cell_xmin, cell_ymin, cell_xmax, cell_ymax,
                        cell_polys, cell_lines, cell_roads, meters_per_pixel, road_width_scale
                    )

                with Image.open(map_cell_path) as cell_img:
                    veg_array = classify_vegetation_color_vectorized(np.array(cell_img.convert('RGB')))
                Image.fromarray(veg_array).save(veg_cell_path)

                status_label.config(
                    text=f"Rendered cell {cell_index}/{total_cells} ({col},{row})...",
                    fg="orange"
                )
                if cell_index % 10 == 0 or cell_index == total_cells:
                    print(f"Rendered cell {cell_index}/{total_cells}: ({col},{row})")
                if cell_index % 100 == 0:
                    gc.collect()
                root.update()

        del polys, lines, gdf_features_utm, gdf_edges_utm
        gc.collect()

        complete_map_filename = os.path.join(output_base, "complete_map.png")
        complete_veg_filename = os.path.join(output_base, "complete_vegetation_map.png")
        complete_cells = count_complete_cells(output_dir, veg_output_dir, nb_cells)
        if total_map_px <= MAX_COMPLETE_MAP_PX and complete_cells == total_cells:
            status_label.config(text="Stitching complete preview maps...", fg="orange")
            print("Step 5: Stitching complete preview maps...")
            root.update()
            stitch_cells(output_dir, nb_cells, lambda c, r: f"{c},{r}.png", complete_map_filename)
            stitch_cells(veg_output_dir, nb_cells, lambda c, r: f"{c},{r}_veg.png", complete_veg_filename)
            print(f"Complete map saved: {complete_map_filename}")
            print(f"Complete vegetation map saved: {complete_veg_filename}")
        elif total_map_px > MAX_COMPLETE_MAP_PX:
            print(
                f"Skipping complete preview maps ({total_map_px}px exceeds {MAX_COMPLETE_MAP_PX}px limit). "
                f"Individual tiles are in '{output_dir}' and '{veg_output_dir}'."
            )
        else:
            print(
                f"Skipping complete preview maps ({complete_cells}/{total_cells} cells ready). "
                "Enable resume and run again to finish remaining tiles."
            )

        status_label.config(text=f"{nb_cells}x{nb_cells} grids generated in '{output_base}'.", fg="#004d00")
        print(f"Map grid generation completed: {complete_cells}/{total_cells} tiles in '{output_dir}' and '{veg_output_dir}'.")

    except Exception as e:
        showerror("Error", f"Map generation error: {e}")
        status_label.config(text="Error during map generation.", fg="red")
        print(f"ERROR during map generation: {e}")

def generate_vegetation_maps(lat, lon, nb_cells, status_label, output_name=None):
    try:
        output_base = get_output_dir(lat, lon, nb_cells, output_name)
        input_dir = os.path.join(output_base, "map_cells")
        output_dir = os.path.join(output_base, "map_vegetation")

        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"No map cells found in '{input_dir}'. Generate maps first.")

        status_label.config(text="Starting vegetation map generation...", fg="orange")
        print("Vegetation: start processing images.")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        filenames = [f for f in os.listdir(input_dir) if f.endswith(".png")]
        total_files = len(filenames)
        print(f"Vegetation: found {total_files} files to process.")

        for idx, filename in enumerate(filenames, 1):
            status_label.config(text=f"Processing {filename} ({idx}/{total_files})...", fg="orange")
            print(f"Vegetation: processing {filename} ({idx}/{total_files})")
            root.update()

            path = os.path.join(input_dir, filename)
            img = Image.open(path).convert("RGB")
            img_array = np.array(img)
            new_array = classify_vegetation_color_vectorized(img_array)

            base, _ = os.path.splitext(filename)
            Image.fromarray(new_array).save(os.path.join(output_dir, f"{base}_veg.png"))

        status_label.config(text=f"Vegetation maps generated in '{output_dir}'.", fg="#004d00")
        print(f"Vegetation: generation completed, saved in '{output_dir}'.")

    except Exception as e:
        showerror("Error", f"Vegetation generation error: {e}")
        status_label.config(text="Error during vegetation generation.", fg="red")
        print(f"ERROR during vegetation generation: {e}")





# ========== GUI ==========
root = tk.Tk()
root.title("OSM + Vegetation Map Generator")

preview_frame = tk.Frame(root, padx=10, pady=10)
preview_frame.pack(side='top', fill='x')
tk.Label(preview_frame, text="Generation area preview").pack(anchor='w')
preview_label = tk.Label(
    preview_frame, bg='#e8e8e8', text="Loading preview...", fg='gray',
)
preview_label.pack(pady=2)
preview_info = tk.Label(
    preview_frame, text="", fg='gray', font=('TkDefaultFont', 8), justify='left',
)
preview_info.pack(anchor='w', pady=2)
tk.Label(
    preview_frame, text="© OpenStreetMap contributors", fg='gray',
    font=('TkDefaultFont', 7),
).pack(anchor='e')

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(side='top')

STATUS_WRAP = 380
status_frame = tk.Frame(root, height=64)
status_frame.pack(side='bottom', fill='x', padx=10, pady=5)
status_frame.pack_propagate(False)
status_label = tk.Label(
    status_frame, text="", fg="green", wraplength=STATUS_WRAP,
    justify='left', anchor='nw',
)
status_label.pack(fill='both', expand=True, anchor='w')

preview_job = None
preview_request_id = 0

def apply_preview_image(photo, info_text):
    preview_label.config(image=photo, text='')
    preview_label.image = photo
    preview_info.config(text=info_text)

def show_preview_error(message):
    preview_label.config(image='', text=message)
    preview_label.image = None
    preview_info.config(text='')

def update_preview():
    global preview_request_id
    preview_request_id += 1
    request_id = preview_request_id
    try:
        lat = float(lat_entry.get())
        lon = float(lon_entry.get())
        nb_cells = int(cells_entry.get())
        margin_factor = float(margin_entry.get())
        if nb_cells < 1:
            raise ValueError("grid size")
    except ValueError:
        show_preview_error("Enter valid latitude, longitude, and grid size")
        return

    side_km = (CELL_SIZE_M * nb_cells) / 1000
    info_text = (
        f"{nb_cells}×{nb_cells} cells · {side_km:.2f} km/side  |  "
        f"Red = generated map  ·  Blue = OSM download buffer"
    )

    def worker():
        try:
            image = build_osm_preview(lat, lon, nb_cells, margin_factor)
            if request_id == preview_request_id:
                def apply():
                    photo = ImageTk.PhotoImage(image)
                    apply_preview_image(photo, info_text)
                root.after(0, apply)
        except (URLError, TimeoutError, ValueError) as exc:
            if request_id == preview_request_id:
                root.after(0, lambda: show_preview_error(f"Preview unavailable: {exc}"))
        except Exception as exc:
            if request_id == preview_request_id:
                root.after(0, lambda: show_preview_error(f"Preview unavailable: {exc}"))

    threading.Thread(target=worker, daemon=True).start()

def schedule_preview_update(event=None):
    global preview_job
    if preview_job is not None:
        root.after_cancel(preview_job)
    preview_job = root.after(400, update_preview)

def add_entry(label_text, default_value, row, tooltip=None):
    tk.Label(frame, text=label_text).grid(row=row, column=0, sticky='e')
    entry = tk.Entry(frame, width=25)
    entry.insert(0, str(default_value))
    entry.grid(row=row, column=1)
    if tooltip:
        def on_enter(event): status_label.config(text=tooltip, fg="gray")
        def on_leave(event): status_label.config(text="", fg="green")
        entry.bind("<Enter>", on_enter)
        entry.bind("<Leave>", on_leave)
    return entry

lat_entry = add_entry("Latitude:", 47.80328791813283, 0, "Latitude of the map center point.")
lon_entry = add_entry("Longitude:", -3.7209986709586205, 1, "Longitude of the map center point.")
cells_entry = add_entry("Number of cells (NxN):", 2, 2,
                        "Number of cells per side (e.g., 2 means 4 map tiles).")
margin_entry = add_entry("Download margin (%):", 0.8, 3,
                         "Extra download buffer as % of map size (300m–2500m; large maps use the cap).")
width_entry = add_entry("Road width scale:", 100, 4,
                        "Scale factor for road widths on the generated maps.")
output_name_entry = add_entry("Output folder name:", "", 5,
                              "Optional name under output/. Leave blank for lat_lon_NxN naming.")

for entry in (lat_entry, lon_entry, cells_entry, margin_entry):
    entry.bind('<KeyRelease>', schedule_preview_update)
schedule_preview_update()

resume_var = tk.BooleanVar(value=False)
resume_frame = tk.Frame(frame)
resume_frame.grid(row=6, column=0, columnspan=2, sticky='w')
resume_label = tk.Label(resume_frame, text="Resume incomplete generation")
resume_label.pack(side='left')
resume_check = tk.Checkbutton(resume_frame, variable=resume_var)
resume_check.pack(side='left')
for widget in (resume_frame, resume_label, resume_check):
    widget.bind("<Enter>", lambda e: status_label.config(
        text="Keep existing tiles and only render missing cells. Uncheck to wipe and start over.",
        fg="gray",
    ))
    widget.bind("<Leave>", lambda e: status_label.config(text="", fg="green"))

def on_generate_maps():
    try:
        lat = float(lat_entry.get())
        lon = float(lon_entry.get())
        n = int(cells_entry.get())
        margin = float(margin_entry.get())
        width_scale = float(width_entry.get())
        resume = resume_var.get()
        output_name = output_name_entry.get()
        if n < 1:
            raise ValueError("Number of cells must be ≥ 1")
        if margin < 0:
            raise ValueError("Margin must be ≥ 0")
        generate_map_grid(lat, lon, n, width_scale, margin, status_label,
                          resume=resume, output_name=output_name)
    except Exception as e:
        showerror("Error", f"Invalid parameter: {e}")
        status_label.config(text="Parameter error.", fg="red")

def on_generate_vegetation():
    try:
        lat = float(lat_entry.get())
        lon = float(lon_entry.get())
        n = int(cells_entry.get())
        output_name = output_name_entry.get()
        generate_vegetation_maps(lat, lon, n, status_label, output_name=output_name)
    except Exception as e:
        showerror("Error", f"Invalid parameter: {e}")
        status_label.config(text="Parameter error.", fg="red")

tk.Button(frame, text="Generate Maps + Vegetation", command=on_generate_maps).grid(row=7, column=0, columnspan=2, pady=5)

root.update_idletasks()
root.minsize(PREVIEW_WIDTH + 40, root.winfo_height())

root.mainloop()