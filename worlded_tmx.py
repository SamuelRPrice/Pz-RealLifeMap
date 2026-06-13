"""Generate WorldEd-compatible TMX files from complete_map PNGs."""

import base64
import gzip
import os
import re
import struct
import subprocess

import numpy as np
from PIL import Image

CELL_PX = 300
DEFAULT_WORLDED_PATH = (
    r"C:\Users\Sam\Documents\projects\zomboid mapping\pz-tools-ce_42_16_rev3"
)
WORLDED_EXE = os.path.join("WorldEd", "PZWorldEd.exe")

_skeleton_cache = {}


def find_reference_tmx(worlded_dir):
    maps_dir = os.path.join(worlded_dir, "maps")
    if not os.path.isdir(maps_dir):
        return None
    for root, _dirs, files in os.walk(maps_dir):
        if "complete_map_0_0.tmx" in files:
            return os.path.join(root, "complete_map_0_0.tmx")
    for root, _dirs, files in os.walk(maps_dir):
        for name in sorted(files):
            if name.endswith(".tmx"):
                return os.path.join(root, name)
    return None


def get_tmx_skeleton(worlded_dir, export_dir):
    cache_key = (os.path.normcase(worlded_dir), os.path.normcase(export_dir))
    if cache_key in _skeleton_cache:
        return _skeleton_cache[cache_key]

    ref_path = find_reference_tmx(worlded_dir)
    if not ref_path:
        raise FileNotFoundError(
            f"No reference .tmx found under {maps_dir_path(worlded_dir)}. "
            "Run WorldEd BMP-to-TMX once on any map to create a template."
        )

    with open(ref_path, encoding="utf-8") as ref_file:
        content = ref_file.read()

    marker = "<bmp-image"
    split_at = content.find(marker)
    if split_at == -1:
        raise ValueError(f"Reference TMX is missing bmp-image data: {ref_path}")

    skeleton = content[:split_at].rstrip() + "\n"
    worlded_rel = os.path.relpath(
        os.path.join(worlded_dir, "WorldEd"), export_dir
    ).replace("\\", "/")
    skeleton = re.sub(
        r'(<rules-file file=")[^"]*(")',
        rf'\g<1>{worlded_rel}/Rules.txt\2',
        skeleton,
        count=1,
    )
    skeleton = re.sub(
        r'(<blends-file file=")[^"]*(")',
        rf'\g<1>{worlded_rel}/Blends.txt\2',
        skeleton,
        count=1,
    )

    _skeleton_cache[cache_key] = skeleton
    return skeleton


def maps_dir_path(worlded_dir):
    return os.path.join(worlded_dir, "maps")


def stitch_cell_grid(cells_dir, nb_cells, filename_fn, output_path):
    total_px = CELL_PX * nb_cells
    complete = Image.new("RGB", (total_px, total_px))
    for row in range(nb_cells):
        for col in range(nb_cells):
            cell_path = os.path.join(cells_dir, filename_fn(col, row))
            with Image.open(cell_path) as cell_img:
                complete.paste(cell_img.convert("RGB"), (col * CELL_PX, row * CELL_PX))
    complete.save(output_path)
    complete.close()


def ensure_complete_maps(output_base, nb_cells):
    main_path = os.path.join(output_base, "complete_map.png")
    veg_path = os.path.join(output_base, "complete_map_veg.png")
    cells_dir = os.path.join(output_base, "map_cells")
    veg_dir = os.path.join(output_base, "map_vegetation")

    if not os.path.isfile(main_path):
        stitch_cell_grid(cells_dir, nb_cells, lambda c, r: f"{c},{r}.png", main_path)
    if not os.path.isfile(veg_path):
        stitch_cell_grid(veg_dir, nb_cells, lambda c, r: f"{c},{r}_veg.png", veg_path)
    return main_path, veg_path


def collect_colors(rgb_array):
    pixels = rgb_array.reshape(-1, 3)
    unique = set()
    for r, g, b in pixels:
        ri, gi, bi = int(r), int(g), int(b)
        if ri == 0 and gi == 0 and bi == 0:
            continue
        unique.add((ri, gi, bi))
    return sorted(unique)


def encode_bmp_pixels(rgb_array):
    colors = collect_colors(rgb_array)
    color_to_index = {color: index + 1 for index, color in enumerate(colors)}

    flat = rgb_array.reshape(-1, 3)
    indices = []
    for r, g, b in flat:
        key = (int(r), int(g), int(b))
        indices.append(0 if key == (0, 0, 0) else color_to_index[key])

    tile_data = struct.pack(f"<{len(indices)}I", *indices)
    return colors, base64.b64encode(gzip.compress(tile_data)).decode("ascii")


def format_bmp_image_xml(index, rgb_array, seed=1):
    colors, pixel_data = encode_bmp_pixels(rgb_array)
    lines = [f' <bmp-image index="{index}" seed="{seed}">']
    for r, g, b in colors:
        lines.append(f'  <color rgb="{r} {g} {b}"/>')
    lines.append("  <pixels>")
    lines.append(f"   {pixel_data}")
    lines.append("  </pixels>")
    lines.append(" </bmp-image>")
    return "\n".join(lines)


def tmx_filename(prefix, col, row, world_origin=(0, 0)):
    ox, oy = world_origin
    return f"{prefix}_{ox + col}_{oy + row}.tmx"


def export_dir_for_world(worlded_dir, world_name):
    return os.path.join(maps_dir_path(worlded_dir), world_name)


def generate_tmx_grid(main_image_path, veg_image_path, nb_cells, export_dir, worlded_dir,
                      prefix="complete_map", world_origin=(0, 0), status_callback=None):
    os.makedirs(export_dir, exist_ok=True)
    skeleton = get_tmx_skeleton(worlded_dir, export_dir)

    with Image.open(main_image_path) as main_img, Image.open(veg_image_path) as veg_img:
        main_rgb = main_img.convert("RGB")
        veg_rgb = veg_img.convert("RGB")
        main_w, main_h = main_rgb.size
        veg_w, veg_h = veg_rgb.size

        if (main_w, main_h) != (veg_w, veg_h):
            raise ValueError("Main and vegetation images must be the same size")
        if main_w % CELL_PX or main_h % CELL_PX:
            raise ValueError(f"Image dimensions must be divisible by {CELL_PX}")

        grid_w = main_w // CELL_PX
        grid_h = main_h // CELL_PX
        if grid_w != nb_cells or grid_h != nb_cells:
            raise ValueError(
                f"Image is {grid_w}x{grid_h} cells but expected {nb_cells}x{nb_cells}"
            )

        total = nb_cells * nb_cells
        written = 0
        for row in range(nb_cells):
            for col in range(nb_cells):
                x0 = col * CELL_PX
                y0 = row * CELL_PX
                main_crop = np.array(main_rgb.crop((x0, y0, x0 + CELL_PX, y0 + CELL_PX)))
                veg_crop = np.array(veg_rgb.crop((x0, y0, x0 + CELL_PX, y0 + CELL_PX)))

                tmx_path = os.path.join(
                    export_dir, tmx_filename(prefix, col, row, world_origin)
                )
                with open(tmx_path, "w", encoding="utf-8", newline="\n") as tmx_file:
                    tmx_file.write(skeleton)
                    tmx_file.write(format_bmp_image_xml(0, main_crop) + "\n")
                    tmx_file.write(format_bmp_image_xml(1, veg_crop) + "\n")
                    tmx_file.write("</map>\n")

                written += 1
                if status_callback:
                    status_callback(written, total, col, row)

    return written


def _rel_path(from_dir, target):
    return os.path.relpath(os.path.abspath(target), os.path.abspath(from_dir)).replace("\\", "/")


def write_pzw(world_path, worlded_dir, export_dir, main_image_path, nb_cells,
              world_origin=(0, 0), prefix="complete_map"):
    template_path = os.path.join(worlded_dir, "WorldEd", "test", "untitled.pzw")
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"WorldEd template not found: {template_path}")

    with open(template_path, encoding="utf-8") as template_file:
        content = template_file.read()

    pzw_dir = os.path.dirname(os.path.abspath(world_path))
    worlded_root = os.path.abspath(worlded_dir)
    export_abs = os.path.abspath(export_dir)
    main_abs = os.path.abspath(main_image_path)

    content = re.sub(
        r'<world version="1\.0" width="\d+" height="\d+">',
        f'<world version="1.0" width="{nb_cells}" height="{nb_cells}">',
        content,
        count=1,
    )
    content = re.sub(
        r'(<tmxexportdir path=")[^"]*(")',
        rf'\1{_rel_path(pzw_dir, export_abs)}\2',
        content,
        count=1,
    )
    for tag, target in (
        ("rulesfile", os.path.join(worlded_root, "WorldEd", "Rules.txt")),
        ("blendsfile", os.path.join(worlded_root, "WorldEd", "Blends.txt")),
        ("mapbasefile", os.path.join(worlded_root, "WorldEd", "MapBaseXML.txt")),
    ):
        content = re.sub(
            rf'(<{tag} path=")[^"]*(")',
            rf'\1{_rel_path(pzw_dir, target)}\2',
            content,
            count=1,
        )

    bmp_line = (
        f' <bmp path="{_rel_path(pzw_dir, main_abs)}" '
        f'x="{world_origin[0]}" y="{world_origin[1]}" '
        f'width="{nb_cells}" height="{nb_cells}"/>'
    )
    if re.search(r"<bmp\s", content):
        content = re.sub(r"<bmp[^>]*/>", bmp_line, content, count=1)
    else:
        content = content.replace("</LuaSettings>", f"</LuaSettings>\n{bmp_line}", 1)

    content = re.sub(r"\s*<cell[^>]*/>", "", content)

    cell_lines = []
    for row in range(nb_cells):
        for col in range(nb_cells):
            tmx_path = os.path.join(export_abs, tmx_filename(prefix, col, row, world_origin))
            cell_lines.append(
                f' <cell x="{col}" y="{row}" map="{_rel_path(pzw_dir, tmx_path)}"/>'
            )

    content = re.sub(r"</world>\s*$", "\n".join(cell_lines) + "\n</world>", content.rstrip())

    with open(world_path, "w", encoding="utf-8", newline="\n") as pzw_file:
        pzw_file.write(content)
    return world_path


def launch_worlded(worlded_dir, pzw_path):
    exe_path = os.path.join(worlded_dir, WORLDED_EXE)
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"PZWorldEd.exe not found: {exe_path}")
    subprocess.Popen([exe_path, os.path.abspath(pzw_path)], cwd=os.path.dirname(exe_path))


def generate_worlded_output(output_base, nb_cells, worlded_dir, world_name=None,
                            open_worlded=False, status_callback=None):
    worlded_dir = os.path.abspath(worlded_dir)
    if not os.path.isdir(worlded_dir):
        raise FileNotFoundError(f"pz-tools directory not found: {worlded_dir}")

    world_name = world_name or os.path.basename(output_base)
    main_path, veg_path = ensure_complete_maps(output_base, nb_cells)
    export_dir = export_dir_for_world(worlded_dir, world_name)

    def tmx_status(done, total, col, row):
        if status_callback:
            status_callback(f"TMX {done}/{total} ({col},{row})")

    count = generate_tmx_grid(
        main_path, veg_path, nb_cells, export_dir, worlded_dir,
        status_callback=tmx_status,
    )

    pzw_name = f"{world_name}.pzw"
    pzw_path = os.path.join(os.path.abspath(output_base), pzw_name)
    write_pzw(pzw_path, worlded_dir, export_dir, main_path, nb_cells)

    worlded_copy = os.path.join(worlded_dir, "WorldEd", "test", pzw_name)
    write_pzw(worlded_copy, worlded_dir, export_dir, main_path, nb_cells)

    if open_worlded:
        launch_worlded(worlded_dir, pzw_path)

    return {
        "tmx_count": count,
        "export_dir": export_dir,
        "pzw_path": pzw_path,
        "pzw_worlded_path": worlded_copy,
        "main_map": main_path,
        "veg_map": veg_path,
    }
