"""Export generated map PNGs to WorldEd TMX files and a .pzw world."""

import argparse
import json
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image

import worlded_tmx

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worlded_export_settings.json")
DEFAULT_SETTINGS = {
    "output_dir": "",
    "nb_cells": "",
    "world_name": "",
    "worlded_path": worlded_tmx.DEFAULT_WORLDED_PATH,
    "open_worlded": False,
}


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as settings_file:
            saved = json.load(settings_file)
        if isinstance(saved, dict):
            for key in DEFAULT_SETTINGS:
                if key in saved:
                    settings[key] = saved[key]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return settings


def save_settings(settings):
    payload = {key: settings.get(key, DEFAULT_SETTINGS[key]) for key in DEFAULT_SETTINGS}
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as settings_file:
            json.dump(payload, settings_file, indent=2)
    except OSError as exc:
        print(f"Could not save settings: {exc}")


def infer_nb_cells(output_base):
    folder_name = os.path.basename(os.path.abspath(output_base))
    match = re.search(r"(\d+)x(\d+)$", folder_name)
    if match and match.group(1) == match.group(2):
        return int(match.group(1))

    main_path = os.path.join(output_base, "complete_map.png")
    if os.path.isfile(main_path):
        with Image.open(main_path) as image:
            width, height = image.size
        if width == height and width % worlded_tmx.CELL_PX == 0:
            return width // worlded_tmx.CELL_PX

    cells_dir = os.path.join(output_base, "map_cells")
    if os.path.isdir(cells_dir):
        max_index = -1
        for name in os.listdir(cells_dir):
            cell_match = re.match(r"(\d+),(\d+)\.png$", name)
            if cell_match:
                max_index = max(max_index, int(cell_match.group(1)), int(cell_match.group(2)))
        if max_index >= 0:
            return max_index + 1

    raise ValueError(
        "Could not infer grid size. Name the folder like 'name_10x10', include complete_map.png, "
        "or pass --cells N."
    )


def validate_output_dir(output_base):
    output_base = os.path.abspath(output_base)
    if not os.path.isdir(output_base):
        raise FileNotFoundError(f"Output folder not found: {output_base}")

    cells_dir = os.path.join(output_base, "map_cells")
    veg_dir = os.path.join(output_base, "map_vegetation")
    main_path = os.path.join(output_base, "complete_map.png")
    if not os.path.isfile(main_path) and not os.path.isdir(cells_dir):
        raise FileNotFoundError(
            f"No map data in '{output_base}'. Expected complete_map.png or map_cells/."
        )
    if not os.path.isfile(os.path.join(output_base, "complete_map_veg.png")) and not os.path.isdir(veg_dir):
        raise FileNotFoundError(
            f"No vegetation data in '{output_base}'. Expected complete_map_veg.png or map_vegetation/."
        )
    return output_base


def run_export(output_base, nb_cells, worlded_dir, world_name=None, open_worlded=False,
                status_callback=None):
    output_base = validate_output_dir(output_base)
    worlded_dir = (worlded_dir or worlded_tmx.DEFAULT_WORLDED_PATH).strip()
    world_name = (world_name or "").strip() or os.path.basename(output_base)

    if status_callback:
        status_callback("Preparing complete map images...")
    result = worlded_tmx.generate_worlded_output(
        output_base,
        nb_cells,
        worlded_dir,
        world_name=world_name,
        open_worlded=open_worlded,
        status_callback=status_callback,
    )
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Export Pz-RealLifeMap output to WorldEd TMX files and a .pzw world.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Path to a generated map folder under output/ (e.g. output/52.457_-2.14_10x10)",
    )
    parser.add_argument(
        "--cells",
        type=int,
        help="Grid size N for an NxN map (auto-detected from folder name or images if omitted)",
    )
    parser.add_argument(
        "--world-name",
        help="Name for TMX export folder and .pzw file (defaults to output folder name)",
    )
    parser.add_argument(
        "--worlded-path",
        default=worlded_tmx.DEFAULT_WORLDED_PATH,
        help="Path to pz-tools-ce installation",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Launch PZWorldEd with the generated .pzw file",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the export window instead of running from the command line",
    )
    return parser


def run_cli(args):
    output_base = validate_output_dir(args.output_dir)
    nb_cells = args.cells or infer_nb_cells(output_base)
    if nb_cells < 1:
        raise ValueError("Number of cells must be >= 1")

    def status(message):
        print(message)

    result = run_export(
        output_base,
        nb_cells,
        args.worlded_path,
        world_name=args.world_name,
        open_worlded=args.open,
        status_callback=status,
    )
    print(f"Wrote {result['tmx_count']} TMX files to: {result['export_dir']}")
    print(f"World file: {result['pzw_path']}")
    print(f"WorldEd copy: {result['pzw_worlded_path']}")
    return result


def run_gui(saved_settings):
    root = tk.Tk()
    root.title("WorldEd Export")

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill="both", expand=True)

    status_label = tk.Label(frame, text="", fg="green", wraplength=420, justify="left")
    status_label.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def add_labeled_entry(label, default, row, width=42):
        tk.Label(frame, text=label).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=3)
        entry = tk.Entry(frame, width=width)
        entry.insert(0, default)
        entry.grid(row=row, column=1, columnspan=2, sticky="we", pady=3)
        return entry

    output_entry = add_labeled_entry("Map output folder:", saved_settings["output_dir"], 0)
    cells_entry = add_labeled_entry("Cells (NxN, optional):", saved_settings["nb_cells"], 1, width=10)
    world_name_entry = add_labeled_entry("World name (optional):", saved_settings["world_name"], 2)
    worlded_path_entry = add_labeled_entry("pz-tools path:", saved_settings["worlded_path"], 3)

    open_var = tk.BooleanVar(value=saved_settings["open_worlded"])
    open_frame = tk.Frame(frame)
    open_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=3)
    tk.Label(open_frame, text="Open WorldEd after export").pack(side="left")
    tk.Checkbutton(open_frame, variable=open_var).pack(side="left")

    def browse_output_dir():
        initial = output_entry.get().strip() or os.path.join(os.getcwd(), "output")
        if not os.path.isdir(initial):
            initial = os.getcwd()
        chosen = filedialog.askdirectory(initialdir=initial, title="Select map output folder")
        if chosen:
            output_entry.delete(0, tk.END)
            output_entry.insert(0, chosen)

    tk.Button(frame, text="Browse...", command=browse_output_dir).grid(row=0, column=3, padx=(8, 0))

    def collect_settings():
        return {
            "output_dir": output_entry.get(),
            "nb_cells": cells_entry.get(),
            "world_name": world_name_entry.get(),
            "worlded_path": worlded_path_entry.get(),
            "open_worlded": open_var.get(),
        }

    def on_export():
        try:
            settings = collect_settings()
            output_base = settings["output_dir"].strip()
            if not output_base:
                raise ValueError("Choose a map output folder")

            cells_text = settings["nb_cells"].strip()
            nb_cells = int(cells_text) if cells_text else infer_nb_cells(output_base)
            world_name = settings["world_name"].strip() or None

            save_settings(settings)
            status_label.config(text="Exporting...", fg="orange")
            root.update()

            def status(message):
                status_label.config(text=message, fg="orange")
                root.update()

            result = run_export(
                output_base,
                nb_cells,
                settings["worlded_path"],
                world_name=world_name,
                open_worlded=settings["open_worlded"],
                status_callback=status,
            )
            status_label.config(
                text=(
                    f"Exported {result['tmx_count']} TMX files.\n"
                    f"Open in WorldEd: {result['pzw_path']}"
                ),
                fg="#004d00",
            )
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            status_label.config(text="Export failed.", fg="red")

    tk.Button(frame, text="Export TMX + .pzw", command=on_export).grid(
        row=6, column=0, columnspan=4, pady=(12, 0),
    )

    def on_close():
        save_settings(collect_settings())
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.gui or not args.output_dir:
        run_gui(load_settings())
        return 0

    try:
        run_cli(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
