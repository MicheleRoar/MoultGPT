#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_annotation.py

Viewer/editor for iNat-style annotations:
- Full-screen window, but reserves a bottom strip for a non-overlapping legend
- Shows organism/exuviae bounding boxes
- Shows pose keypoints
- Replaces skeleton segments with DIRECTION ARROWS (thorax -> head)
  * wide schema: head_org/thorax_org and head_exu/thorax_exu
  * kp_* fallback: if 'thorax' and 'head' exist, draw one arrow
- Delete ORGANISM / EXUVIAE / SEGMENTS with backups
- Diagnostics overlay (keypoints/”arrows”/seg-cols)

Keyboard:
  [h] help toggle
  [n] next row
  [b] back row
  [1] select ORGANISM
  [2] select EXUVIAE
  [3] select SEGMENTS (arrows)
  [d] delete selected (ORG / EXU / SEG)
  [D] delete ALL (boxes + segments)
  [r] restore (from backup created by pressing [1]/[2]/[3])
  [s] save CSV
  [k] toggle arrows visibility
  [K] cycle arrow linewidth
  [f]/[F] re-maximize window
  [q]/[ESC] quit
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import ast

# ---------------- Configuration ----------------
DEFAULT_CSV    = "../../data/inat_raw/inat_dataset.csv"
DEFAULT_IMAGES = "../../data/inat_raw"

# Bounding boxes
ORG_COLS = ['x_organism','y_organism','w_organism','h_organism']
EXU_COLS = ['x_exuviae','y_exuviae','w_exuviae','h_exuviae']

# Arrow colors per category
ORG_ARROW_COLOR = "yellow"  # organism
EXU_ARROW_COLOR = "lime"    # exuviae (green)


# Keypoint wide schema (your other script)
WIDE_KP_PAIRS = {
    'head_org'  : ('x_head_org',   'y_head_org'),
    'thorax_org': ('x_thorax_org', 'y_thorax_org'),
    'head_exu'  : ('x_head_exu',   'y_head_exu'),
    'thorax_exu': ('x_thorax_exu', 'y_thorax_exu'),
}

# Arrow style/colors (coherent with your other scripts)
ARROW_COLOR_DEFAULT  = "lime"    # green
ARROW_COLOR_SELECTED = "yellow"  # highlight when SEGMENTS selected

# Legend strip height (fraction of figure)
LEGEND_H = 0.12  # adjust if you want taller/shorter legend area

# ---------------- Helpers ----------------
def go_fullscreen(fig):
    """Best-effort fullscreen across backends (Qt, Tk, macOS)."""
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, "window"):
            try:
                mng.window.showMaximized()
            except Exception:
                pass
        try:
            mng.window.state("zoomed")
        except Exception:
            pass
        try:
            mng.full_screen_toggle()
        except Exception:
            pass
    except Exception:
        pass

def resolve_image_path(root: Path, stage: str, observation_id, photo_id):
    """Compose <root>/<stage>/<observation>_<photo>.jpg and check existence."""
    try:
        obs = str(int(observation_id))
        pho = str(int(photo_id))
    except Exception:
        return None
    fname = f"{obs}_{pho}.jpg"
    p = root / stage / fname
    return p if p.exists() else None

def is_valid_box(row, cols):
    vals = [row.get(c, np.nan) for c in cols]
    if any(pd.isna(v) for v in vals):
        return False
    try:
        x,y,w,h = map(float, vals)
    except Exception:
        return False
    return all(v >= 0 for v in (x,y,w,h)) and w > 0 and h > 0

def draw_box(ax, row, cols, edgecolor, lw=2, selected=False):
    """Draw a rectangle if bbox is valid."""
    if not is_valid_box(row, cols):
        return None
    x,y,w,h = [float(row[c]) for c in cols]
    rect = patches.Rectangle(
        (x,y), w, h, fill=False,
        edgecolor=('yellow' if selected else edgecolor),
        linewidth=lw + (1 if selected else 0)
    )
    ax.add_patch(rect)
    return rect

def _add_kp(kp_dict, name, x, y, vis=None, conf=None):
    """Add a keypoint if coords are valid."""
    if pd.isna(x) or pd.isna(y):
        return
    try:
        xf, yf = float(x), float(y)
    except Exception:
        return
    if xf < 0 or yf < 0:
        return
    kp_dict[name] = {'x': xf, 'y': yf, 'v': vis, 'c': conf}

def collect_keypoints(row):
    """
    Collect keypoints from:
      - kp_<name>_{x,y} (+ optional _v/_c)
      - wide schema x_head_org/y_head_org, x_thorax_org/y_thorax_org, ...
    """
    kp_dict = {}
    cols = list(row.index)

    # 1) Standard kp_* pairs
    xcols = [c for c in cols if c.startswith('kp_') and c.endswith('_x')]
    for xname in xcols:
        base = xname[:-2]
        yname = base + '_y'
        if yname not in row.index:
            continue
        x = row.get(xname, np.nan)
        y = row.get(yname, np.nan)

        vname = base + '_v'
        cname = base + '_c'
        vis = None
        conf = None
        if vname in row.index and not pd.isna(row[vname]):
            try:
                vis = bool(int(row[vname]))
            except Exception:
                vis = None
        if cname in row.index and not pd.isna(row[cname]):
            try:
                conf = float(row[cname])
            except Exception:
                conf = None

        name = base[3:] if base.startswith('kp_') else base
        _add_kp(kp_dict, name, x, y, vis=vis, conf=conf)

    # 2) Wide schema pairs (if present)
    for name, (cx, cy) in WIDE_KP_PAIRS.items():
        if cx in row.index and cy in row.index:
            _add_kp(kp_dict, name, row.get(cx, np.nan), row.get(cy, np.nan))

    return kp_dict

def draw_keypoints(ax, kp_dict, label=True):
    """Plot keypoints and (optionally) small labels."""
    for name, d in kp_dict.items():
        if d.get('v') is not None and not d['v']:
            continue
        ax.scatter([d['x']], [d['y']], s=25, linewidths=0.5,
                   edgecolors='black', zorder=3)
        if label:
            ax.text(d['x']+3, d['y']+3, name, fontsize=7, color='black',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
                    zorder=4)

def help_text():
    return (
        "[h] help   [n] next   [b] back   [q/ESC] quit   [s] save\n"
        "[1] select ORGANISM   [2] select EXUVIAE   [3] select SEGMENTS (arrows)\n"
        "[d] delete selected   [D] delete ALL (boxes + segments)   [r] restore\n"
        "[k] toggle arrows   [K] thicker arrows   [F] fullscreen"
    )

# --------- Arrow drawing (thorax -> head) ----------
def _has_wide_org(kp):  # both thorax_org and head_org present?
    return ('thorax_org' in kp) and ('head_org' in kp)

def _has_wide_exu(kp):
    return ('thorax_exu' in kp) and ('head_exu' in kp)

def draw_direction_arrows(ax, kp_dict, lw=4.0, selected=False):
    """
    Draw arrows FROM thorax TO head.
    Colors:
      - ORGANISM: yellow
      - EXUVIAE : green (lime)
    If selected=True (SEGMENTS selected), add +1 to linewidth.
    """
    def arrow(p_from, p_to, color):
        width = lw + (1 if selected else 0)
        ax.annotate(
            "",
            xy=(p_to['x'], p_to['y']),
            xytext=(p_from['x'], p_from['y']),
            arrowprops=dict(
                arrowstyle="-|>",   # thicker head
                linewidth=width,
                color=color,
                shrinkA=0, shrinkB=0,
                mutation_scale=12 + width * 2  # bigger arrow head with width
            ),
            zorder=3
        )

    drew_any = 0

    # Wide schema first: draw both if present, with their category colors
    if 'thorax_org' in kp_dict and 'head_org' in kp_dict:
        arrow(kp_dict['thorax_org'], kp_dict['head_org'], ORG_ARROW_COLOR)
        drew_any += 1
    if 'thorax_exu' in kp_dict and 'head_exu' in kp_dict:
        arrow(kp_dict['thorax_exu'], kp_dict['head_exu'], EXU_ARROW_COLOR)
        drew_any += 1

    # Fallback: generic kp_* 'thorax' -> 'head' (treat as organism)
    if drew_any == 0 and ('thorax' in kp_dict and 'head' in kp_dict):
        arrow(kp_dict['thorax'], kp_dict['head'], ORG_ARROW_COLOR)
        drew_any += 1

    return drew_any


# ------------- UI -------------
def run(csv_path: Path, image_root: Path, start_index: int = 0):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    if not image_root.exists():
        raise FileNotFoundError(f"Images root not found: {image_root.resolve()}")

    df = pd.read_csv(csv_path)

    # Required columns
    for c in ['stage','observation_id','photo_id']:
        if c not in df.columns:
            raise ValueError(f"Missing required column: '{c}'")

    # Ensure bbox columns exist
    for c in ORG_COLS + EXU_COLS:
        if c not in df.columns:
            df[c] = -1

    # Soft-create an empty 'segments' column if missing (for deletion)
    if 'segments' not in df.columns:
        df['segments'] = "[]"

    # Discover any seg_<A>_<B> (we still clear these on deletion)
    seg_wide_cols = [c for c in df.columns if c.startswith('seg_')]

    idx = max(0, min(len(df)-1, start_index))
    selected = None  # 'org' | 'exu' | 'seg'
    backups = {}
    show_help = True
    show_arrows = True
    arrow_lw_cycle = [3.0, 4.0, 5.0, 6.0]  # thicker arrows
    arrow_lw_idx = 1  # start at 4.0


    # Figure with two axes: image (top) + legend (bottom)
    fig = plt.figure()
    # Fullscreen
    go_fullscreen(fig)

    # Axes: allocate a clean band at the bottom for legend
    ax_img  = fig.add_axes([0, LEGEND_H, 1, 1-LEGEND_H])  # left, bottom, width, height
    ax_help = fig.add_axes([0, 0, 1, LEGEND_H])
    ax_help.set_axis_off()

    try:
        fig.canvas.manager.set_window_title("validate_annotation.py")
    except Exception:
        pass

    def render():
        ax_img.clear()
        ax_help.clear(); ax_help.set_axis_off()

        row = df.iloc[idx]
        stage = str(row.get('stage'))
        obs   = row.get('observation_id')
        pho   = row.get('photo_id')

        img_path = resolve_image_path(image_root, stage, obs, pho)
        if img_path is None:
            ax_img.text(0.5, 0.5, f"Image not found:\n{stage}/{obs}_{pho}.jpg",
                        ha='center', va='center', transform=ax_img.transAxes)
            ax_img.set_axis_off()
            fig.canvas.draw_idle()
            return

        img = mpimg.imread(img_path)
        ax_img.imshow(img)
        ax_img.set_axis_off()

        # Boxes
        draw_box(ax_img, row, ORG_COLS, edgecolor='blue', selected=(selected=='org'))
        draw_box(ax_img, row, EXU_COLS, edgecolor='red',  selected=(selected=='exu'))

        # Keypoints + arrows
        kp_dict = collect_keypoints(row)
        arrows = 0
        if kp_dict and show_arrows:
            arrows = draw_direction_arrows(
                ax_img, kp_dict,
                lw=arrow_lw_cycle[arrow_lw_idx],
                selected=(selected == 'seg')
            )
        if kp_dict:
            draw_keypoints(ax_img, kp_dict, label=True)

        # Diagnostics (put above the legend band, bottom-right of image panel)
        ax_img.text(
            0.99, 0.01 + (0.02/ max(1e-6, (1-LEGEND_H))),  # safe offset above border
            f"kps: {len(kp_dict)} | arrows: {arrows} | seg_cols: {len(seg_wide_cols)}",
            transform=ax_img.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        # Legend (in its own axis — no overlap with image)
        if show_help:
            ax_help.text(
                0.5, 0.1, help_text(),
                fontsize=10, va='bottom', ha='center',
                linespacing=1.25,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.35'),
                transform=ax_help.transAxes,
            )

        # Title
        ax_img.set_title(
            f"[{idx+1}/{len(df)}] {stage}/{obs}_{pho}.jpg — selected: {selected or '-'} — arrows: {'on' if show_arrows else 'off'}",
            fontsize=11
        )

        fig.canvas.draw_idle()

    def save():
        df.to_csv(csv_path, index=False)
        print(f"[saved] {csv_path}")

    def backup_current():
        to_backup = ORG_COLS + EXU_COLS + ['segments'] + seg_wide_cols
        # backup wide keypoints too (so restoring also reverts arrows' endpoints)
        for _, (cx, cy) in WIDE_KP_PAIRS.items():
            if cx in df.columns and cy in df.columns:
                to_backup.extend([cx, cy])
        backups[idx] = df.loc[df.index[idx], to_backup].copy()

    def restore_current():
        if idx in backups:
            df.loc[df.index[idx], backups[idx].index] = backups[idx]
            print(f"Restored row {idx}.")
            save()
        else:
            print("No backup for this row. Press [1], [2], or [3] before edits to create one.")

    def _clear_wide_kps_at(idx_row):
        """Optional: clear wide head/thorax keypoints to -1 when clearing segments (so arrows vanish)."""
        for _, (cx, cy) in WIDE_KP_PAIRS.items():
            if cx in df.columns:
                df.at[df.index[idx_row], cx] = -1
            if cy in df.columns:
                df.at[df.index[idx_row], cy] = -1

    def clear_segments_at(idx_row):
        """
        Delete segments for the given row:
          - Set 'segments' to "[]"
          - Set all seg_* wide cols to 0/False
          - Also clear wide keypoints (head/thorax org/exu) to -1 (if present)
        """
        df.at[df.index[idx_row], 'segments'] = "[]"
        for c in seg_wide_cols:
            df.at[df.index[idx_row], c] = 0
        _clear_wide_kps_at(idx_row)

    def delete_selected():
        if selected == 'org':
            for c in ORG_COLS:
                df.at[df.index[idx], c] = -1
            print(f"Deleted ORGANISM box at row {idx}")
            save()
        elif selected == 'exu':
            for c in EXU_COLS:
                df.at[df.index[idx], c] = -1
            print(f"Deleted EXUVIAE box at row {idx}")
            save()
        elif selected == 'seg':
            clear_segments_at(idx)
            print(f"Deleted SEGMENTS (arrows endpoints) at row {idx}")
            save()
        else:
            print("No selection. Use [1]=ORG, [2]=EXU, or [3]=SEGMENTS.")

    def delete_all():
        for c in ORG_COLS + EXU_COLS:
            df.at[df.index[idx], c] = -1
        clear_segments_at(idx)
        print(f"Deleted ALL (boxes + segments) at row {idx}")
        save()

    def on_key(event):
        nonlocal idx, selected, show_help, show_arrows, arrow_lw_idx
        k = event.key
        if k in ('q', 'escape'):
            save(); plt.close(fig); return
        if k == 'h':
            show_help = not show_help
        elif k == 'n':
            idx = min(len(df)-1, idx+1)
        elif k == 'b':
            idx = max(0, idx-1)
        elif k == '1':
            selected = 'org'; backup_current()
        elif k == '2':
            selected = 'exu'; backup_current()
        elif k == '3':
            selected = 'seg'; backup_current()
        elif k == 'd':
            delete_selected()
        elif k == 'D':
            backup_current(); delete_all()
        elif k == 'r':
            restore_current()
        elif k == 's':
            save()
        elif k == 'k':
            show_arrows = not show_arrows
        elif k == 'K':
            arrow_lw_idx = (arrow_lw_idx + 1) % len(arrow_lw_cycle)
        elif k in ('f', 'F'):
            go_fullscreen(fig)
        render()

    fig.canvas.mpl_connect('key_press_event', on_key)
    render()
    plt.show()

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Validate/delete bbox annotations and thorax->head arrows (wide or kp_* schema).")
    ap.add_argument("--csv", default=DEFAULT_CSV, type=Path, help="Path to CSV dataset")
    ap.add_argument("--images", default=DEFAULT_IMAGES, type=Path, help="Root folder for images")
    ap.add_argument("--start", default=0, type=int, help="Start index (0-based)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.csv, args.images, args.start)
