#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

# ===================== CONFIG =====================
CSV_PATH = "../../data/inat_raw/inat_dataset.csv"
IMAGE_ROOT = "../../data/inat_raw"
TARGET_STAGES = {"moulting"}   # set to None to allow all stages
AUTO_SAVE_EVERY = 1

# Layout
LEGEND_H = 0.14   # fraction reserved for legend (bottom strip)
TOP_PAD  = 0.06   # NEW: top breathing room (fraction of figure height)


# Colors
ORG_BOX_COLOR = "deepskyblue"
EXU_BOX_COLOR = "tomato"
KP_POINT_COLOR = "yellow"
ORG_LINE_COLOR = "lime"   # organism segment
EXU_LINE_COLOR = "gold"   # exuviae segment

# Columns
BBOX_COLS = [
    "x_exuviae","y_exuviae","w_exuviae","h_exuviae",
    "x_organism","y_organism","w_organism","h_organism",
]
KP_COLS = [
    "x_head_org","y_head_org","x_thorax_org","y_thorax_org",
    "x_head_exu","y_head_exu","x_thorax_exu","y_thorax_exu",
]
EXTRA_COLS = ["exu_ref"]  # set "missing_head" if H_exu is skipped

CLICK_LABELS = ["H_org", "T_org", "H_exu", "T_exu"]  # expected order

# ===================== HELPERS =====================
def ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

def resolve_image_path(stage: str, obs_id, photo_id):
    """Build <IMAGE_ROOT>/<stage>/<observation_id>_<photo_id>.jpg"""
    try:
        fname = f"{int(obs_id)}_{int(photo_id)}.jpg"
    except Exception:
        return None
    p = os.path.join(IMAGE_ROOT, str(stage), fname)
    return p if os.path.exists(p) else None

def valid_coord(x, y) -> bool:
    return pd.notna(x) and pd.notna(y) and int(x) != -1 and int(y) != -1

def plot_kp(ax, x, y, label, color=KP_POINT_COLOR):
    ax.plot([x],[y], marker='o', color=color, markersize=5, zorder=3)
    ax.text(int(x)+4, int(y)-4, label, color=color, fontsize=10,
            bbox=dict(facecolor='black', alpha=0.25, edgecolor='none', pad=1.5), zorder=4)

def draw_bbox_if(ax, row, x, y, w, h, color, label):
    if all(pd.notna(row[c]) for c in (x,y,w,h)):
        ax.add_patch(Rectangle((row[x], row[y]), row[w], row[h],
                               linewidth=1.8, edgecolor=color, facecolor='none', zorder=2))
        ax.text(int(row[x]), max(0, int(row[y])-7), label, color=color, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.0), zorder=3)

def draw_existing_segments(ax, row):
    """Draw existing head<->thorax segments (exactly like your original)."""
    if valid_coord(row["x_head_org"], row["y_head_org"]) and valid_coord(row["x_thorax_org"], row["y_thorax_org"]):
        ax.plot([row["x_head_org"], row["x_thorax_org"]],
                [row["y_head_org"], row["y_thorax_org"]],
                color=ORG_LINE_COLOR, linewidth=2, zorder=1)
    if valid_coord(row["x_head_exu"], row["y_head_exu"]) and valid_coord(row["x_thorax_exu"], row["y_thorax_exu"]):
        ax.plot([row["x_head_exu"], row["x_thorax_exu"]],
                [row["y_head_exu"], row["y_thorax_exu"]],
                color=EXU_LINE_COLOR, linewidth=2, zorder=1)

def label_to_cols(label: str):
    if label == "H_org": return ("x_head_org","y_head_org")
    if label == "T_org": return ("x_thorax_org","y_thorax_org")
    if label == "H_exu": return ("x_head_exu","y_head_exu")
    if label == "T_exu": return ("x_thorax_exu","y_thorax_exu")
    raise ValueError(label)

def set_if_empty(df: pd.DataFrame, idx: int, col: str, val: int):
    if pd.isna(df.at[idx, col]):
        df.at[idx, col] = int(val)

def legend_text(fname: str) -> str:
    return (
        f"{fname}\n"
        "Click order: H_org → T_org → (opt) H_exu → (opt) T_exu\n"
        "Keys: x=skip current label • r=reset image • n=next/skip image • b=back(clear prev) • q=quit"
    )

# ===================== MAIN =====================
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Ensure required columns exist
    ensure_cols(df, BBOX_COLS)
    ensure_cols(df, KP_COLS)
    ensure_cols(df, EXTRA_COLS)
    for c in ["stage", "observation_id", "photo_id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in CSV.")

    # Filter rows with existing files (and stage filter)
    def row_is_candidate(r):
        stage = r.get("stage")
        if TARGET_STAGES is not None and stage not in TARGET_STAGES:
            return False
        return resolve_image_path(stage, r.get("observation_id"), r.get("photo_id")) is not None

    mask = df.apply(row_is_candidate, axis=1)
    work = df[mask].copy().reset_index()  # keep original df index in 'index'
    if work.empty:
        print("No valid images found (stage filter + file existence).")
        return

    indices = list(work.index)
    print(f"Images to annotate: {len(indices)}")

    # ---------- Create ONE persistent window (no flicker/resize) ----------
    fig = plt.figure(figsize=(14, 9), dpi=100)  # large, stable size
    # one big image area + one legend strip
    fig = plt.figure(figsize=(14, 9), dpi=100)

    # Top spacer axis (empty, just to create visual breathing)
    ax_top  = fig.add_axes([0, 1 - TOP_PAD, 1, TOP_PAD])
    ax_top.set_axis_off()

    # Image axis shrinks accordingly (between legend bottom strip and top spacer)
    ax_img  = fig.add_axes([0, LEGEND_H, 1, 1 - LEGEND_H - TOP_PAD])

    # Bottom legend axis (no overlap with image)
    ax_help = fig.add_axes([0, 0, 1, LEGEND_H])
    ax_help.set_axis_off()


    try:
        fig.canvas.manager.set_window_title("Annotate keypoints")
    except Exception:
        pass

    # show once, non-blocking
    plt.show(block=False)

    i = 0
    annotated = 0

    # Keep references for event connections to disconnect between images
    cid_click = None
    cid_key = None

    while i < len(indices):
        ridx = indices[i]
        row = work.loc[ridx]
        df_idx = int(row["index"])

        stage = str(row["stage"])
        obs_id = row["observation_id"]
        photo_id = row["photo_id"]

        img_path = resolve_image_path(stage, obs_id, photo_id)
        if img_path is None:
            print(f"⚠️ File not found: {stage}/{obs_id}_{photo_id}.jpg")
            i += 1
            continue

        fname = os.path.basename(img_path)
        img = mpimg.imread(img_path)

        # ---- render current image/overlays ----
        ax_img.clear(); ax_img.imshow(img); ax_img.set_axis_off()
        draw_bbox_if(ax_img, row, "x_organism","y_organism","w_organism","h_organism",
                     ORG_BOX_COLOR, "organism")
        draw_bbox_if(ax_img, row, "x_exuviae","y_exuviae","w_exuviae","h_exuviae",
                     EXU_BOX_COLOR, "exuvia")

        existing = {
            "H_org": ("x_head_org","y_head_org"),
            "T_org": ("x_thorax_org","y_thorax_org"),
            "H_exu": ("x_head_exu","y_head_exu"),
            "T_exu": ("x_thorax_exu","y_thorax_exu"),
        }
        for name, (cx, cy) in existing.items():
            xv, yv = row[cx], row[cy]
            if valid_coord(xv, yv):
                plot_kp(ax_img, xv, yv, name)

        draw_existing_segments(ax_img, row)

        ax_img.set_title(f"[{stage.upper()}] {fname}", fontsize=12)

        # legend (no overlap)
        ax_help.clear(); ax_help.set_axis_off()
        ax_help.text(
            0.5, 0.15, legend_text(fname),
            fontsize=12, va='bottom', ha='center',
            linespacing=1.25,
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', boxstyle='round,pad=0.45'),
            transform=ax_help.transAxes,
        )
        ax_help.text(
            0.5, 0.86, "Segments: ORGANISM = lime • EXUVIAE = gold   |   Points = yellow",
            fontsize=11, va='center', ha='center', transform=ax_help.transAxes
        )

        fig.canvas.draw_idle()

        # ---- interaction state for this image ----
        items = []  # up to 4 entries: (x,y) or ('SKIP','SKIP')
        flags = {"skip": False, "back": False, "quit": False}

        def maybe_draw_segments():
            if len(items) >= 2 and items[0] != ('SKIP','SKIP') and items[1] != ('SKIP','SKIP'):
                (hx, hy), (tx, ty) = items[0], items[1]
                ax_img.plot([hx, tx], [hy, ty], color=ORG_LINE_COLOR, linewidth=2)
            if len(items) >= 4 and items[2] != ('SKIP','SKIP') and items[3] != ('SKIP','SKIP'):
                (hx, hy), (tx, ty) = items[2], items[3]
                ax_img.plot([hx, tx], [hy, ty], color=EXU_LINE_COLOR, linewidth=2)

        def redraw_full():
            ax_img.clear(); ax_img.imshow(img); ax_img.set_axis_off()
            draw_bbox_if(ax_img, row, "x_organism","y_organism","w_organism","h_organism",
                         ORG_BOX_COLOR, "organism")
            draw_bbox_if(ax_img, row, "x_exuviae","y_exuviae","w_exuviae","h_exuviae",
                         EXU_BOX_COLOR, "exuvia")
            for name,(cx,cy) in existing.items():
                xv,yv = row[cx], row[cy]
                if valid_coord(xv, yv):
                    plot_kp(ax_img, xv, yv, name)
            draw_existing_segments(ax_img, row)
            # redraw current clicks
            for j, pt in enumerate(items):
                if pt == ('SKIP','SKIP'):
                    continue
                lbl = CLICK_LABELS[j]
                plot_kp(ax_img, pt[0], pt[1], lbl)
            maybe_draw_segments()
            fig.canvas.draw_idle()

        def on_click(ev):
            if ev.inaxes is not ax_img or ev.xdata is None or ev.ydata is None:
                return
            if len(items) >= 4:
                return
            x, y = int(ev.xdata), int(ev.ydata)
            label = CLICK_LABELS[len(items)]
            items.append((x, y))
            plot_kp(ax_img, x, y, label)
            maybe_draw_segments()
            if len(items) == 4:
                # we'll exit the wait loop below, no window close
                pass
            fig.canvas.draw_idle()

        def on_key(ev):
            k = ev.key
            if k == 'r':
                items.clear()
                redraw_full()
            elif k == 'x':
                if len(items) < 4:
                    label = CLICK_LABELS[len(items)]
                    items.append(('SKIP','SKIP'))
                    ax_img.text(8, 18 + 18*len(items), f"Skipped: {label}",
                                color='white', fontsize=10,
                                bbox=dict(facecolor='black', alpha=0.3, edgecolor='none'))
                    maybe_draw_segments()
                    fig.canvas.draw_idle()
            elif k == 'n':
                flags["skip"] = True
            elif k == 'b':
                flags["back"] = True
            elif k == 'q':
                flags["quit"] = True

        # (Re)connect handlers for this image only
        if cid_click is not None:
            fig.canvas.mpl_disconnect(cid_click)
        if cid_key is not None:
            fig.canvas.mpl_disconnect(cid_key)
        cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
        cid_key   = fig.canvas.mpl_connect('key_press_event', on_key)

        # ---- modal wait loop without closing/resizing the window ----
        # stop waiting when: 4 inputs given, or user pressed n/b/q
        while True:
            plt.pause(0.05)
            if flags["quit"] or flags["back"] or flags["skip"] or len(items) == 4:
                break

        # ------ actions after interaction ------
        if flags["quit"]:
            break
        if flags["back"]:
            if i > 0:
                prev_ridx = indices[i-1]
                prev_df_idx = int(work.loc[prev_ridx, "index"])
                for c in KP_COLS:
                    df.at[prev_df_idx, c] = pd.NA
                print("↩️ Back one image. Previous keypoints cleared.")
                i -= 1
            continue
        if flags["skip"]:
            print(f"⏭️ Skipped image: {fname}")
            i += 1
            continue

        # Persist if we have 4 assignments (clicks or 'x' skips)
        if len(items) == 4:
            for j, label in enumerate(CLICK_LABELS):
                cx, cy = label_to_cols(label)
                val = items[j]
                if val == ('SKIP','SKIP'):
                    df.at[df_idx, cx] = -1
                    df.at[df_idx, cy] = -1
                    if label == "H_exu":
                        df.at[df_idx, "exu_ref"] = "missing_head"
                else:
                    set_if_empty(df, df_idx, cx, val[0])
                    set_if_empty(df, df_idx, cy, val[1])

            annotated += 1
            if annotated % AUTO_SAVE_EVERY == 0:
                df.to_csv(CSV_PATH, index=False)
                print(f"💾 Saved CSV: {CSV_PATH} (progress {annotated})")
            i += 1
            continue

        print("No action taken (need 4 assignments). Re-opening the same image.")

    # done
    df.to_csv(CSV_PATH, index=False)
    print("\n✅ Done. CSV updated (keypoints + exu_ref):", CSV_PATH)

if __name__ == "__main__":
    main()
