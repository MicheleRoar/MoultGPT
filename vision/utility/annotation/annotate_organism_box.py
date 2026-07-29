import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

# === CONFIGURATION ===
CSV_PATH = "../../data/inat_raw/inat_dataset.csv"  # CSV ridotto
IMAGE_ROOT = "../../data/inat_raw"                         # cartelle per stage
TARGET_STAGES = {"moulting", "post-moult"}                 # escludi/aggiungi come vuoi

# === LOAD METADATA ===
df = pd.read_csv(CSV_PATH)

# Assicurati che le colonne box esistano
for col in ["x_exuviae","y_exuviae","w_exuviae","h_exuviae",
            "x_organism","y_organism","w_organism","h_organism"]:
    if col not in df.columns:
        df[col] = pd.NA

# === FILTER: solo immagini da annotare (organism mancante) ===
mask = (
    df["stage"].isin(TARGET_STAGES) &
    df[["x_organism","y_organism","w_organism","h_organism"]].isna().any(axis=1)
)
df_indices = df[mask].index.tolist()
print(f"Ready to annotate {len(df_indices)} organism boxes.")

i = 0
while i < len(df_indices):
    idx = df_indices[i]
    row = df.loc[idx]

    stage    = row["stage"]
    obs_id   = row["observation_id"]
    photo_id = row["photo_id"]
    fname    = f"{obs_id}_{photo_id}.jpg"
    img_path = os.path.join(IMAGE_ROOT, stage, fname)

    if not os.path.exists(img_path):
        print(f"File not found: {img_path} -> skip")
        i += 1
        continue

    # Carica immagine
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(
        f"[{stage.upper()}] {fname}\n"
        "'n'=skip  'b'=back  'd'=delete  'q'=quit",
        fontsize=10
    )

    # Disegna box exuviae se presente
    if pd.notna(row["x_exuviae"]) and pd.notna(row["y_exuviae"]) and pd.notna(row["w_exuviae"]) and pd.notna(row["h_exuviae"]):
        ax.add_patch(Rectangle(
            (row["x_exuviae"], row["y_exuviae"]),
            row["w_exuviae"], row["h_exuviae"],
            linewidth=1.5, edgecolor="r", facecolor="none"
        ))

    # Se c'è già una box organism (caso revisione), mostrala
    if pd.notna(row["x_organism"]) and pd.notna(row["y_organism"]) and pd.notna(row["w_organism"]) and pd.notna(row["h_organism"]):
        ax.add_patch(Rectangle(
            (row["x_organism"], row["y_organism"]),
            row["w_organism"], row["h_organism"],
            linewidth=1.5, edgecolor="b", facecolor="none", linestyle="--"
        ))

    # Interazione
    rect_coords = []
    organism_rect = [None]
    flags = {"skip": False, "back": False, "delete": False, "quit": False}

    def on_press(event):
        if event.xdata is None or event.ydata is None:
            return
        rect_coords.clear()
        rect_coords.append((event.xdata, event.ydata))
        organism_rect[0] = Rectangle((event.xdata, event.ydata), 1, 1,
                                     linewidth=1.8, edgecolor="b", facecolor="none")
        ax.add_patch(organism_rect[0]); fig.canvas.draw()

    def on_motion(event):
        if organism_rect[0] is not None and event.xdata is not None and event.ydata is not None:
            x0, y0 = rect_coords[0]; x1, y1 = event.xdata, event.ydata
            organism_rect[0].set_bounds(min(x0,x1), min(y0,y1), abs(x1-x0), abs(y1-y0))
            fig.canvas.draw_idle()

    def on_release(event):
        if not rect_coords or any(flags.values()):
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0 = rect_coords[0]; x1, y1 = event.xdata, event.ydata
        x_min, y_min = int(min(x0,x1)), int(min(y0,y1))
        w, h = int(abs(x1-x0)), int(abs(y1-y0))
        df.at[idx, "x_organism"] = x_min
        df.at[idx, "y_organism"] = y_min
        df.at[idx, "w_organism"] = w
        df.at[idx, "h_organism"] = h
        print(f"{fname} → organism box: ({x_min}, {y_min}, {w}, {h})")
        plt.close()

    def on_key(event):
        if event.key == "n":
            flags["skip"] = True; print(f"⏭️ Skipped: {fname}"); plt.close()
        elif event.key == "b":
            flags["back"] = True; print("↩️ Back one"); plt.close()
        elif event.key == "d":
            flags["delete"] = True
            for c in ["x_organism","y_organism","w_organism","h_organism"]:
                df.at[idx, c] = pd.NA
            print(f"🗑️ Deleted organism box for {fname}")
            plt.close()
        elif event.key == "q":
            flags["quit"] = True; print("Exiting."); plt.close()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if flags["quit"]:
        break
    elif flags["back"]:
        if i > 0: i -= 1
        continue
    else:
        i += 1
        df.to_csv(CSV_PATH, index=False)

print("\nFinished organism annotation.")
print(f"File updated: {CSV_PATH}")
