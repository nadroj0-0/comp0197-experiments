import os
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


WIDTH    = 2800
HEIGHT   = 1600
M_LEFT   = 110
M_RIGHT  = 60
M_TOP    = 80
M_BOTTOM = 100

BG      = (245, 247, 250)
GRID_C  = (210, 215, 220)
AXIS_C  = (50,  50,  50)
TEXT_C  = (30,  30,  30)

B_TRAIN = (40,  110, 255)   # baseline train  — bright blue
B_VAL   = (30,  30,   30)   # baseline val    — black
R_TRAIN = (220, 60,  50)    # reg train       — bright red
R_VAL   = (30,  160,  60)   # reg val         — green
B_GAP   = (180, 210, 255)   # baseline gap fill
R_GAP   = (255, 185, 175)   # reg gap fill


def nice_ticks(min_v: float, max_v: float, n: int = 6) -> list:
    """
    Generate clean round tick values for a chart axis.

    Args:
        min_v (float): Minimum data value.
        max_v (float): Maximum data value.
        n     (int):   Approximate number of ticks desired.

    Returns:
        list[float]: Rounded tick values spanning [min_v, max_v].
    """
    span = max_v - min_v
    step = span / (n - 1)
    mag  = 10 ** math.floor(math.log10(step))
    step = round(step / mag) * mag
    ticks, v = [], math.floor(min_v / step) * step
    while v <= max_v + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


def make_scalers(epochs, min_acc, max_acc):
    """
    Return (sx, sy) callables that map data coords to pixel coords.

    Args:
        epochs  (list[int]):  Epoch list (used for x range).
        min_acc (float):      Minimum accuracy for y axis.
        max_acc (float):      Maximum accuracy for y axis.

    Returns:
        tuple:
            sx (callable): Maps epoch index (int) → x pixel (float).
            sy (callable): Maps accuracy value (float) → y pixel (float).
    """
    n      = len(epochs)
    plot_w = (WIDTH - M_LEFT - M_RIGHT) * 2
    plot_h = (HEIGHT - M_TOP - M_BOTTOM) * 2

    def sx(i):
        return M_LEFT * 2 + (i / (n - 1)) * plot_w

    def sy(v):
        return M_TOP * 2 + plot_h - ((v - min_acc) / (max_acc - min_acc)) * plot_h

    return sx, sy


def draw_axes_and_grid(draw, sx, sy, epochs, ticks, font):
    """
    Draw grid lines, tick labels, and x/y axes onto a PIL ImageDraw canvas.

    Args:
        draw   (ImageDraw.Draw): Active draw context.
        sx     (callable):       x scaler from make_scalers().
        sy     (callable):       y scaler from make_scalers().
        epochs (list[int]):      Epoch numbers for x-axis labels.
        ticks  (list[float]):    Y-axis tick values from nice_ticks().
    """
    # horizontal grid + y tick labels
    for t in ticks:
        y = sy(t)
        draw.line([(M_LEFT*2, y), (WIDTH*2 - M_RIGHT*2, y)], fill=GRID_C, width=2)
        draw.text((M_LEFT*2 - 140, y - 9), f"{t:.2f}", fill=TEXT_C, font=font)

    # vertical grid + x tick labels
    n = len(epochs)
    step = max(1, n // 10)
    for i in range(0, n, step):
        x = sx(i)
        draw.line([(x, M_TOP*2), (x, HEIGHT*2 - M_BOTTOM*2)], fill=GRID_C, width=2)
        draw.text((x - 8, HEIGHT*2 - M_BOTTOM*2 + 8), str(epochs[i]), fill=TEXT_C, font=font)

    # axes
    draw.line([(M_LEFT*2, M_TOP*2), (M_LEFT*2, HEIGHT*2 - M_BOTTOM*2)],            fill=AXIS_C, width=6)
    draw.line([(M_LEFT*2, HEIGHT*2 - M_BOTTOM*2), (WIDTH*2 - M_RIGHT*2, HEIGHT*2 - M_BOTTOM*2)], fill=AXIS_C, width=6)


def draw_gap_fill(draw, sx, sy, train_acc, val_acc, color):
    """
    Shade the generalisation gap between train and val accuracy curves.

    Args:
        draw      (ImageDraw.Draw): Active draw context.
        sx        (callable):       x scaler.
        sy        (callable):       y scaler.
        train_acc (list[float]):    Training accuracy per epoch.
        val_acc   (list[float]):    Validation accuracy per epoch.
        color     (tuple):          RGB fill colour for the gap polygon.
    """
    for i in range(len(train_acc) - 1):
        poly = [
            (sx(i),   sy(train_acc[i])),
            (sx(i+1), sy(train_acc[i+1])),
            (sx(i+1), sy(val_acc[i+1])),
            (sx(i),   sy(val_acc[i])),
        ]
        draw.polygon(poly, fill=color)


def draw_line_curve(draw, sx, sy, data, color, width=3):
    """
    Draw a connected line curve from a list of y-values.

    Args:
        draw  (ImageDraw.Draw): Active draw context.
        sx    (callable):       x scaler.
        sy    (callable):       y scaler.
        data  (list[float]):    Y values (one per epoch).
        color (tuple):          RGB line colour.
        width (int):            Line width in pixels.
    """
    pts = [(sx(i), sy(v)) for i, v in enumerate(data)]
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i+1]], fill=color, width=width)

def draw_dashed_curve(draw, sx, sy, data, color, width=3, dash=8, gap=5):
    pts = [(sx(i), sy(v)) for i, v in enumerate(data)]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i+1]
        length = math.hypot(x1-x0, y1-y0)
        steps = int(length / (dash + gap))
        for s in range(steps):
            t0 = s * (dash + gap) / length
            t1 = min((s * (dash + gap) + dash) / length, 1.0)
            draw.line([(x0 + t0*(x1-x0), y0 + t0*(y1-y0)),
                       (x0 + t1*(x1-x0), y0 + t1*(y1-y0))], fill=color, width=width)


def draw_legend(draw, entries, x, y, font):
    """
    Draw a simple box legend with colour swatches and labels.

    Args:
        draw    (ImageDraw.Draw):          Active draw context.
        entries (list[tuple[tuple,str]]):  List of (RGB color, label string) pairs.
        x       (int):                     Left x of legend box.
        y       (int):                     Top y of legend box.
    """
    pad    = 28
    row_h  = 56
    box_h  = pad * 2 + row_h * len(entries)
    box_w  = 660

    draw.rectangle([x - pad, y - pad, x + box_w, y + box_h], fill=(255,255,255), outline=(180,180,180), width=2)

    for i, (color, label) in enumerate(entries):
        ry = y + i * row_h
        draw.line([(x, ry + 6), (x + 70, ry + 12)], fill=color, width=8)
        draw.text((x + 90, ry), label, fill=TEXT_C, font=font)


def generate_gap_plot(
    b_epochs, b_train_acc, b_val_acc,
    r_epochs, r_train_acc, r_val_acc,
    save_path: Path
):
    """
    Generate and save the generalisation_gap.png plot showing train vs validation
    accuracy for both baseline and regularised models with gap shading.

    Args:
        b_epochs    (list[int]):   Baseline epoch numbers.
        b_train_acc (list[float]): Baseline training accuracy per epoch.
        b_val_acc   (list[float]): Baseline validation accuracy per epoch.
        r_epochs    (list[int]):   Regularised epoch numbers.
        r_train_acc (list[float]): Regularised training accuracy per epoch.
        r_val_acc   (list[float]): Regularised validation accuracy per epoch.
        save_path   (Path):        Output file path for the PNG.
    """
    epochs  = b_epochs  # both models trained for same number of epochs
    all_acc = b_train_acc + b_val_acc + r_train_acc + r_val_acc
    min_acc = min(all_acc)
    max_acc = max(all_acc)

    pad     = (max_acc - min_acc) * 0.12
    min_acc -= pad
    max_acc = min(1.0, max_acc + pad)

    font_title = ImageFont.load_default(size=70)
    font = ImageFont.load_default(size=60)
    font_med = ImageFont.load_default(size=55)
    font_small = ImageFont.load_default(size=45)

    sx, sy = make_scalers(epochs, min_acc, max_acc)
    ticks  = nice_ticks(min_acc, max_acc)

    img  = Image.new("RGB", (WIDTH * 2, HEIGHT * 2), BG)
    draw = ImageDraw.Draw(img)

    draw_axes_and_grid(draw, sx, sy, epochs, ticks, font=font_med)

    # gap fills (draw before curves so lines sit on top)
    draw_gap_fill(draw, sx, sy, b_train_acc, b_val_acc, B_GAP)
    draw_gap_fill(draw, sx, sy, r_train_acc, r_val_acc, R_GAP)

    # curves
    draw_line_curve(draw, sx, sy, b_train_acc, B_TRAIN, width=6)
    draw_line_curve(draw, sx, sy, b_val_acc, B_VAL, width=6)
    draw_line_curve(draw, sx, sy, r_train_acc, R_TRAIN, width=6)
    draw_line_curve(draw, sx, sy, r_val_acc, R_VAL, width=6)

    # legend
    entries = [
        (B_TRAIN, "Baseline Train"),
        (B_VAL,   "Baseline Validation"),
        (R_TRAIN, "Regularised Train"),
        (R_VAL,   "Regularised Validation"),
    ]
    draw_legend(draw, entries, M_LEFT*2 + 50, M_TOP*2 + 20, font=font_med)

    # labels
    draw.text((WIDTH*2 // 2 - 260, 22), "Generalisation Gap: Baseline vs Regularised CNN", fill=TEXT_C, font=font_title)
    draw.text((WIDTH*2 // 2 - 25,  HEIGHT*2 - M_BOTTOM*2 +80), "Epoch",    fill=TEXT_C, font=font)
    draw.text((12, HEIGHT*2 // 2 - 10),                      "Accuracy", fill=TEXT_C, font=font_small)

    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    img.save(save_path)
    print(f"Plot saved to: {save_path}")

def generate_gap_per_epoch_plot(
    b_epochs, b_train_acc, b_val_acc,
    r_epochs, r_train_acc, r_val_acc,
    save_path: Path
):
    """
    Plot the generalisation gap (train - val accuracy) per epoch for
    both models, using each model's own epoch list so they never misalign.

    Args:
        b_epochs    (list[int]):   Baseline epoch numbers.
        b_train_acc (list[float]): Baseline training accuracy per epoch.
        b_val_acc   (list[float]): Baseline validation accuracy per epoch.
        r_epochs    (list[int]):   Regularised epoch numbers.
        r_train_acc (list[float]): Regularised training accuracy per epoch.
        r_val_acc   (list[float]): Regularised validation accuracy per epoch.
        save_path   (Path):        Output file path for the PNG.
    """
    b_gap = [t - v for t, v in zip(b_train_acc, b_val_acc)]
    r_gap = [t - v for t, v in zip(r_train_acc, r_val_acc)]

    all_gap = b_gap + r_gap
    min_gap = min(all_gap)
    max_gap = max(all_gap)
    pad     = (max_gap - min_gap) * 0.12
    min_gap -= pad
    max_gap += pad

    # use the longer epoch list for the x-axis
    epochs = b_epochs if len(b_epochs) >= len(r_epochs) else r_epochs

    font_title = ImageFont.load_default(size=70)
    font_med   = ImageFont.load_default(size=55)
    font_small = ImageFont.load_default(size=45)
    font       = ImageFont.load_default(size=60)

    sx, sy = make_scalers(epochs, min_gap, max_gap)
    ticks  = nice_ticks(min_gap, max_gap)

    img  = Image.new("RGB", (WIDTH * 2, HEIGHT * 2), BG)
    draw = ImageDraw.Draw(img)

    draw_axes_and_grid(draw, sx, sy, epochs, ticks, font=font_med)

    # each model uses its own epoch list so x positions are correct
    b_sx, b_sy = make_scalers(b_epochs, min_gap, max_gap)
    r_sx, r_sy = make_scalers(r_epochs, min_gap, max_gap)
    draw_line_curve(draw, b_sx, b_sy, b_gap, B_TRAIN, width=6)
    draw_line_curve(draw, r_sx, r_sy, r_gap, R_TRAIN, width=6)

    entries = [
        (B_TRAIN, "Baseline Gap"),
        (R_TRAIN, "Regularised Gap"),
    ]
    draw_legend(draw, entries, M_LEFT*2 + 50, M_TOP*2 + 20, font=font_med)

    draw.text((WIDTH*2 // 2 - 260, 22),              "Generalisation Gap per Epoch", fill=TEXT_C, font=font_title)
    draw.text((WIDTH*2 // 2 - 25,  HEIGHT*2 - M_BOTTOM*2 + 80), "Epoch",           fill=TEXT_C, font=font)
    draw.text((12, HEIGHT*2 // 2 - 10),              "Train - Val Accuracy",        fill=TEXT_C, font=font_small)

    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    img.save(save_path)
    print(f"Gap-per-epoch plot saved to: {save_path}")