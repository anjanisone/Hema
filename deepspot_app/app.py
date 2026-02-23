"""
DeepSpot Analysis - Pollen Viability Detection (Tkinter UI).
Internal testing build: SQLite on network drive, dummy YOLO.
Modern UI: header bar, card panels, cohesive palette.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .config import get_db_path, get_results_dir, use_real_yolo, get_model_path
from . import db
from .inference import run_inference

# Image extensions to scan
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Modern palette
BG_APP = "#f1f5f9"
SURFACE = "#ffffff"
SURFACE_BORDER = "#e2e8f0"
HEADER_BG = "#1e293b"
HEADER_TEXT = "#f8fafc"
ACCENT = "#6366f1"
ACCENT_HOVER = "#4f46e5"
TEXT = "#0f172a"
TEXT_MUTED = "#64748b"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
CARD_BG = "#f8fafc"
INPUT_BG = "#ffffff"
BTN_COLORS = {
    "browse": "#0d9488",
    "run": "#2563eb",
    "batch": "#7c3aed",
    "export": "#d97706",
    "test": "#db2777",
    "history": "#475569",
}
FONT_UI = "Segoe UI"
FONT_SIZE = 10
FONT_SIZE_SM = 9
FONT_SIZE_LG = 12


def _button_text_color():
    """Foreground color for tk.Button so label is always visible. On macOS tk often ignores bg and shows gray."""
    return "#1e293b" if sys.platform == "darwin" else "#ffffff"


class DeepSpotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepSpot Analysis - Pollen Viability Detection")
        self.minsize(1000, 700)
        self.geometry("1200x780")
        self.configure(bg=BG_APP)

        self.folder_path = tk.StringVar(value="")
        self.image_files = []
        self.all_detection_results = {}
        self.current_image_path = None
        self.run_id = None

        db.init_db()
        self._setup_styles()
        self._build_ui()
        if use_real_yolo():
            self._update_status(f"Ready. Model: {get_model_path()} | DB: {get_db_path()}")
        else:
            self._update_status(f"Ready (dummy model). Set DEEPSPOT_MODEL_PATH for real YOLO. DB: {get_db_path()}")

    def _setup_styles(self):
        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        self.style.configure(".", background=BG_APP, foreground=TEXT, font=(FONT_UI, FONT_SIZE))
        self.style.configure("TFrame", background=BG_APP)
        self.style.configure("TLabel", background=BG_APP, foreground=TEXT, font=(FONT_UI, FONT_SIZE))
        self.style.configure("TLabelframe", background=SURFACE, foreground=TEXT)
        self.style.configure("TLabelframe.Label", background=SURFACE, foreground=TEXT, font=(FONT_UI, FONT_SIZE, "bold"))
        self.style.configure("TEntry", fieldbackground=INPUT_BG, padding=6)
        self.style.configure("TCombobox", fieldbackground=INPUT_BG, padding=4)
        self.style.configure("Card.TFrame", background=SURFACE)
        self.style.configure("Header.TLabel", background=HEADER_BG, foreground=HEADER_TEXT, font=(FONT_UI, 14, "bold"))
        self.style.configure("Muted.TLabel", foreground=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM))
        self.style.configure("CardHeader.TLabel", background=CARD_BG, foreground=TEXT, font=(FONT_UI, FONT_SIZE, "bold"))
        self.style.map("TRadiobutton", background=[("active", CARD_BG)])

    def _build_ui(self):
        main = tk.Frame(self, bg=BG_APP, padx=12, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ---- Header bar ----
        header = tk.Frame(main, bg=HEADER_BG, height=56)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        btn_container = tk.Frame(header, bg=HEADER_BG)
        btn_container.pack(side=tk.LEFT, padx=12, pady=10)
        fg_btn = _button_text_color()
        for label, key, cmd in [
            ("Browse folder", "browse", self._on_browse),
            ("Run", "run", self._on_run),
            ("Batch run", "batch", self._on_batch_run),
            ("Export", "export", self._on_export),
            ("Test Data", "test", self._load_test_data),
            ("History", "history", self._on_history),
        ]:
            b = tk.Button(
                btn_container, text=label, command=cmd,
                bg=BTN_COLORS[key], fg=fg_btn, activebackground=BTN_COLORS[key], activeforeground=fg_btn,
                font=(FONT_UI, FONT_SIZE, "bold"), relief=tk.FLAT, bd=0, padx=14, pady=8, cursor="hand2",
                highlightthickness=0,
            )
            b.pack(side=tk.LEFT, padx=3)
        title = tk.Label(header, text="  DeepSpot  ·  Pollen Viability Detection  ", bg=HEADER_BG, fg=HEADER_TEXT, font=(FONT_UI, 14, "bold"))
        title.pack(side=tk.LEFT, padx=20)

        # ---- Content: cards ----
        content = tk.Frame(main, bg=BG_APP)
        content.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        # Left card: Input
        left_outer = tk.Frame(content, bg=SURFACE_BORDER, padx=1, pady=1)
        left_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left = tk.Frame(left_outer, bg=SURFACE, padx=16, pady=14)
        left.pack(fill=tk.BOTH, expand=True)
        left_width = 320
        tk.Label(left, text="Input", bg=SURFACE, fg=TEXT, font=(FONT_UI, FONT_SIZE_LG, "bold")).pack(anchor=tk.W, pady=(0, 10))

        tk.Label(left, text="Model", bg=SURFACE, fg=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM)).pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="DeepSpot (YOLO)")
        model_combo = ttk.Combobox(left, textvariable=self.model_var, width=26, state="readonly")
        model_combo["values"] = ("DeepSpot (YOLO)",)
        model_combo.pack(anchor=tk.W, pady=(2, 12))

        tk.Label(left, text="Crop", bg=SURFACE, fg=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM)).pack(anchor=tk.W)
        crop_frame = tk.Frame(left, bg=SURFACE)
        crop_frame.pack(anchor=tk.W, pady=(2, 12))
        self.crop_var = tk.StringVar(value="Maize")
        for v in ("Maize", "Wheat", "Soybean"):
            ttk.Radiobutton(crop_frame, text=v, variable=self.crop_var, value=v).pack(anchor=tk.W)

        tk.Label(left, text="Objective", bg=SURFACE, fg=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM)).pack(anchor=tk.W)
        self.objective_var = tk.StringVar(value="2")
        obj_combo = ttk.Combobox(left, textvariable=self.objective_var, width=8, state="readonly")
        obj_combo["values"] = ("2", "4", "10", "20")
        obj_combo.pack(anchor=tk.W, pady=(2, 12))

        tk.Label(left, text="Raw images", bg=SURFACE, fg=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM)).pack(anchor=tk.W)
        self.image_combo_var = tk.StringVar()
        self.image_combo = ttk.Combobox(left, textvariable=self.image_combo_var, width=26, state="readonly")
        self.image_combo.pack(anchor=tk.W, pady=2)
        self.image_combo.bind("<<ComboboxSelected>>", lambda e: self._on_select_image())
        img_preview_frame = tk.Frame(left, bg=SURFACE)
        img_preview_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.preview_canvas = tk.Canvas(img_preview_frame, width=left_width - 32, height=200, bg=CARD_BG, highlightthickness=0)
        self.preview_canvas.pack()
        self.preview_canvas.create_text((left_width - 32) // 2, 100, text="No image selected.\nBrowse folder to load images.", fill=TEXT_MUTED, font=(FONT_UI, FONT_SIZE), justify=tk.CENTER)

        # Right: Results card
        right_outer = tk.Frame(content, bg=SURFACE_BORDER, padx=1, pady=1)
        right_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(right_outer, bg=SURFACE, padx=16, pady=14)
        right.pack(fill=tk.BOTH, expand=True)

        # Summary cards (Count, Total, Viability)
        sum_frame = tk.Frame(right, bg=SURFACE)
        sum_frame.pack(fill=tk.X, pady=(0, 12))
        self.count_var = tk.StringVar(value="—")
        self.total_var = tk.StringVar(value="—")
        self.viability_var = tk.StringVar(value="—")
        for (lbl, var) in [("Count in Image", self.count_var), ("Total Count", self.total_var), ("Viability (%)", self.viability_var)]:
            card = tk.Frame(sum_frame, bg=CARD_BG, padx=14, pady=10)
            card.pack(side=tk.LEFT, padx=(0, 10))
            tk.Label(card, text=lbl, bg=CARD_BG, fg=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM)).pack(anchor=tk.W)
            e = ttk.Entry(card, textvariable=var, width=10)
            e.pack(anchor=tk.W, pady=(4, 0))

        # Metrics table (styled)
        tk.Label(right, text="Metrics", bg=SURFACE, fg=TEXT, font=(FONT_UI, FONT_SIZE_LG, "bold")).pack(anchor=tk.W, pady=(4, 6))
        metrics_outer = tk.Frame(right, bg=SURFACE_BORDER)
        metrics_outer.pack(fill=tk.X, pady=(0, 12))
        metrics_inner = tk.Frame(metrics_outer, bg=SURFACE)
        metrics_inner.pack(fill=tk.X, padx=1, pady=1)
        metrics_frame = tk.Frame(metrics_inner, bg=SURFACE, padx=8, pady=8)
        metrics_frame.pack(fill=tk.X)
        # Table header
        hrow = tk.Frame(metrics_frame, bg=HEADER_BG)
        hrow.pack(fill=tk.X, pady=(0, 2))
        tk.Label(hrow, text="", bg=HEADER_BG, fg=HEADER_TEXT, font=(FONT_UI, FONT_SIZE_SM), width=24, anchor=tk.W, padx=6, pady=6).pack(side=tk.LEFT)
        tk.Label(hrow, text="Viable", bg=HEADER_BG, fg=HEADER_TEXT, font=(FONT_UI, FONT_SIZE_SM, "bold"), width=10, anchor=tk.CENTER, padx=6, pady=6).pack(side=tk.LEFT)
        tk.Label(hrow, text="All", bg=HEADER_BG, fg=HEADER_TEXT, font=(FONT_UI, FONT_SIZE_SM, "bold"), width=10, anchor=tk.CENTER, padx=6, pady=6).pack(side=tk.LEFT)
        rows = [
            ("Avg diameter (µm)", "avg_diameter_um"),
            ("Avg area (µm²)", "avg_area_um2"),
            ("Avg perimeter (µm)", "avg_perimeter_um"),
            ("Avg perimeter / √area", "avg_perimeter_sqrt_area"),
            ("Avg circularity", "avg_circularity"),
            ("Avg aspect ratio", "avg_aspect_ratio"),
        ]
        self.metric_vars = {}
        for i, (label, key) in enumerate(rows):
            row_bg = CARD_BG if i % 2 == 1 else SURFACE
            rframe = tk.Frame(metrics_frame, bg=row_bg)
            rframe.pack(fill=tk.X, pady=1)
            tk.Label(rframe, text=label, bg=row_bg, fg=TEXT, font=(FONT_UI, FONT_SIZE_SM), width=24, anchor=tk.W, padx=6, pady=4).pack(side=tk.LEFT)
            v_viable = tk.StringVar(value="—")
            v_all = tk.StringVar(value="—")
            e1 = ttk.Entry(rframe, textvariable=v_viable, width=10)
            e1.pack(side=tk.LEFT, padx=6, pady=4)
            e2 = ttk.Entry(rframe, textvariable=v_all, width=10)
            e2.pack(side=tk.LEFT, padx=6, pady=4)
            self.metric_vars[key] = (v_viable, v_all)

        # Image panels
        img_result_frame = tk.Frame(right, bg=SURFACE)
        img_result_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        for (title_text, attr) in [
            ("Segmented image", "seg_canvas"),
            ("Annotated image (Green: Viable, Red: Unviable)", "ann_canvas"),
        ]:
            pan_outer = tk.Frame(img_result_frame, bg=SURFACE_BORDER, padx=1, pady=1)
            pan_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
            pan_inner = tk.Frame(pan_outer, bg=SURFACE, padx=8, pady=8)
            pan_inner.pack(fill=tk.BOTH, expand=True)
            tk.Label(pan_inner, text=title_text, bg=SURFACE, fg=TEXT, font=(FONT_UI, FONT_SIZE, "bold")).pack(anchor=tk.W, pady=(0, 6))
            canv = tk.Canvas(pan_inner, width=320, height=220, bg=CARD_BG, highlightthickness=0)
            canv.pack(fill=tk.BOTH, expand=True)
            setattr(self, attr, canv)

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = tk.Frame(main, bg=SURFACE_BORDER, height=28)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        status_bar.pack_propagate(False)
        tk.Label(status_bar, textvariable=self.status_var, bg=SURFACE_BORDER, fg=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM), anchor=tk.W).pack(side=tk.LEFT, padx=10, pady=4)

    def _update_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    def _on_browse(self):
        path = filedialog.askdirectory(title="Select folder with images")
        if not path:
            return
        self.folder_path.set(path)
        self._scan_images(path)

    def _scan_images(self, folder):
        self.image_files = []
        try:
            for name in sorted(os.listdir(folder)):
                if name.lower().endswith(IMG_EXTS):
                    self.image_files.append(os.path.join(folder, name))
        except Exception as e:
            messagebox.showerror("Error", f"Could not list folder: {e}")
            return
        if not self.image_files:
            self._update_status("No images found in folder.")
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(self.preview_canvas.winfo_reqwidth() // 2, 100,
                                           text="No images found.", fill=TEXT_MUTED, font=(FONT_UI, FONT_SIZE))
            self.seg_canvas.delete("all")
            self.ann_canvas.delete("all")
            return
        self.image_combo["values"] = [os.path.basename(p) for p in self.image_files]
        self.image_combo.current(0)
        self._on_select_image()
        db.set_config("last_folder", folder)
        self._update_status(f"Loaded {len(self.image_files)} images from folder.")

    def _on_select_image(self):
        idx = self.image_combo.current()
        if idx < 0 or idx >= len(self.image_files):
            return
        path = self.image_files[idx]
        self.current_image_path = path
        self._show_preview(path)
        # If we have results for this image, show them
        if path in self.all_detection_results:
            self._display_result(path, self.all_detection_results[path])
        else:
            self._clear_results()

    def _show_preview(self, path):
        self.preview_canvas.delete("all")
        if not HAS_PIL:
            self.preview_canvas.create_text(140, 100, text="PIL not installed.\nInstall Pillow.", fill=TEXT_MUTED)
            return
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                max_w, max_h = self.preview_canvas.winfo_reqwidth(), 200
                r = min(max_w / w, max_h / h, 1.0)
                nw, nh = int(w * r), int(h * r)
                im = im.resize((nw, nh), Image.Resampling.LANCZOS)
                self._preview_photo = ImageTk.PhotoImage(im)
                self.preview_canvas.create_image(max_w // 2, max_h // 2, image=self._preview_photo)
        except Exception as e:
            self.preview_canvas.create_text(140, 100, text=f"Could not load image.\n{e}", fill=TEXT_MUTED)

    def _clear_results(self):
        self.count_var.set("—")
        self.total_var.set("—")
        self.viability_var.set("—")
        for key, (v1, v2) in self.metric_vars.items():
            v1.set("—")
            v2.set("—")
        self.seg_canvas.delete("all")
        self.ann_canvas.delete("all")

    def _display_result(self, image_path, result):
        self.count_var.set(str(result["total_count"]))
        self.total_var.set(str(result["total_count"]))
        self.viability_var.set(str(result["viability_pct"]))
        m_viable = result["metrics"]["viable"]
        m_all = result["metrics"]["all"]
        for key, (v1, v2) in self.metric_vars.items():
            v1.set(f"{m_viable.get(key, 0):.2f}")
            v2.set(f"{m_all.get(key, 0):.2f}")
        self._draw_result_images(image_path, result)

    def _display_result_from_record(self, record):
        """Fill main panel from a DB record (no image path / no detections)."""
        self.count_var.set(str(record.get("total_detections") or 0))
        self.total_var.set(str(record.get("total_detections") or 0))
        self.viability_var.set(str(record.get("viability_pct") or 0))
        for key, (v1, v2) in self.metric_vars.items():
            val = record.get(key)
            if key == "avg_perimeter_sqrt_area" and val is None:
                p, a = record.get("avg_perimeter_um"), record.get("avg_area_um2")
                if p is not None and a is not None and a > 0:
                    val = p / (a ** 0.5)
            if val is not None:
                v1.set(f"{float(val):.2f}")
                v2.set(f"{float(val):.2f}")
            else:
                v1.set("—")
                v2.set("—")
        self.seg_canvas.delete("all")
        self.ann_canvas.delete("all")
        self.seg_canvas.create_text(160, 110, text="(from history — no image)", fill=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM))
        self.ann_canvas.create_text(160, 110, text="(from history — no image)", fill=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM))

    def load_history_record(self, record):
        """Load a history record onto the main screen: set folder, select image, show data."""
        folder_path = record.get("folder_path") or ""
        image_name = record.get("image_name") or ""
        image_path = record.get("image_path") or (os.path.join(folder_path, image_name) if folder_path and image_name else None)
        self.folder_path.set(folder_path)
        if folder_path and os.path.isdir(folder_path):
            self._scan_images(folder_path)
            for i, path in enumerate(self.image_files):
                if os.path.basename(path) == image_name:
                    self.image_combo.current(i)
                    self.current_image_path = path
                    self._show_preview(path)
                    crop = self.crop_var.get()
                    obj = self.objective_var.get()
                    result = run_inference(path, crop_type=crop, objective=obj)
                    self.all_detection_results[path] = result
                    self._display_result(path, result)
                    self._update_status(f"Loaded from history: {image_name}")
                    return
        if image_path and os.path.isfile(image_path):
            self.current_image_path = image_path
            if folder_path and not self.image_files:
                self._scan_images(folder_path)
            self._show_preview(image_path)
            crop = self.crop_var.get()
            obj = self.objective_var.get()
            result = run_inference(image_path, crop_type=crop, objective=obj)
            self.all_detection_results[image_path] = result
            self._display_result(image_path, result)
            self._update_status(f"Loaded from history: {image_name}")
            return
        self._display_result_from_record(record)
        self._update_status(f"Loaded from history (metrics only): {image_name}")

    def _draw_result_images(self, image_path, result):
        self.seg_canvas.delete("all")
        self.ann_canvas.delete("all")
        if not HAS_PIL or not result.get("detections"):
            return
        try:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                max_wh = 320
                max_ht = 220
                r = min(max_wh / w, max_ht / h, 1.0)
                nw, nh = int(w * r), int(h * r)
                im = im.resize((nw, nh), Image.Resampling.LANCZOS)
                scale = nw / w
                cx, cy = max_wh // 2, max_ht // 2
                from PIL import ImageDraw
                seg_img = im.copy()
                draw = ImageDraw.Draw(seg_img)
                for d in result["detections"]:
                    x1, y1, x2, y2 = d["box"]
                    x1, y1 = int(x1 * scale), int(y1 * scale)
                    x2, y2 = int(x2 * scale), int(y2 * scale)
                    draw.rectangle([x1, y1, x2, y2], outline="#2563eb", width=2)
                self._seg_photo = ImageTk.PhotoImage(seg_img)
                self.seg_canvas.create_image(cx, cy, image=self._seg_photo)
                ann_img = im.copy()
                draw2 = ImageDraw.Draw(ann_img)
                for d in result["detections"]:
                    x1, y1, x2, y2 = d["box"]
                    x1, y1 = int(x1 * scale), int(y1 * scale)
                    x2, y2 = int(x2 * scale), int(y2 * scale)
                    color = "#10b981" if d["class"] == 0 else "#ef4444"
                    draw2.rectangle([x1, y1, x2, y2], outline=color, width=2)
                self._ann_photo = ImageTk.PhotoImage(ann_img)
                self.ann_canvas.create_image(cx, cy, image=self._ann_photo)
        except Exception as e:
            self.seg_canvas.create_text(160, 110, text=str(e), fill=TEXT_MUTED)

    def _on_run(self):
        if not self.current_image_path:
            messagebox.showinfo("Run", "Select an image first (Browse folder, then choose image).")
            return
        self._update_status("Running inference...")
        self.update_idletasks()
        crop = self.crop_var.get()
        obj = self.objective_var.get()
        result = run_inference(self.current_image_path, crop_type=crop, objective=obj)
        self.all_detection_results[self.current_image_path] = result
        self._display_result(self.current_image_path, result)
        # Log to DB (single image = one run)
        self.run_id = db.insert_run(
            os.path.dirname(self.current_image_path), crop, self.model_var.get(), obj, 1,
        )
        db.insert_image_result(
            self.run_id,
            self.current_image_path,
            os.path.basename(self.current_image_path),
            result["total_count"],
            result["viable_count"],
            result["viability_pct"],
            result["inference_time_ms"],
            result["metrics"]["all"]["avg_diameter_um"],
            result["metrics"]["all"]["avg_area_um2"],
            result["metrics"]["all"]["avg_perimeter_um"],
            result["metrics"]["all"]["avg_circularity"],
            result["metrics"]["all"]["avg_aspect_ratio"],
        )
        self._update_status(f"Done. Count={result['total_count']}, Viability={result['viability_pct']}%. Logged to DB.")

    def _on_batch_run(self):
        if not self.image_files:
            messagebox.showinfo("Batch run", "Browse folder first to load images.")
            return
        self._update_status("Batch run...")
        self.update_idletasks()
        crop = self.crop_var.get()
        obj = self.objective_var.get()
        self.run_id = db.insert_run(self.folder_path.get(), crop, self.model_var.get(), obj, len(self.image_files))
        for i, path in enumerate(self.image_files):
            self._update_status(f"Batch run: {i+1}/{len(self.image_files)} — {os.path.basename(path)}")
            self.update_idletasks()
            result = run_inference(path, crop_type=crop, objective=obj)
            self.all_detection_results[path] = result
            db.insert_image_result(
                self.run_id,
                path,
                os.path.basename(path),
                result["total_count"],
                result["viable_count"],
                result["viability_pct"],
                result["inference_time_ms"],
                result["metrics"]["all"]["avg_diameter_um"],
                result["metrics"]["all"]["avg_area_um2"],
                result["metrics"]["all"]["avg_perimeter_um"],
                result["metrics"]["all"]["avg_circularity"],
                result["metrics"]["all"]["avg_aspect_ratio"],
            )
        if self.current_image_path and self.current_image_path in self.all_detection_results:
            self._display_result(self.current_image_path, self.all_detection_results[self.current_image_path])
        self._update_status(f"Batch complete. {len(self.image_files)} images logged to DB.")

    def _on_export(self):
        if not self.all_detection_results:
            messagebox.showinfo("Export", "Run analysis first (Run or Batch run).")
            return
        out_dir = get_results_dir()
        base = os.path.join(out_dir, "DeepSpot_Export")
        os.makedirs(out_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"{base}_{ts}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Image Name,Total Detections,Viable Count,Non-Viable Count,Viability (%),Inference Time (ms),Avg Diameter,Avg Area,Avg Aspect\n")
            for path, res in self.all_detection_results.items():
                name = os.path.basename(path)
                nv = res["total_count"] - res["viable_count"]
                m = res["metrics"]["all"]
                f.write(f"{name},{res['total_count']},{res['viable_count']},{nv},{res['viability_pct']},{res['inference_time_ms']},{m['avg_diameter_um']:.2f},{m['avg_area_um2']:.2f},{m['avg_aspect_ratio']:.2f}\n")
        self._update_status(f"Exported to {csv_path}")
        messagebox.showinfo("Export", f"Results exported to:\n{csv_path}")

    def _load_test_data(self):
        # Create a few dummy image files so Test Data works without user images
        test_dir = os.path.join(os.path.dirname(__file__), "test_images")
        os.makedirs(test_dir, exist_ok=True)
        if HAS_PIL:
            from PIL import Image as PImage
            for i in range(3):
                path = os.path.join(test_dir, f"test_{i+1}.png")
                if not os.path.exists(path):
                    img = PImage.new("RGB", (400, 300), color=(240, 248, 255))
                    img.save(path)
            self.folder_path.set(test_dir)
            self._scan_images(test_dir)
            self._update_status("Test data loaded (dummy images). Use Run or Batch run.")
        else:
            messagebox.showinfo("Test Data", "Install Pillow to use Test Data. Or use Browse folder with your images.")

    def _on_history(self):
        """Show history: image name, folder name, open at. Click a record to load its data on the main screen."""
        win = tk.Toplevel(self)
        win.title("Run History")
        win.minsize(650, 420)
        win.geometry("750x460")
        win.configure(bg=BG_APP)
        header = tk.Frame(win, bg=HEADER_BG, height=44)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="  Run History  ·  Shared DB  ", bg=HEADER_BG, fg=HEADER_TEXT, font=(FONT_UI, 12, "bold")).pack(side=tk.LEFT, padx=12, pady=10)
        f = tk.Frame(win, bg=BG_APP, padx=16, pady=14)
        f.pack(fill=tk.BOTH, expand=True)
        tk.Label(f, text="Click a row to load that record on the main screen. Same history on all laptops when DB is on a network drive.", bg=BG_APP, fg=TEXT_MUTED, font=(FONT_UI, FONT_SIZE_SM), wraplength=700).pack(anchor=tk.W)
        tk.Label(f, text=f"DB: {get_db_path()}", bg=BG_APP, fg=TEXT_MUTED, font=(FONT_UI, 8)).pack(anchor=tk.W, pady=(2, 0))
        tk.Label(f, text="Image name · Folder name · Open at", bg=BG_APP, fg=TEXT, font=(FONT_UI, FONT_SIZE_SM, "bold")).pack(anchor=tk.W, pady=(12, 6))
        tree_outer = tk.Frame(f, bg=SURFACE_BORDER)
        tree_outer.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        tree_inner = tk.Frame(tree_outer, bg=SURFACE, padx=1, pady=1)
        tree_inner.pack(fill=tk.BOTH, expand=True)
        tree_frame = tk.Frame(tree_inner, bg=SURFACE)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        cols = ("image_name", "folder_name", "open_at")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=16, selectmode="browse")
        vsb = ttk.Scrollbar(tree_frame)
        tree.configure(yscrollcommand=vsb.set)
        vsb.configure(command=tree.yview)
        tree.heading("image_name", text="Image name")
        tree.heading("folder_name", text="Folder name")
        tree.heading("open_at", text="Open at")
        tree.column("image_name", width=220)
        tree.column("folder_name", width=320)
        tree.column("open_at", width=180)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        history_records = []

        def refresh():
            for i in tree.get_children():
                tree.delete(i)
            history_records.clear()
            try:
                rows = db.get_recent_image_results(limit=300)
                for idx, r in enumerate(rows):
                    history_records.append(r)
                    folder = (r.get("folder_path") or "")
                    folder_short = folder[:55] + ("..." if len(folder) > 55 else "")
                    open_at = (r.get("created_at") or "")[:19].replace("T", " ")
                    tree.insert("", tk.END, iid=str(idx), values=(r.get("image_name") or "", folder_short, open_at))
            except Exception as e:
                messagebox.showerror("History", f"Could not load DB: {e}", parent=win)

        def on_click(evt):
            sel = tree.selection()
            if not sel:
                return
            idx = int(sel[0])
            if idx < 0 or idx >= len(history_records):
                return
            record = history_records[idx]
            win.destroy()
            self.load_history_record(record)
            self.focus_force()

        tree.bind("<<TreeviewSelect>>", on_click)
        tk.Label(f, text="Click a row to open its data on the main screen.", bg=BG_APP, fg=TEXT_MUTED, font=(FONT_UI, 8)).pack(anchor=tk.W, pady=(0, 6))
        refresh_btn = tk.Button(
            f, text="Refresh", command=refresh,
            bg=ACCENT, fg=_button_text_color(), activebackground=ACCENT, activeforeground=_button_text_color(),
            font=(FONT_UI, FONT_SIZE, "bold"), relief=tk.FLAT, bd=0, padx=14, pady=8, cursor="hand2",
            highlightthickness=0,
        )
        refresh_btn.pack(anchor=tk.W)
        refresh()


def main():
    app = DeepSpotApp()
    app.mainloop()


if __name__ == "__main__":
    main()
