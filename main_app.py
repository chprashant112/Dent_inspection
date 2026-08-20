import os
import sys
import glob
import time
import importlib.util
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import serial
import serial.tools.list_ports


# =============================================================================
# 1. CAMERA THREAD WORKER
# =============================================================================
class CameraStream:
    """Threaded camera capture class for smooth, lag-free UI frame processing."""
    def __init__(self, src=0):
        # Convert numeric strings to int if passing camera index
        if str(src).isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.read_lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


# =============================================================================
# 2. BUILT-IN DETECTOR LOGIC (Laplacian V1)
# =============================================================================
def detect_laplacian_v1(img, k_multiplier=3.0, ksize=3):
    """
    Laplacian V1 detection logic.
    Accepts BGR image input, processes gray/laplacian, returns output BGR image.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply Laplacian Filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # Dynamic Thresholding
    mean_val, std_dev = cv2.meanStdDev(laplacian_abs)
    threshold_value = mean_val[0][0] + (k_multiplier * std_dev[0][0])
    _, binary_mask = cv2.threshold(laplacian_abs, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours and draw bounding boxes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 1 or h > 1:
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return result_img


# =============================================================================
# 3. MAIN APPLICATION GUI
# =============================================================================
class VisionInspectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Industrial Defect Inspection & Lighting Control System")
        self.root.geometry("1400x850")
        self.root.minsize(1000, 700)

        # Serial Connection Instance
        self.ser = None

        # Camera Instance
        self.cam = None

        # Algorithm Plugin Engine
        self.algorithms = {}
        self.load_algorithms()

        # Transformation & ROI State
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # ROI in Image Pixel Space: (x1, y1, x2, y2)
        self.roi = None
        self.is_drawing_roi = False
        self.roi_start = None

        # Frame Processing
        self.current_raw_frame = None

        self._build_layout()
        self.update_loop()

    # -------------------------------------------------------------------------
    # GUI LAYOUT CREATION
    # -------------------------------------------------------------------------
    def _build_layout(self):
        # Main Split Frame: Left (Video Canvas) & Right (Controls)
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True)

        # ---------------- LEFT PANEL (CANVAS & TOOLBAR) ----------------
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)

        # Canvas Control Toolbar
        toolbar = ttk.Frame(left_frame, padding=5)
        toolbar.pack(fill="x", side="top")

        ttk.Button(toolbar, text="Reset View / Zoom", command=self.reset_zoom).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Clear ROI", command=self.clear_roi).pack(side="left", padx=5)
        self.lbl_info = ttk.Label(toolbar, text="Left-Click + Drag: Draw ROI | Right-Click + Drag: Pan | Scroll: Zoom")
        self.lbl_info.pack(side="right", padx=10)

        # Canvas for displaying image stream
        self.canvas = tk.Canvas(left_frame, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Mouse Bindings for Canvas Interactivity
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", self.on_zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)  # Linux scroll down

        self.canvas.bind("<ButtonPress-1>", self.on_roi_start)
        self.canvas.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_roi_end)

        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)

        # ---------------- RIGHT PANEL (SCROLLABLE CONTROLS) ----------------
        right_container = ttk.Frame(main_paned)
        main_paned.add(right_container, weight=1)

        # Add Scrollable Frame mechanism to Right Panel
        right_canvas = tk.Canvas(right_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=right_canvas.yview)
        self.scroll_frame = ttk.Frame(right_canvas, padding=10)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )

        right_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        right_canvas.configure(yscrollcommand=scrollbar.set)

        right_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Build Sections in Right Panel
        self._build_camera_controls()
        self._build_algorithm_controls()
        self._build_light_controls()

    # -------------------------------------------------------------------------
    # RIGHT PANEL SECTION BUILDERS
    # -------------------------------------------------------------------------
    def _build_camera_controls(self):
        cam_frame = ttk.LabelFrame(self.scroll_frame, text=" Camera Controls ", padding=10)
        cam_frame.pack(fill="x", pady=5)

        ttk.Label(cam_frame, text="Source Index / Path:").grid(row=0, column=0, sticky="w", pady=2)
        self.ent_cam_src = ttk.Entry(cam_frame, width=12)
        self.ent_cam_src.insert(0, "0")
        self.ent_cam_src.grid(row=0, column=1, padx=5, pady=2)

        self.btn_cam_toggle = ttk.Button(cam_frame, text="Start Camera", command=self.toggle_camera)
        self.btn_cam_toggle.grid(row=0, column=2, padx=5, pady=2)

    def _build_algorithm_controls(self):
        algo_frame = ttk.LabelFrame(self.scroll_frame, text=" Detection Algorithm & Controls ", padding=10)
        algo_frame.pack(fill="x", pady=5)

        # Algorithm Selector
        ttk.Label(algo_frame, text="Select Algorithm:").grid(row=0, column=0, sticky="w", pady=5)
        self.algo_cb = ttk.Combobox(algo_frame, values=list(self.algorithms.keys()), state="readonly")
        if self.algorithms:
            self.algo_cb.set("Laplacian V1 (Built-in)")
        self.algo_cb.grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)

        ttk.Button(algo_frame, text="Reload Algorithms", command=self.reload_algorithms_list).grid(row=1, column=0, columnspan=3, sticky="ew", pady=2)

        ttk.Separator(algo_frame, orient="horizontal").grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)

        # Laplacian Parameters
        ttk.Label(algo_frame, text="k-Multiplier (Threshold):").grid(row=3, column=0, columnspan=2, sticky="w")
        self.lbl_k_val = ttk.Label(algo_frame, text="3.0")
        self.lbl_k_val.grid(row=3, column=2, sticky="e")

        self.scale_k = ttk.Scale(algo_frame, from_=0.5, to=10.0, value=3.0, command=self._on_k_slider_move)
        self.scale_k.grid(row=4, column=0, columnspan=3, sticky="ew", pady=2)

        ttk.Label(algo_frame, text="Filter Kernel Size (ksize):").grid(row=5, column=0, sticky="w", pady=5)
        self.ksize_cb = ttk.Combobox(algo_frame, values=["1", "3", "5", "7"], width=5, state="readonly")
        self.ksize_cb.set("3")
        self.ksize_cb.grid(row=5, column=1, sticky="w", padx=5, pady=5)

    def _build_light_controls(self):
        # LIGHT CONTROLLER INTEGRATION
        conn_frame = ttk.LabelFrame(self.scroll_frame, text=" OPT Light Controller ", padding=10)
        conn_frame.pack(fill="x", pady=5)

        # Serial Connection Setup
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, padx=2)
        self.port_cb = ttk.Combobox(conn_frame, values=self.get_ports(), width=10)
        self.port_cb.grid(row=0, column=1, padx=2)

        ttk.Label(conn_frame, text="Baud:").grid(row=0, column=2, padx=2)
        self.baud_cb = ttk.Combobox(conn_frame, values=["19200", "9600", "115200"], width=8)
        self.baud_cb.set("19200")
        self.baud_cb.grid(row=0, column=3, padx=2)

        self.btn_connect = ttk.Button(conn_frame, text="Connect", command=self.toggle_serial_connection)
        self.btn_connect.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)

        # Channel Brightness Sliders (1 to 4)
        ctrl_frame = ttk.LabelFrame(self.scroll_frame, text=" Channel Brightness (0 - 255) ", padding=10)
        ctrl_frame.pack(fill="x", pady=5)

        self.sliders = []
        self.val_labels = []

        for ch in range(1, 5):
            row_frame = ttk.Frame(ctrl_frame)
            row_frame.pack(fill="x", pady=4)

            ttk.Label(row_frame, text=f"Ch {ch}:", width=6).pack(side="left")

            slider = ttk.Scale(
                row_frame,
                from_=0,
                to=255,
                orient="horizontal",
                command=lambda val, c=ch: self.on_slider_move(c, val)
            )
            slider.pack(side="left", fill="x", expand=True, padx=5)
            self.sliders.append(slider)

            val_label = ttk.Label(row_frame, text="0", width=4)
            val_label.pack(side="right")
            self.val_labels.append(val_label)

    # -------------------------------------------------------------------------
    # DYNAMIC ALGORITHM PLUGIN ENGINE
    # -------------------------------------------------------------------------
    def load_algorithms(self):
        """Scans `./algorithms` directory for plugins and maps built-in functions."""
        self.algorithms = {
            "Laplacian V1 (Built-in)": detect_laplacian_v1
        }

        algo_dir = os.path.join(os.path.dirname(__file__), "algorithms")
        if not os.path.exists(algo_dir):
            os.makedirs(algo_dir, exist_ok=True)

        # Look for .py files inside algorithms folder
        for file_path in glob.glob(os.path.join(algo_dir, "*.py")):
            filename = os.path.basename(file_path)
            if filename.startswith("__"):
                continue

            module_name = filename[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                if hasattr(mod, "process_frame"):
                    algo_title = getattr(mod, "NAME", module_name)
                    self.algorithms[algo_title] = mod.process_frame
            except Exception as e:
                print(f"Failed to load algorithm plugin {filename}: {e}")

    def reload_algorithms_list(self):
        self.load_algorithms()
        self.algo_cb['values'] = list(self.algorithms.keys())
        messagebox.showinfo("Algorithm Loader", f"Loaded {len(self.algorithms)} algorithm(s).")

    # -------------------------------------------------------------------------
    # CAMERA CONTROLS & STREAMING
    # -------------------------------------------------------------------------
    def toggle_camera(self):
        if self.cam and self.cam.started:
            self.cam.stop()
            self.cam = None
            self.btn_cam_toggle.config(text="Start Camera")
        else:
            src = self.ent_cam_src.get().strip()
            try:
                self.cam = CameraStream(src).start()
                self.btn_cam_toggle.config(text="Stop Camera")
            except Exception as e:
                messagebox.showerror("Camera Error", f"Unable to open video source:\n{e}")

    def _on_k_slider_move(self, val):
        self.lbl_k_val.config(text=f"{float(val):.1f}")

    # -------------------------------------------------------------------------
    # SERIAL LIGHT CONTROLLER LOGIC
    # -------------------------------------------------------------------------
    def get_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        return ports if ports else ["COM1", "COM3", "/dev/ttyUSB0"]

    def toggle_serial_connection(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.btn_connect.config(text="Connect")
            messagebox.showinfo("Status", "Disconnected from device.")
        else:
            port = self.port_cb.get()
            baud = self.baud_cb.get()
            if not port:
                messagebox.showwarning("Error", "Please select a COM port.")
                return
            try:
                self.ser = serial.Serial(port, int(baud), timeout=1)
                self.btn_connect.config(text="Disconnect")
                messagebox.showinfo("Status", f"Connected to {port}")
            except Exception as e:
                messagebox.showerror("Connection Error", str(e))

    def send_command(self, channel, val):
        if self.ser and self.ser.is_open:
            # OPT standard command syntax: $3<Channel><Value_in_3_digits>
            cmd = f"$3{channel}{val:03d}\r\n"
            try:
                self.ser.write(cmd.encode('ascii'))
            except Exception as e:
                print(f"Write error: {e}")

    def on_slider_move(self, channel, val):
        val_int = int(float(val))
        self.val_labels[channel - 1].config(text=str(val_int))
        self.send_command(channel, val_int)

    # -------------------------------------------------------------------------
    # CANVAS INTERACTIVITY: ZOOM, PAN & ROI TRANSFORMATIONS
    # -------------------------------------------------------------------------
    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

    def clear_roi(self):
        self.roi = None

    def screen_to_img_coords(self, sx, sy):
        """Converts screen GUI canvas coordinates back to raw frame pixel coordinates."""
        ix = int((sx - self.pan_x) / self.zoom_level)
        iy = int((sy - self.pan_y) / self.zoom_level)
        return ix, iy

    def on_zoom(self, event):
        # Zoom factor determination
        if event.num == 4 or event.delta > 0:
            zoom_factor = 1.1
        else:
            zoom_factor = 0.9

        # Prevent zoom from going too small or unnecessarily huge
        new_zoom = self.zoom_level * zoom_factor
        if new_zoom < 0.2 or new_zoom > 10.0:
            return

        # Zoom focused at current mouse position
        mx, my = event.x, event.y
        self.pan_x = mx - (mx - self.pan_x) * zoom_factor
        self.pan_y = my - (my - self.pan_y) * zoom_factor
        self.zoom_level = new_zoom

    def on_pan_start(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_pan_drag(self, event):
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.pan_x += dx
        self.pan_y += dy
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_roi_start(self, event):
        self.is_drawing_roi = True
        self.roi_start = (event.x, event.y)

    def on_roi_drag(self, event):
        if self.is_drawing_roi:
            x1, y1 = self.screen_to_img_coords(self.roi_start[0], self.roi_start[1])
            x2, y2 = self.screen_to_img_coords(event.x, event.y)
            self.roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def on_roi_end(self, event):
        self.is_drawing_roi = False
        if self.roi_start:
            x1, y1 = self.screen_to_img_coords(self.roi_start[0], self.roi_start[1])
            x2, y2 = self.screen_to_img_coords(event.x, event.y)
            # Ensure ROI is valid size
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                self.roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            else:
                self.roi = None

    # -------------------------------------------------------------------------
    # VIDEO PROCESSING & UI REFRESH LOOP
    # -------------------------------------------------------------------------
    def update_loop(self):
        if self.cam and self.cam.started:
            grabbed, frame = self.cam.read()
            if grabbed and frame is not None:
                self.current_raw_frame = frame
        
        if self.current_raw_frame is None:
            # Generate placeholder frame when camera is off
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Stopped / No Feed", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            display_frame = placeholder
        else:
            display_frame = self.process_frame(self.current_raw_frame)

        self._render_canvas(display_frame)
        self.root.after(30, self.update_loop)

    def process_frame(self, frame):
        """Applies selected algorithm to frame or cropped ROI."""
        out_frame = frame.copy()
        h, w = out_frame.shape[:2]

        selected_algo_name = self.algo_cb.get()
        algo_fn = self.algorithms.get(selected_algo_name, detect_laplacian_v1)

        k_mult = float(self.scale_k.get())
        ksize = int(self.ksize_cb.get())

        params = {
            'k_multiplier': k_mult,
            'ksize': ksize
        }

        # Apply detection either inside ROI or on full frame
        if self.roi:
            x1, y1, x2, y2 = self.roi
            # Clamp ROI inside frame dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                roi_crop = out_frame[y1:y2, x1:x2]
                
                # Dynamic signature check for dynamic plugins
                try:
                    processed_roi = algo_fn(roi_crop, **params)
                except TypeError:
                    processed_roi = algo_fn(roi_crop, params)

                out_frame[y1:y2, x1:x2] = processed_roi
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan ROI outline
        else:
            try:
                out_frame = algo_fn(out_frame, **params)
            except TypeError:
                out_frame = algo_fn(out_frame, params)

        return out_frame

    def _render_canvas(self, frame):
        """Resizes frame according to Zoom/Pan and displays on Tkinter Canvas."""
        h, w = frame.shape[:2]
        new_w = max(1, int(w * self.zoom_level))
        new_h = max(1, int(h * self.zoom_level))

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert PIL to ImageTk
        pil_img = Image.fromarray(resized_img)
        self.tk_img = ImageTk.PhotoImage(image=pil_img)

        # Clear canvas and redraw image
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.tk_img)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = VisionInspectionApp(root)
    root.mainloop()
