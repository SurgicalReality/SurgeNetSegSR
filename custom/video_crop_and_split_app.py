"""
Interactive Video Crop and Split Application

Combine the functionality of:
- interactive_crop_adjustment.ipynb: Interactive crop parameter adjustment
- crop_video.py: Video cropping functionality
- split_video.py: Video splitting functionality

Features:
- Load and preview surgical videos
- Interactively adjust crop parameters with real-time preview
- Save crop settings to crop_config.json
- Split cropped videos into clips of specified duration
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import threading
from tqdm import tqdm
import traceback
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from moviepy import VideoFileClip
except ImportError as e:
    print(f"Warning: moviepy not installed properly. Video splitting will not work.")
    print(f"Install with: pip install moviepy")
    print(f"Import error: {e}")

# Default output directory
DEFAULT_OUTPUT_DIR = r"C:\Users\Skyfinder\Projects\SurgeNetSeg\custom\data\output_clips"


def _process_clip_worker(args):
    """
    Module-level worker for ProcessPoolExecutor.
    Extracts, crops and encodes one clip from the source video.
    Returns (success: bool, clip_path: str, error_tb: str | None).
    """
    video_path, start_time, end_time, clip_path, crop_params, out_fps = args
    try:
        from moviepy import VideoFileClip
        from moviepy.video.fx import Crop
        video = VideoFileClip(str(video_path))
        clip = video.subclipped(start_time, end_time)

        left   = crop_params['left']
        right  = crop_params['right']
        top    = crop_params['top']
        bottom = crop_params['bottom']

        if left > 0 or right > 0 or top > 0 or bottom > 0:
            w, h = clip.size
            new_w = w - left - right
            new_h = h - top - bottom
            if new_w <= 0 or new_h <= 0:
                raise ValueError(f"Invalid crop: would produce {new_w}x{new_h}")
            clip_cropped = clip.with_effects([Crop(x1=left, y1=top, x2=w - right, y2=h - bottom)])
        else:
            clip_cropped = clip

        clip_cropped.write_videofile(
            str(clip_path),
            codec='libx264',
            audio=False,
            remove_temp=True,
            logger=None,
            fps=out_fps,
        )
        clip_cropped.close()
        clip.close()
        video.close()
        return (True, str(clip_path), None)
    except Exception:
        import traceback as _tb
        return (False, str(clip_path), _tb.format_exc())


class VideoCropAndSplitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Crop & Split Tool")
        self.root.geometry("1600x1000")
        
        # Video variables
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame = 0
        self.video_path = None
        self.video_name = "unknown"
        self.max_display_size = (720, 1280)
        self.original_width = 0
        self.original_height = 0
        
        # Crop variables
        self.crop_params = {
            'left': 0,
            'right': 0,
            'top': 0,
            'bottom': 0
        }
        
        # Processing control
        self.stop_processing = False
        self.processing_thread = None
        self._executor = None
        self._futures = []
        
        # Config file path
        self.config_path = "./custom/crop_config.json"
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface with scrollable content"""
        # Create main container with scrollbar
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(main_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create scrollable canvas
        self.main_canvas = tk.Canvas(main_container, yscrollcommand=scrollbar.set)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.main_canvas.yview)
        
        # Create scrollable frame inside canvas
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        self.scrollable_frame_id = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Bind mousewheel to canvas
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Update scroll region when frame changes size
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        
        # ===== VIDEO LOADING SECTION =====
        load_frame = ttk.LabelFrame(self.scrollable_frame, text="Load Video", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 10), padx=10, anchor="w")
        
        ttk.Label(load_frame, text="Video File:").pack(side=tk.LEFT)
        self.video_path_var = tk.StringVar()
        ttk.Entry(load_frame, textvariable=self.video_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="Browse...", command=self.browse_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        
        self.load_status = tk.StringVar(value="No video loaded")
        ttk.Label(load_frame, textvariable=self.load_status, foreground="red").pack(side=tk.LEFT, padx=5)
        
        # ===== CROP ADJUSTMENT SECTION =====
        crop_frame = ttk.LabelFrame(self.scrollable_frame, text="Crop Adjustment", padding=10)
        crop_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10), padx=10, anchor="w")
        
        # Frame slider
        slider_frame = ttk.Frame(crop_frame)
        slider_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Time and frame info
        self.time_label = ttk.Label(slider_frame, text="00:00:00 / 00:00:00")
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        self.frame_info_label = ttk.Label(slider_frame, text="Frame: 0 / 0")
        self.frame_info_label.pack(side=tk.LEFT, padx=5)
        
        # Two-column layout for crop controls and preview
        two_column_frame = ttk.Frame(crop_frame)
        two_column_frame.pack(fill=tk.BOTH, expand=True)
        
        # LEFT SIDE: Video display with crop guides
        left_side = ttk.LabelFrame(two_column_frame, text="Crop Adjustment", padding=5)
        left_side.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        
        # Video display canvas - fixed size
        self.video_canvas_width = 960
        self.video_canvas_height = 720
        self.video_label = tk.Canvas(left_side, bg="black", width=self.video_canvas_width, height=self.video_canvas_height)
        self.video_label.pack(pady=5)
        
        # Crop sliders below video
        sliders_frame = ttk.LabelFrame(left_side, text="Crop Parameters (pixels)", padding=10)
        sliders_frame.pack(fill=tk.X, pady=10)
        
        # Left crop slider
        left_frame = ttk.Frame(sliders_frame)
        left_frame.pack(fill=tk.X, pady=5)
        ttk.Label(left_frame, text="Left:", width=10).pack(side=tk.LEFT)
        self.left_var = tk.IntVar(value=0)
        self.left_slider = ttk.Scale(left_frame, from_=0, to=1920, orient=tk.HORIZONTAL, variable=self.left_var, command=self.on_crop_change)
        self.left_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.left_label = ttk.Label(left_frame, text="0", width=5)
        self.left_label.pack(side=tk.LEFT)
        ttk.Button(left_frame, text="-", width=2, command=lambda: self.adjust_slider(self.left_var, -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_frame, text="+", width=2, command=lambda: self.adjust_slider(self.left_var, 1)).pack(side=tk.LEFT, padx=2)

        # Right crop slider
        right_frame = ttk.Frame(sliders_frame)
        right_frame.pack(fill=tk.X, pady=5)
        ttk.Label(right_frame, text="Right:", width=10).pack(side=tk.LEFT)
        self.right_var = tk.IntVar(value=0)
        self.right_slider = ttk.Scale(right_frame, from_=0, to=1920, orient=tk.HORIZONTAL, variable=self.right_var, command=self.on_crop_change)
        self.right_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.right_label = ttk.Label(right_frame, text="0", width=5)
        self.right_label.pack(side=tk.LEFT)
        ttk.Button(right_frame, text="-", width=2, command=lambda: self.adjust_slider(self.right_var, -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(right_frame, text="+", width=2, command=lambda: self.adjust_slider(self.right_var, 1)).pack(side=tk.LEFT, padx=2)

        # Top crop slider
        top_frame = ttk.Frame(sliders_frame)
        top_frame.pack(fill=tk.X, pady=5)
        ttk.Label(top_frame, text="Top:", width=10).pack(side=tk.LEFT)
        self.top_var = tk.IntVar(value=0)
        self.top_slider = ttk.Scale(top_frame, from_=0, to=1080, orient=tk.HORIZONTAL, variable=self.top_var, command=self.on_crop_change)
        self.top_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.top_label = ttk.Label(top_frame, text="0", width=5)
        self.top_label.pack(side=tk.LEFT)
        ttk.Button(top_frame, text="-", width=2, command=lambda: self.adjust_slider(self.top_var, -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="+", width=2, command=lambda: self.adjust_slider(self.top_var, 1)).pack(side=tk.LEFT, padx=2)

        # Bottom crop slider
        bottom_frame = ttk.Frame(sliders_frame)
        bottom_frame.pack(fill=tk.X, pady=5)
        ttk.Label(bottom_frame, text="Bottom:", width=10).pack(side=tk.LEFT)
        self.bottom_var = tk.IntVar(value=0)
        self.bottom_slider = ttk.Scale(bottom_frame, from_=0, to=1080, orient=tk.HORIZONTAL, variable=self.bottom_var, command=self.on_crop_change)
        self.bottom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.bottom_label = ttk.Label(bottom_frame, text="0", width=5)
        self.bottom_label.pack(side=tk.LEFT)
        ttk.Button(bottom_frame, text="-", width=2, command=lambda: self.adjust_slider(self.bottom_var, -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_frame, text="+", width=2, command=lambda: self.adjust_slider(self.bottom_var, 1)).pack(side=tk.LEFT, padx=2)
        
        # RIGHT SIDE: Crop preview
        right_side = ttk.LabelFrame(two_column_frame, text="Crop Preview", padding=5)
        right_side.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))

        # Preview resolution label
        self.preview_res_label = ttk.Label(right_side, text="Cropped output: 0x0")
        self.preview_res_label.pack(anchor="w", pady=(0, 5))

        # Preview canvas - resizes dynamically to match cropped output resolution
        self.max_preview_w = 960
        self.max_preview_h = 720
        self.preview_label = tk.Canvas(right_side, bg="black", width=self.max_preview_w, height=self.max_preview_h)
        self.preview_label.pack(pady=5)

        # Border patch previews (200x200 each, showing a 20x20 px region centred on each crop line)
        patches_outer = ttk.LabelFrame(right_side, text="Border Patches  (20×20 px → 200×200 display)", padding=5)
        patches_outer.pack(fill=tk.X, pady=5)

        patches_row1 = ttk.Frame(patches_outer)
        patches_row1.pack(fill=tk.X)
        patches_row2 = ttk.Frame(patches_outer)
        patches_row2.pack(fill=tk.X)

        left_patch_frame = ttk.LabelFrame(patches_row1, text="Left border", padding=3)
        left_patch_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.patch_left = tk.Canvas(left_patch_frame, bg="black", width=200, height=200)
        self.patch_left.pack()

        right_patch_frame = ttk.LabelFrame(patches_row1, text="Right border", padding=3)
        right_patch_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.patch_right = tk.Canvas(right_patch_frame, bg="black", width=200, height=200)
        self.patch_right.pack()

        top_patch_frame = ttk.LabelFrame(patches_row2, text="Top border", padding=3)
        top_patch_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.patch_top = tk.Canvas(top_patch_frame, bg="black", width=200, height=200)
        self.patch_top.pack()

        bottom_patch_frame = ttk.LabelFrame(patches_row2, text="Bottom border", padding=3)
        bottom_patch_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.patch_bottom = tk.Canvas(bottom_patch_frame, bg="black", width=200, height=200)
        self.patch_bottom.pack()

        # ===== CROP SAVE SECTION =====
        save_frame = ttk.LabelFrame(self.scrollable_frame, text="Save Crop Settings", padding=10)
        save_frame.pack(fill=tk.X, pady=(0, 10), padx=10, anchor="w")

        res_frame = ttk.Frame(save_frame)
        res_frame.pack(fill=tk.X, pady=5)
        self.original_res_label = ttk.Label(res_frame, text="Original: 0x0")
        self.original_res_label.pack(side=tk.LEFT, padx=10)
        self.new_res_label = ttk.Label(res_frame, text="Cropped: 0x0")
        self.new_res_label.pack(side=tk.LEFT, padx=10)

        save_btn_frame = ttk.Frame(save_frame)
        save_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(save_btn_frame, text="Save Crop Settings", command=self.save_crop_params).pack(side=tk.LEFT, padx=5)
        self.crop_status_var = tk.StringVar(value="No crop settings saved")
        self.crop_status_label = ttk.Label(save_btn_frame, textvariable=self.crop_status_var, foreground="gray")
        self.crop_status_label.pack(side=tk.LEFT, padx=5)

        # ===== SPLIT VIDEO SECTION =====
        split_frame = ttk.LabelFrame(self.scrollable_frame, text="Split Video", padding=10)
        split_frame.pack(fill=tk.X, pady=(0, 10), padx=10, anchor="w")

        output_frame = ttk.Frame(split_frame)
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_dir).pack(side=tk.LEFT, padx=5)

        duration_frame = ttk.Frame(split_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        ttk.Label(duration_frame, text="Clip Duration (seconds):").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="30")
        ttk.Entry(duration_frame, textvariable=self.duration_var, width=10).pack(side=tk.LEFT, padx=5)

        frame_rate_frame = ttk.Frame(split_frame)
        frame_rate_frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame_rate_frame, text="Output Frame Rate (fps):").pack(side=tk.LEFT)
        ttk.Label(frame_rate_frame, text="(leave blank for video native fps)", foreground="gray").pack(side=tk.LEFT, padx=5)
        self.frame_rate_var = tk.StringVar(value="30")
        ttk.Entry(frame_rate_frame, textvariable=self.frame_rate_var, width=10).pack(side=tk.LEFT, padx=5)

        workers_frame = ttk.Frame(split_frame)
        workers_frame.pack(fill=tk.X, pady=5)
        ttk.Label(workers_frame, text="Parallel Workers:").pack(side=tk.LEFT)
        cpu_count = multiprocessing.cpu_count()
        default_workers = max(1, min(16, cpu_count))
        self.workers_var = tk.IntVar(value=default_workers)
        ttk.Spinbox(workers_frame, from_=1, to=cpu_count, textvariable=self.workers_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(workers_frame, text=f"(CPU cores available: {cpu_count})", foreground="gray").pack(side=tk.LEFT, padx=5)

        split_btn_frame = ttk.Frame(split_frame)
        split_btn_frame.pack(fill=tk.X, pady=5)
        self.split_btn = ttk.Button(split_btn_frame, text="Crop & Split Video", command=self.start_split_video)
        self.split_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(split_btn_frame, text="Stop", command=self.stop_split_video, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.split_status_var = tk.StringVar(value="Ready")
        self.split_status_label = ttk.Label(split_btn_frame, textvariable=self.split_status_var, foreground="gray")
        self.split_status_label.pack(side=tk.LEFT, padx=5)

        # Progress bar for cropping
        crop_progress_frame = ttk.Frame(split_frame)
        crop_progress_frame.pack(fill=tk.X, pady=5)
        ttk.Label(crop_progress_frame, text="Crop Progress:").pack(side=tk.LEFT)
        self.crop_progress = ttk.Progressbar(crop_progress_frame, mode='determinate', length=200)
        self.crop_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.crop_progress_label = ttk.Label(crop_progress_frame, text="0%", width=5)
        self.crop_progress_label.pack(side=tk.LEFT)

        # Progress bar for splitting
        split_progress_frame = ttk.Frame(split_frame)
        split_progress_frame.pack(fill=tk.X, pady=5)
        ttk.Label(split_progress_frame, text="Split Progress:").pack(side=tk.LEFT)
        self.split_progress = ttk.Progressbar(split_progress_frame, mode='determinate', length=200)
        self.split_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.split_progress_label = ttk.Label(split_progress_frame, text="0%", width=5)
        self.split_progress_label.pack(side=tk.LEFT)

    def _on_frame_configure(self, event):
        """Update scroll region when frame changes size"""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def adjust_slider(self, var, delta):
        """Adjust slider value by delta (for arrow key support)"""
        current_value = var.get()
        new_value = max(0, current_value + delta)
        var.set(new_value)
        self.on_crop_change(None)
    
    def browse_video(self):
        """Open file dialog to browse for video"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
    
    def browse_output_dir(self):
        """Open directory dialog for output location"""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir_var.set(dir_path)
    
    def frames_to_time(self, frame_num):
        """Convert frame number to HH:MM:SS format"""
        if self.fps == 0:
            return "00:00:00"
        seconds = frame_num / self.fps
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def load_video(self):
        """Load video file"""
        video_path = self.video_path_var.get()
        
        if not video_path:
            messagebox.showerror("Error", "Please enter or browse a video file path")
            return
        
        if not os.path.exists(video_path):
            messagebox.showerror("Error", f"Video file not found: {video_path}")
            return
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video: {video_path}")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        
        # Extract video name
        self.video_name = Path(video_path).stem
        self.video_path = video_path
        
        # Update slider max values based on actual video dimensions
        self.left_slider.config(to=self.original_width)
        self.right_slider.config(to=self.original_width)
        self.top_slider.config(to=self.original_height)
        self.bottom_slider.config(to=self.original_height)
        
        # Update frame slider
        self.frame_slider.configure(to=self.total_frames - 1 if self.total_frames > 0 else 100)
        
        # Update status
        self.load_status.set(f"✓ Video loaded: {Path(video_path).name} ({self.original_width}x{self.original_height}, {self.fps:.1f}fps)")
        
        # Try to load existing crop params
        self.load_crop_params()
        
        # Display first frame
        self.display_frame(0)
        
        # Set default output directory if not already set
        if not self.output_dir_var.get():
            self.output_dir_var.set(DEFAULT_OUTPUT_DIR)
    
    def load_crop_params(self):
        """Load crop parameters from config file if available"""
        if not os.path.exists(self.config_path):
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            videos = config.get('videos', {})
            for prefix in videos:
                if prefix in self.video_name:
                    crop_params = videos[prefix].get('crop_params', {})
                    self.crop_params = {
                        'left': crop_params.get('left', 0),
                        'right': crop_params.get('right', 0),
                        'top': crop_params.get('top', 0),
                        'bottom': crop_params.get('bottom', 0)
                    }
                    
                    # Update sliders
                    self.left_var.set(self.crop_params['left'])
                    self.right_var.set(self.crop_params['right'])
                    self.top_var.set(self.crop_params['top'])
                    self.bottom_var.set(self.crop_params['bottom'])
                    
                    self.crop_status_var.set(f"✓ Loaded crop settings for {prefix}")
                    self.crop_status_label.config(foreground="green")
                    self.on_crop_change(None)
                    return
        except Exception as e:
            print(f"Error loading crop params: {e}")
    
    def set_video_size(self, width, height):
        """Legacy method - no longer used"""
        pass

    def display_frame(self, frame_number):
        """Display a single frame with crop guides on canvas and preview"""
        if self.cap is None:
            return
        self.current_frame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Draw crop guides for adjustment canvas only
        h, w = frame_rgb.shape[:2]
        left = self.crop_params['left']
        right = self.crop_params['right']
        top = self.crop_params['top']
        bottom = self.crop_params['bottom']
        # Draw lines for adjustment
        cv2.line(frame_rgb, (left, 0), (left, h), (255, 0, 0), 2)
        cv2.line(frame_rgb, (w - right, 0), (w - right, h), (255, 0, 0), 2)
        cv2.line(frame_rgb, (0, top), (w, top), (255, 0, 0), 2)
        cv2.line(frame_rgb, (0, h - bottom), (w, h - bottom), (255, 0, 0), 2)
        self._display_on_canvas(frame_rgb, self.video_label)
        # Cropped output (no lines) — use clean frame without drawn lines
        cropped = frame_rgb[top:h-bottom, left:w-right]
        self._display_preview(cropped)
        self._display_border_patches(frame_rgb)
        # Update time/frame info
        current_time = self.frames_to_time(self.current_frame)
        total_time = self.frames_to_time(self.total_frames)
        self.time_label.config(text=f"{current_time} / {total_time}")
        self.frame_info_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}")

    def _display_on_canvas(self, frame_rgb, canvas):
        """Display frame on canvas, scaling proportionally to fit canvas size"""
        h, w = frame_rgb.shape[:2]
        target_w, target_h = self.video_canvas_width, self.video_canvas_height
        
        # Calculate scaling to fit within canvas while maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if scale < 1:
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        else:
            new_w, new_h = w, h
        
        # Convert to PhotoImage
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(img)
        
        # Display on canvas - centered
        canvas.delete("all")
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas.create_image(x_offset, y_offset, anchor="nw", image=photo)
        canvas.image = photo
    
    def _display_preview(self, cropped_frame):
        """Display cropped preview; canvas resizes to match crop output resolution (scaled to fit max)."""
        h, w = cropped_frame.shape[:2]
        if w == 0 or h == 0:
            return
        # Scale to fit within max preview size while maintaining aspect ratio
        scale = min(self.max_preview_w / w, self.max_preview_h / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        if new_w != w or new_h != h:
            cropped_frame = cv2.resize(cropped_frame, (new_w, new_h))
        # Resize canvas to the actual displayed image dimensions
        self.preview_label.config(width=new_w, height=new_h)
        self.preview_res_label.config(text=f"Cropped output: {w}x{h}  (displayed at {new_w}x{new_h})")
        img = Image.fromarray(cropped_frame)
        photo = ImageTk.PhotoImage(img)
        self.preview_label.delete("all")
        self.preview_label.create_image(0, 0, anchor="nw", image=photo)
        self.preview_label.image = photo

    def _display_border_patches(self, frame_rgb):
        """Display zoomed-in 20x20 px patches centred on the midpoint of each crop border line."""
        h, w = frame_rgb.shape[:2]
        left   = self.crop_params['left']
        right  = self.crop_params['right']
        top    = self.crop_params['top']
        bottom = self.crop_params['bottom']
        patch_px   = 20
        display_px = 200
        half = patch_px // 2

        def _show_patch(canvas, cx, cy):
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(w, cx + half)
            y1 = min(h, cy + half)
            if x1 <= x0 or y1 <= y0:
                canvas.delete("all")
                return
            patch = frame_rgb[y0:y1, x0:x1]
            ph, pw = patch.shape[:2]
            if ph != patch_px or pw != patch_px:
                padded = np.zeros((patch_px, patch_px, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded
            zoomed = cv2.resize(patch, (display_px, display_px), interpolation=cv2.INTER_NEAREST)
            img   = Image.fromarray(zoomed)
            photo = ImageTk.PhotoImage(img)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.image = photo

        _show_patch(self.patch_left,   left,         h // 2)
        _show_patch(self.patch_right,  w - right,    h // 2)
        _show_patch(self.patch_top,    w // 2,       top)
        _show_patch(self.patch_bottom, w // 2,       h - bottom)

    def on_slider_change(self, value):
        """Handle frame slider change"""
        frame_number = int(float(value))
        self.display_frame(frame_number)
    
    def on_crop_change(self, value):
        """Handle crop parameter change"""
        self.crop_params['left'] = self.left_var.get()
        self.crop_params['right'] = self.right_var.get()
        self.crop_params['top'] = self.top_var.get()
        self.crop_params['bottom'] = self.bottom_var.get()
        
        # Update labels
        self.left_label.config(text=str(self.crop_params['left']))
        self.right_label.config(text=str(self.crop_params['right']))
        self.top_label.config(text=str(self.crop_params['top']))
        self.bottom_label.config(text=str(self.crop_params['bottom']))
        
        # Update resolution info
        self.original_res_label.config(
            text=f"Original: {self.original_width}x{self.original_height}"
        )
        
        new_width = self.original_width - self.crop_params['left'] - self.crop_params['right']
        new_height = self.original_height - self.crop_params['top'] - self.crop_params['bottom']
        
        self.new_res_label.config(
            text=f"Cropped: {new_width}x{new_height}"
        )
        
        # Update display
        self.display_frame(self.current_frame)
        
        # Update status
        self.crop_status_var.set("Modified - click 'Save Crop Settings' to persist")
        self.crop_status_label.config(foreground="orange")
    
    def on_left_arrow(self, event):
        """Handle left arrow key - move slider left by 5 frames"""
        new_frame = max(0, self.current_frame - 5)
        self.frame_slider.set(new_frame)
    
    def on_right_arrow(self, event):
        """Handle right arrow key - move slider right by 5 frames"""
        new_frame = min(self.total_frames - 1, self.current_frame + 5)
        self.frame_slider.set(new_frame)
    
    def on_ctrl_left_arrow(self, event):
        """Handle Ctrl+left arrow - move slider left by 30 frames"""
        new_frame = max(0, self.current_frame - 30)
        self.frame_slider.set(new_frame)
    
    def on_ctrl_right_arrow(self, event):
        """Handle Ctrl+right arrow - move slider right by 30 frames"""
        new_frame = min(self.total_frames - 1, self.current_frame + 30)
        self.frame_slider.set(new_frame)
    
    def save_crop_params(self):
        """Save crop parameters to JSON config file"""
        if self.cap is None:
            messagebox.showerror("Error", "No video loaded")
            return
        
        # Load existing config
        try:
            with open(self.config_path, 'r') as f:
                all_configs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_configs = {}
        
        if 'videos' not in all_configs:
            all_configs['videos'] = {}
        
        # Save crop params for this video prefix
        all_configs['videos'][self.video_name] = {
            "crop_params": self.crop_params,
            "original_resolution": {
                "width": self.original_width,
                "height": self.original_height
            },
            "new_resolution": {
                "width": self.original_width - self.crop_params['left'] - self.crop_params['right'],
                "height": self.original_height - self.crop_params['top'] - self.crop_params['bottom']
            },
            "video_prefix": self.video_name
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(all_configs, f, indent=4)
            
            self.crop_status_var.set(f"✓ Crop settings saved for {self.video_name}")
            self.crop_status_label.config(foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save crop settings: {e}")
    
    def start_split_video(self):
        """Start video splitting in a separate thread"""
        if self.cap is None:
            messagebox.showerror("Error", "No video loaded")
            return
        
        try:
            clip_duration = float(self.duration_var.get())
            if clip_duration <= 0:
                messagebox.showerror("Error", "Clip duration must be positive")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid clip duration")
            return
        
        # Parse optional frame rate
        frame_rate = None
        if self.frame_rate_var.get().strip():
            try:
                frame_rate = float(self.frame_rate_var.get())
                if frame_rate <= 0:
                    messagebox.showerror("Error", "Frame rate must be positive")
                    return
            except ValueError:
                messagebox.showerror("Error", "Invalid frame rate")
                return
        
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showerror("Error", "Please specify output directory")
            return
        
        # Reset stop flag and update UI
        self.stop_processing = False
        self.split_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.crop_progress['value'] = 0
        self.split_progress['value'] = 0
        self.crop_progress_label.config(text="0%")
        self.split_progress_label.config(text="0%")
        
        # Start split in separate thread
        num_workers = max(1, self.workers_var.get())
        self.processing_thread = threading.Thread(
            target=self.split_and_crop_video,
            args=(self.video_path, clip_duration, output_dir, frame_rate, num_workers),
            daemon=True
        )
        self.processing_thread.start()
    
    def stop_split_video(self):
        """Stop video processing"""
        self.stop_processing = True
        self.stop_btn.config(state=tk.DISABLED)
        self.split_status_var.set("Stopping...")
        # Cancel any pending futures
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.update()
    
    def split_and_crop_video(self, video_path, clip_duration, output_dir, out_frame_rate=None, num_workers=1):
        """Split and crop video (runs in separate thread) - Split first, then apply crop to each clip"""
        try:
            print(f"\n{'='*60}")
            print(f"Split and Crop Video Operation Started")
            print(f"Video: {video_path}")
            print(f"Clip Duration: {clip_duration}s")
            print(f"Output Directory: {output_dir}")
            print(f"Output Frame Rate: {out_frame_rate if out_frame_rate else 'native'}")
            print(f"Workers: {num_workers}")
            print(f"{'='*60}\n")
            
            self.split_status_var.set("Processing... This may take a while")
            self.split_status_label.config(foreground="blue")
            self.root.update()
            
            if self.stop_processing:
                self.split_status_var.set("Cancelled")
                return
            
            # Split the video into clips (and apply crop during the process)
            self.split_status_var.set("Splitting and cropping video...")
            self.root.update()
            
            self.split_video_with_crop(video_path, clip_duration, output_dir, out_frame_rate, num_workers)
            
            if self.stop_processing:
                self.split_status_var.set("Cancelled")
                self.split_status_label.config(foreground="orange")
                print("Operation cancelled by user")
            else:
                self.split_status_var.set(f"✓ Successfully created clips in {output_dir}")
                self.split_status_label.config(foreground="green")
                print(f"\n✓ COMPLETED: Successfully created clips in {output_dir}\n")
                messagebox.showinfo("Success", f"Video split successfully!\n\nOutput directory: {output_dir}")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.split_status_var.set(f"✗ {error_msg}")
            self.split_status_label.config(foreground="red")
            print(f"\n✗ FAILED: {error_msg}")
            print(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to split video: {str(e)}\n\nCheck the terminal for detailed error information.")
        
        finally:
            # Re-enable buttons
            self.split_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def split_video_with_crop(self, video_path, clip_duration, output_dir, out_frame_rate=None, num_workers=1):
        """Split video into clips with crop applied, using multiprocessing for parallel encoding."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Probe video metadata only (fast – no full decode)
        probe = VideoFileClip(str(video_path))
        total_duration = probe.duration
        native_fps = probe.fps
        probe.close()

        fps = out_frame_rate if out_frame_rate is not None else native_fps
        base_name = video_path.stem
        crop_params = dict(self.crop_params)

        # Calculate clip ranges
        num_clips = int(total_duration / clip_duration)
        if total_duration % clip_duration > 0:
            num_clips += 1

        print(f"\n{'='*60}")
        print(f"Starting video split with crop  (workers={num_workers})")
        print(f"Video: {video_path}")
        print(f"Duration: {total_duration:.2f}s, FPS: {fps}, Total clips: {num_clips}")
        print(f"Crop params: L={crop_params['left']}, R={crop_params['right']}, "
              f"T={crop_params['top']}, B={crop_params['bottom']}")
        print(f"Output dir: {output_dir}")
        print(f"{'='*60}\n")

        # Reset progress bars
        self.crop_progress['value'] = 0
        self.split_progress['value'] = 0

        # Build task list
        tasks = []
        for i in range(num_clips):
            start_time = i * clip_duration
            end_time = min((i + 1) * clip_duration, total_duration)
            start_sec = int(round(start_time))
            end_sec   = int(round(end_time))
            clip_filename = f"{base_name}_{start_sec:04d}_{end_sec:04d}_sec.mp4"
            clip_path = output_dir / clip_filename
            tasks.append((str(video_path), start_time, end_time,
                          str(clip_path), crop_params,
                          fps if out_frame_rate is not None else None))

        completed = 0
        failed = []

        self._executor = ProcessPoolExecutor(max_workers=num_workers)
        future_to_idx = {self._executor.submit(_process_clip_worker, t): i
                         for i, t in enumerate(tasks)}
        self._futures = list(future_to_idx.keys())

        try:
            for future in as_completed(future_to_idx):
                if self.stop_processing:
                    break

                idx = future_to_idx[future]
                clip_filename = Path(tasks[idx][3]).name
                try:
                    success, clip_path_out, err_tb = future.result()
                except Exception as exc:
                    success, err_tb = False, traceback.format_exc()
                    clip_path_out = tasks[idx][3]

                completed += 1
                progress = (completed / num_clips) * 100
                self.split_progress['value'] = progress
                self.split_progress_label.config(text=f"{progress:.0f}%")

                if success:
                    print(f"[{completed}/{num_clips}] Done: {clip_filename}")
                    self.split_status_var.set(f"[{completed}/{num_clips}] Done: {clip_filename}")
                else:
                    print(f"[{completed}/{num_clips}] FAILED: {clip_filename}\n{err_tb}")
                    failed.append(clip_filename)
                    self.split_status_var.set(f"[{completed}/{num_clips}] FAILED: {clip_filename}")

                self.root.update()
        finally:
            self._executor.shutdown(wait=False)
            self._executor = None
            self._futures = []

        if failed:
            raise RuntimeError(f"{len(failed)} clip(s) failed: {', '.join(failed)}")

        if not self.stop_processing:
            self.split_progress['value'] = 100
            self.split_progress_label.config(text="100%")

    def crop_clip_manually(self, clip, left, top, right, bottom):
        """
        Manually crop a VideoFileClip using frame-by-frame processing.
        This is a fallback when moviepy's crop function is not available.
        """
        try:
            print(f"Using manual crop method (OpenCV-based)")
            w, h = clip.size
            new_w = w - left - right
            new_h = h - top - bottom
            
            # Create a custom clip that crops each frame
            def crop_frame(frame):
                return frame[top:h-bottom, left:w-right]
            
            # Apply the frame transformation using moviepy v2 VideoClip
            from moviepy import VideoClip
            make_frame = lambda t: crop_frame(clip.get_frame(t))
            cropped_clip = VideoClip(make_frame, duration=clip.duration).with_fps(clip.fps)
            
            return cropped_clip
        except Exception as e:
            print(f"Error in manual crop: {e}")
            print(traceback.format_exc())
            # If even manual crop fails, return original clip
            print("WARNING: Returning uncropped clip")
            return clip

def main():
    """Main entry point"""
    root = tk.Tk()
    app = VideoCropAndSplitApp(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # required for Windows multiprocessing
    main()
