"""
Interactive Video Annotation Application
Annotate surgical videos with anatomical phase labels and save to JSON
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
from datetime import datetime


class VideoAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotation Tool")
        self.root.geometry("1400x900")
        
        # Video variables
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame = 0
        self.video_path = None
        self.video_name = "unknown"
        self.video_display_size = (800, 600)  # Default size
        
        # Annotation variables
        self.annotations = {}
        self.phase_counter = 0
        self.last_marked_frame = 0
        
        # Phase labels
        self.PHASE_LABELS = ['other', 'normal','anterior', 'posterior', 'fissure', 'inferior', 'superior']
        
        # Phase colors
        self.PHASE_COLORS = {
            'other': "#615555",
            'normal': '#7FDBFF',
            'anterior': '#FF6B6B',
            'posterior': "#4ECD54",
            'fissure': '#FFE66D',
            'inferior': "#CB95E1",
            'superior': "#647FEE"
        }
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== VIDEO LOADING SECTION =====
        load_frame = ttk.LabelFrame(main_frame, text="Load Video", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(load_frame, text="Video File:").pack(side=tk.LEFT)
        self.video_path_var = tk.StringVar()
        ttk.Entry(load_frame, textvariable=self.video_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="Browse...", command=self.browse_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        
        self.load_status = tk.StringVar(value="No video loaded")
        ttk.Label(load_frame, textvariable=self.load_status, foreground="red").pack(side=tk.LEFT, padx=5)
        
        # ===== MAIN ANNOTATION SECTION =====
        annotation_frame = ttk.LabelFrame(main_frame, text="Video Annotation Interface", padding=10)
        annotation_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Video display area
        display_frame = ttk.Frame(annotation_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Video display controls
        display_control_frame = ttk.Frame(display_frame)
        display_control_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(display_control_frame, text="Video Size:").pack(side=tk.LEFT)
        ttk.Button(display_control_frame, text="Small", width=8, command=lambda: self.set_video_size(400, 300)).pack(side=tk.LEFT, padx=2)
        ttk.Button(display_control_frame, text="Medium", width=8, command=lambda: self.set_video_size(600, 450)).pack(side=tk.LEFT, padx=2)
        ttk.Button(display_control_frame, text="Large", width=8, command=lambda: self.set_video_size(800, 600)).pack(side=tk.LEFT, padx=2)
        ttk.Button(display_control_frame, text="Full", width=8, command=lambda: self.set_video_size(1000, 750)).pack(side=tk.LEFT, padx=2)
        
        self.video_label = ttk.Label(display_frame, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Controls below video
        controls_frame = ttk.Frame(annotation_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Left side: scrubber controls
        left_controls = ttk.LabelFrame(controls_frame, text="Video Controls", padding=5)
        left_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Frame slider
        slider_label_frame = ttk.Frame(left_controls)
        slider_label_frame.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(slider_label_frame, text="Frame:").pack(side=tk.LEFT)
        
        slider_frame = ttk.Frame(left_controls)
        slider_frame.pack(fill=tk.X, pady=(0, 5))
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.frame_slider.pack(fill=tk.X, expand=True, padx=5)
        
        # Timeline canvas
        timeline_label_frame = ttk.Frame(left_controls)
        timeline_label_frame.pack(fill=tk.X, pady=(5, 3))
        ttk.Label(timeline_label_frame, text="Timeline (click to jump, double-click to edit):").pack(side=tk.LEFT)
        
        timeline_frame = ttk.Frame(left_controls)
        timeline_frame.pack(fill=tk.X, pady=(0, 5))
        self.timeline_canvas = tk.Canvas(timeline_frame, height=30, bg="white", highlightthickness=1)
        self.timeline_canvas.pack(fill=tk.X, expand=True, padx=5)
        self.timeline_canvas.bind('<Button-1>', self.on_timeline_click)
        self.timeline_canvas.bind('<Double-Button-1>', self.on_timeline_double_click)
        
        # Time and frame info
        info_frame = ttk.Frame(left_controls)
        info_frame.pack(fill=tk.X, pady=5)
        self.time_label = ttk.Label(info_frame, text="Current Time: 00:00:00", font=("Arial", 10, "bold"))
        self.time_label.pack(side=tk.LEFT, padx=5)
        self.frame_info_label = ttk.Label(info_frame, text="Frame: 0 / 0")
        self.frame_info_label.pack(side=tk.LEFT, padx=5)
        
        # Right side: phase assignment
        right_controls = ttk.LabelFrame(controls_frame, text="Phase Assignment", padding=5)
        right_controls.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Phase dropdown
        dropdown_frame = ttk.Frame(right_controls)
        dropdown_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dropdown_frame, text="Phase:").pack(side=tk.LEFT)
        self.phase_var = tk.StringVar(value=self.PHASE_LABELS[0])
        phase_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.phase_var, 
                                      values=self.PHASE_LABELS, state="readonly", width=15)
        phase_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Mark button
        ttk.Button(right_controls, text="Mark Timestamp", command=self.mark_timestamp).pack(fill=tk.X, pady=5)
        
        # Status message
        self.mark_status_var = tk.StringVar(value="No timestamps marked yet")
        self.mark_status_label = ttk.Label(right_controls, textvariable=self.mark_status_var, foreground="blue")
        self.mark_status_label.pack(fill=tk.X, pady=5)
        
        # ===== SAVE SECTION =====
        save_frame = ttk.LabelFrame(main_frame, text="Save Annotations", padding=10)
        save_frame.pack(fill=tk.X)
        
        button_frame = ttk.Frame(save_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="💾 Save to JSON", command=self.save_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save to Custom Path...", command=self.save_custom).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh Display", command=self.update_annotations_display).pack(side=tk.LEFT, padx=5)
        
        self.save_status_var = tk.StringVar(value="")
        self.save_status_label = ttk.Label(save_frame, textvariable=self.save_status_var, foreground="green")
        self.save_status_label.pack(fill=tk.X, pady=5)
        
        # Bind keyboard events
        self.root.bind('<Left>', self.on_left_arrow)
        self.root.bind('<Right>', self.on_right_arrow)
        self.root.bind('<Control-Left>', self.on_ctrl_left_arrow)
        self.root.bind('<Control-Right>', self.on_ctrl_right_arrow)
        
    def browse_video(self):
        """Open file dialog to browse for video"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
    
    def frames_to_time(self, frame_num):
        """Convert frame number to HH:MM:SS format"""
        if self.fps == 0:
            return "00:00:00"
        seconds = frame_num / self.fps
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def load_existing_annotations(self):
        """Load existing annotations from view_annotation.json"""
        json_path = "view_annotation.json"
        
        if not os.path.exists(json_path):
            return False
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'videos' in data and self.video_name in data['videos']:
                    video_data = data['videos'][self.video_name]
                    self.annotations = video_data.get('phases', {})
                    self.phase_counter = len(self.annotations)
                    
                    # Find the last marked frame from annotations
                    if self.annotations:
                        last_phase = max(self.annotations.values(), key=lambda x: x['end_time'])
                        # Convert end_time back to frame number
                        end_parts = last_phase['end_time'].split(':')
                        end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])
                        self.last_marked_frame = int(end_seconds * self.fps)
                    
                    self.mark_status_var.set(f"✓ Loaded {self.phase_counter} existing phases")
                    self.mark_status_label.config(foreground="green")
                    return True
        except Exception as e:
            print(f"Error loading annotations: {e}")
        
        return False
    
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
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        
        # Extract video name
        self.video_name = Path(video_path).stem
        self.video_path = video_path
        
        # Reset annotations
        self.annotations = {}
        self.phase_counter = 0
        self.last_marked_frame = 0
        
        # Update slider
        self.frame_slider.configure(to=self.total_frames - 1 if self.total_frames > 0 else 100)
        
        # Update status
        self.load_status.set(f"✓ Video loaded: {Path(video_path).name}")
        
        # Try to load existing annotations
        if self.load_existing_annotations():
            self.load_status.set(f"✓ Video loaded: {Path(video_path).name} (+ existing annotations)")
        else:
            self.mark_status_var.set("No timestamps marked yet")
            self.mark_status_label.config(foreground="blue")
        
        # Display first frame
        self.display_frame(0)
        self.update_annotations_display()
        
    def set_video_size(self, width, height):
        """Set the video display size and redraw"""
        self.video_display_size = (width, height)
        self.display_frame(self.current_frame)
    
    def display_frame(self, frame_number):
        """Display a single frame"""
        if self.cap is None:
            return
        
        self.current_frame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to fit display size
            height, width = frame_rgb.shape[:2]
            max_width, max_height = self.video_display_size
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(img)
            
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            # Update time and frame info
            current_time = self.frames_to_time(self.current_frame)
            self.time_label.config(text=f"Current Time: {current_time}")
            self.frame_info_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}")
    
    def on_slider_change(self, value):
        """Handle slider change"""
        frame_number = int(float(value))
        self.display_frame(frame_number)
    
    def on_left_arrow(self, event):
        """Handle left arrow key - move slider left by 5 frame"""
        new_frame = max(0, self.current_frame - 5)
        self.frame_slider.set(new_frame)
    
    def on_right_arrow(self, event):
        """Handle right arrow key - move slider right by 5 frame"""
        new_frame = min(self.total_frames - 1, self.current_frame + 5)
        self.frame_slider.set(new_frame)
    
    def on_ctrl_left_arrow(self, event):
        """Handle Ctrl+left arrow - move slider left by 30 frames"""
        new_frame = max(0, self.current_frame - 30)
        self.frame_slider.set(new_frame)
    
    def on_ctrl_right_arrow(self, event):
        """Handle Ctrl+right arrow - move slider right by 10 frames"""
        new_frame = min(self.total_frames - 1, self.current_frame + 30)
        self.frame_slider.set(new_frame)
    
    def on_timeline_click(self, event):
        """Handle timeline canvas click to jump to frame"""
        if self.total_frames == 0:
            return
        
        canvas_width = self.timeline_canvas.winfo_width()
        if canvas_width <= 1:
            return
        
        # Calculate frame from click position
        fraction = event.x / canvas_width
        frame_num = int(fraction * self.total_frames)
        frame_num = max(0, min(self.total_frames - 1, frame_num))
        self.frame_slider.set(frame_num)
    
    def on_timeline_double_click(self, event):
        """Handle timeline double-click to edit phase"""
        if self.total_frames == 0 or not self.annotations:
            return
        
        canvas_width = self.timeline_canvas.winfo_width()
        if canvas_width <= 1:
            return
        
        # Find which phase was clicked
        click_fraction = event.x / canvas_width
        clicked_frame = int(click_fraction * self.total_frames)
        
        clicked_phase_key = None
        for phase_key in self.annotations.keys():
            phase_data = self.annotations[phase_key]
            # Convert times to frames
            start_time = phase_data['start_time']
            end_time = phase_data['end_time']
            
            start_parts = start_time.split(':')
            start_seconds = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
            start_frame = int(start_seconds * self.fps)
            
            end_parts = end_time.split(':')
            end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])
            end_frame = int(end_seconds * self.fps)
            
            if start_frame <= clicked_frame <= end_frame:
                clicked_phase_key = phase_key
                break
        
        if clicked_phase_key:
            self.open_edit_phase_dialog(clicked_phase_key)
    
    def open_edit_phase_dialog(self, phase_key):
        """Open a dialog to edit a phase"""
        phase_data = self.annotations[phase_key]
        
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {phase_key}")
        dialog.geometry("500x500")
        dialog.resizable(False, False)
        
        # Phase view selection
        view_frame = ttk.Frame(dialog, padding=15)
        view_frame.pack(fill=tk.X)
        ttk.Label(view_frame, text="Phase View:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        view_var = tk.StringVar(value=phase_data['view'])
        for label in self.PHASE_LABELS:
            ttk.Radiobutton(view_frame, text=label, variable=view_var, value=label).pack(anchor=tk.W, pady=3)
        
        # Start time
        start_frame = ttk.Frame(dialog, padding=15)
        start_frame.pack(fill=tk.X)
        ttk.Label(start_frame, text=f"Start Time: {phase_data['start_time']}", font=("Arial", 10)).pack(anchor=tk.W)
        
        # End time
        end_frame = ttk.Frame(dialog, padding=15)
        end_frame.pack(fill=tk.X)
        ttk.Label(end_frame, text=f"End Time: {phase_data['end_time']}", font=("Arial", 10)).pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(dialog, padding=5)
        button_frame.pack(fill=tk.X)
        
        def save_changes():
            phase_data['view'] = view_var.get()
            self.mark_status_var.set(f"✓ Updated {phase_key}: {phase_data['view']}")
            self.mark_status_label.config(foreground="green")
            self.update_annotations_display()
            dialog.destroy()
        
        def delete_phase():
            if messagebox.askyesno("Confirm Delete", f"Delete {phase_key}?"):
                del self.annotations[phase_key]
                # Renumber phases
                self.phase_counter = len(self.annotations)
                self.mark_status_var.set(f"✓ Deleted {phase_key}")
                self.mark_status_label.config(foreground="green")
                self.update_annotations_display()
                dialog.destroy()
        
        ttk.Button(button_frame, text="Save Changes", command=save_changes).pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Delete Phase", command=delete_phase).pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
    
    def draw_timeline(self):
        """Draw the timeline with phase annotations"""
        self.timeline_canvas.delete('all')
        
        if self.total_frames == 0 or not self.annotations:
            return
        
        canvas_width = self.timeline_canvas.winfo_width()
        canvas_height = self.timeline_canvas.winfo_height()
        
        if canvas_width <= 1:
            return
        
        # Draw each phase as a colored rectangle
        for phase_key in sorted(self.annotations.keys(), key=lambda x: int(x.split('_')[1])):
            phase_data = self.annotations[phase_key]
            view = phase_data['view']
            color = self.PHASE_COLORS.get(view, '#CCCCCC')
            
            # Convert times to frame numbers
            start_time = phase_data['start_time']
            end_time = phase_data['end_time']
            
            start_parts = start_time.split(':')
            start_seconds = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
            start_frame = int(start_seconds * self.fps)
            
            end_parts = end_time.split(':')
            end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])
            end_frame = int(end_seconds * self.fps)
            
            # Calculate positions
            x1 = (start_frame / self.total_frames) * canvas_width
            x2 = (end_frame / self.total_frames) * canvas_width
            
            # Draw rectangle
            self.timeline_canvas.create_rectangle(
                x1, 2, x2, canvas_height - 2,
                fill=color, outline="black", width=1
            )
            
            # Add label if space permits
            if x2 - x1 > 30:
                mid_x = (x1 + x2) / 2
                self.timeline_canvas.create_text(
                    mid_x, canvas_height / 2,
                    text=view[:3].upper(), font=("Arial", 8, "bold"),
                    fill="black"
                )
    
    def mark_timestamp(self):
        """Mark current timestamp with selected phase"""
        if self.cap is None:
            messagebox.showerror("Error", "No video loaded")
            return
        
        phase_label = self.phase_var.get()
        
        if self.assign_phase(self.current_frame, phase_label):
            current_time = self.frames_to_time(self.current_frame)
            self.mark_status_var.set(f"✓ Marked {phase_label} at {current_time}")
            self.mark_status_label.config(foreground="green")
            self.update_annotations_display()
        else:
            self.mark_status_var.set("✗ Failed to mark timestamp")
            self.mark_status_label.config(foreground="red")
    
    def assign_phase(self, frame_number, phase_label):
        """Assign phase from last marked frame to current frame"""
        if self.fps is None or self.fps == 0:
            return False
        
        self.phase_counter += 1
        start_frame = self.last_marked_frame
        end_frame = frame_number
        
        start_time = self.frames_to_time(start_frame)
        end_time = self.frames_to_time(end_frame)
        
        phase_key = f"phase_{self.phase_counter}"
        self.annotations[phase_key] = {
            "view": phase_label,
            "start_time": start_time,
            "end_time": end_time
        }
        
        self.last_marked_frame = frame_number
        return True
    
    def update_annotations_display(self):
        """Update the timeline visualization"""
        # Redraw timeline
        self.root.after(100, self.draw_timeline)
    
    def save_annotations_to_json(self, output_path):
        """Save annotations to JSON file, preserving existing video annotations"""
        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save")
            return False
        
        # Load existing data if file exists
        existing_data = {"videos": {}}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing annotations: {e}")
        
        # Ensure 'videos' key exists
        if 'videos' not in existing_data:
            existing_data['videos'] = {}
        
        # Update or append current video's annotations
        existing_data['videos'][self.video_name] = {
            "phases": self.annotations
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
            return False
    
    def save_default(self):
        """Save to default path"""
        if self.save_annotations_to_json("view_annotation.json"):
            self.save_status_var.set("✓ Saved to view_annotation.json")
            self.save_status_label.config(foreground="green")
            messagebox.showinfo("Success", "Annotations saved to view_annotation.json")
        else:
            self.save_status_var.set("✗ Save failed")
            self.save_status_label.config(foreground="red")
    
    def save_custom(self):
        """Save to custom path"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            if self.save_annotations_to_json(file_path):
                self.save_status_var.set(f"✓ Saved to {Path(file_path).name}")
                self.save_status_label.config(foreground="green")
                messagebox.showinfo("Success", f"Annotations saved to {file_path}")
            else:
                self.save_status_var.set("✗ Save failed")
                self.save_status_label.config(foreground="red")


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = VideoAnnotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
