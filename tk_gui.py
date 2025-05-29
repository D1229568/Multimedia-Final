import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import threading
import datetime
import numpy as np
import temp  # pastikan temp.py ada di folder yang sama

class FaceFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Face Filter App')
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.mode = 'loopback'
        self.running = True
        self.recording = False
        self.video_writer = None
        self.frame = None
        self.filter_label = tk.StringVar()
        self.mode_label = tk.StringVar()
        self.cap = cv2.VideoCapture(0)
        self.create_widgets()
        self.update_video()

    def create_widgets(self):
        # Video panel
        self.video_panel = tk.Label(self.root)
        self.video_panel.pack(padx=10, pady=10)

        # Baris 1: tombol mode
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=(0, 8))
        tk.Label(mode_frame, text='Mode:', font=('Arial', 10, 'bold')).pack(side='left', padx=(0, 8))
        tk.Button(mode_frame, text='Loopback', width=12, command=lambda: self.set_mode('loopback')).pack(side='left', padx=2)
        tk.Button(mode_frame, text='Filters', width=12, command=lambda: self.set_mode('filters')).pack(side='left', padx=2)
        tk.Button(mode_frame, text='Background', width=12, command=lambda: self.set_mode('background')).pack(side='left', padx=2)

        # Baris 2: tombol filter (horizontal, scrollable jika banyak)
        filter_scroll = tk.Frame(self.root)
        filter_scroll.pack(pady=(0, 8))
        canvas = tk.Canvas(filter_scroll, height=40)
        scrollbar = tk.Scrollbar(filter_scroll, orient='horizontal', command=canvas.xview)
        self.filter_buttons_frame = tk.Frame(canvas)
        self.filter_buttons = []
        for idx, fname in enumerate(temp.filter_types):
            btn = tk.Button(self.filter_buttons_frame, text=fname, width=12, command=lambda i=idx: self.select_filter(i))
            btn.grid(row=0, column=idx, padx=2, pady=2)
            self.filter_buttons.append(btn)
        # Tambahkan tombol khusus ironman jika belum ada
        if 'ironman' not in temp.filter_types:
            idx = len(self.filter_buttons)
            btn = tk.Button(self.filter_buttons_frame, text='ironman', width=12, command=lambda i=idx: self.select_filter(i))
            btn.grid(row=0, column=idx, padx=2, pady=2)
            self.filter_buttons.append(btn)
        self.filter_buttons_frame.update_idletasks()
        canvas.create_window((0, 0), window=self.filter_buttons_frame, anchor='nw')
        canvas.configure(xscrollcommand=scrollbar.set, scrollregion=canvas.bbox('all'))
        canvas.pack(side='top', fill='x', expand=True)
        scrollbar.pack(side='top', fill='x')
        self.update_filter_buttons()

        # Baris 3: tombol background (jika mode background)
        self.bg_buttons_frame = tk.Frame(self.root)
        self.bg_buttons = []
        for idx in range(len(temp.bg_images)):
            btn = tk.Button(self.bg_buttons_frame, text=f'BG {idx+1}', width=10, command=lambda i=idx: self.select_bg(i))
            btn.grid(row=0, column=idx, padx=2, pady=2)
            self.bg_buttons.append(btn)
        self.bg_buttons_frame.pack(pady=(0, 8))
        self.update_bg_buttons()

        # Baris 4: tombol Screenshot dan Record
        action_frame = tk.Frame(self.root)
        action_frame.pack(pady=(0, 8))
        tk.Button(action_frame, text='Screenshot', width=16, command=self.screenshot).pack(side='left', padx=8)
        tk.Button(action_frame, text='Record', width=16, command=self.toggle_record).pack(side='left', padx=8)

        # Status
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=(0, 4))
        tk.Label(status_frame, textvariable=self.mode_label, font=('Arial', 10, 'bold')).pack(side='left', padx=8)
        tk.Label(status_frame, textvariable=self.filter_label, font=('Arial', 10)).pack(side='left', padx=8)
        self.update_labels()

    def select_filter(self, idx):
        if self.mode == 'filters':
            temp.vf_mode = idx
            self.update_labels()
            self.update_filter_buttons()

    def update_filter_buttons(self):
        for idx, btn in enumerate(self.filter_buttons):
            if self.mode == 'filters':
                btn.config(state='normal', relief='sunken' if temp.vf_mode == idx else 'raised', bg='#d1e7dd' if temp.vf_mode == idx else 'SystemButtonFace')
            else:
                btn.config(state='disabled', relief='raised', bg='SystemButtonFace')

    def select_bg(self, idx):
        if self.mode == 'background':
            temp.bg_mode = idx
            self.update_bg_buttons()

    def update_bg_buttons(self):
        for idx, btn in enumerate(self.bg_buttons):
            if self.mode == 'background':
                btn.config(state='normal', relief='sunken' if temp.bg_mode == idx else 'raised', bg='#d1e7dd' if temp.bg_mode == idx else 'SystemButtonFace')
            else:
                btn.config(state='disabled', relief='raised', bg='SystemButtonFace')

    def set_mode(self, mode):
        self.mode = mode
        self.update_labels()
        self.update_filter_buttons()
        self.update_bg_buttons()

    def prev_filter(self):
        if self.mode == 'filters':
            temp.vf_mode = (temp.vf_mode - 1) % len(temp.filter_types)
            self.update_labels()

    def next_filter(self):
        if self.mode == 'filters':
            temp.vf_mode = (temp.vf_mode + 1) % len(temp.filter_types)
            self.update_labels()

    def screenshot(self):
        if self.frame is not None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'screenshot_{timestamp}.png'
            cv2.imwrite(filename, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
            messagebox.showinfo('Screenshot', f'Screenshot saved: {filename}')

    def toggle_record(self):
        if not self.recording:
            if self.frame is not None:
                h, w = self.frame.shape[:2]
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'recording_{timestamp}.avi'
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                self.recording = True
                messagebox.showinfo('Record', f'Recording started: {filename}')
        else:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            messagebox.showinfo('Record', 'Recording stopped and saved.')

    def update_labels(self):
        self.mode_label.set(f'Mode: {self.mode}')
        if self.mode == 'filters':
            self.filter_label.set(f'Filter: {temp.filter_types[temp.vf_mode]}')
        else:
            self.filter_label.set('')

    def update_video(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_video)
            return
        # === PIPELINE: Pastikan urutan sama dengan mode OpenCV window ===
        # 1. Flip frame (mirror)
        frame = cv2.flip(frame, 1)
        # 2. Proses filter di BGR (jangan convert ke RGB sebelum filter)
        if self.mode == 'loopback':
            out = temp.process_loopback(frame)
        elif self.mode == 'filters':
            out = temp.process_videofilters(frame)
        elif self.mode == 'smartboard':
            out = temp.process_smartboard(frame)
        elif self.mode == 'background':
            out = temp.process_background(frame)
        else:
            out = frame
        # 3. Konversi ke RGB hanya untuk display di Tkinter
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        self.frame = out_rgb.copy()
        img = Image.fromarray(out_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_panel.imgtk = imgtk
        self.video_panel.config(image=imgtk)
        if self.recording and self.video_writer is not None:
            self.video_writer.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        self.root.after(10, self.update_video)

    def on_close(self):
        self.running = False
        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = FaceFilterApp(root)
    root.mainloop()
