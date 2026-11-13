"""
source/ui/main_window.py
Giao diện người dùng Desktop hoàn chỉnh cho ứng dụng Cải thiện ảnh thiếu sáng.
Phụ trách: TV1 (Lead & Tích hợp)

Các tính năng nổi bật:
    1. Tải ảnh đa định dạng (JPG, PNG, BMP, WEBP, ảnh xám, ảnh alpha).
    2. Tùy chỉnh tham số thuật toán theo thời gian thực (Real-time slider).
    3. Tab 1: So sánh ảnh song song Trước / Sau (Before - After) kèm nút bấm giữ so sánh tức thì.
    4. Tab 2: Nhúng biểu đồ Histogram trực tiếp (Matplotlib) so sánh dải mức xám Y của ảnh gốc vs ảnh sau xử lý.
    5. Tab 3: Chế độ so sánh đồng thời 3 thuật toán (Original, Gamma, HE, CLAHE).
    6. Đo lường các chỉ số định lượng: Độ sáng TB (Mean), Độ tương phản (Std), Thời gian thực thi (ms).
"""
import os
import tempfile

# Tránh cảnh báo quyền thư mục tạm của matplotlib trên Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from source.core.io import load_image, save_image
from source.core.pipeline import enhance
from source.core.color import rgb_to_ycbcr, ycbcr_to_rgb
from source.evaluation.benchmark import compare_with_opencv


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HỆ THỐNG CẢI THIỆN ẢNH THIẾU SÁNG - XỬ LÝ ẢNH SỐ (PTIT)")
        self.geometry("1180x820")
        self.minsize(980, 680)

        # Trạng thái dữ liệu
        self.input_image_path: Path | None = None
        self.original_image: np.ndarray | None = None
        self.enhanced_image: np.ndarray | None = None
        self.last_pipeline_result: dict | None = None

        # Trạng thái ảnh so sánh đồng thời (Tab 3)
        self.multi_compare_images: dict = {}
        self.zoom_size_tab3: int = 280

        # Trạng thái ảnh đối chứng OpenCV (Tab 4)
        self.opencv_compare_images: dict = {}
        self.zoom_size_tab4: int = 460

        # Trạng thái kích thước hiển thị ảnh Tab 1
        self.zoom_size: int = 460

        self._init_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Dọn dẹp tài nguyên matplotlib và đóng app dứt điểm."""
        try:
            plt.close("all")
        except Exception:
            pass
        self.destroy()

    def _init_style(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

    def _build_ui(self):
        # 1. Header
        header = ttk.Frame(self, padding=(15, 10))
        header.pack(fill=tk.X)

        title = ttk.Label(
            header,
            text="HỆ THỐNG CẢI THIỆN CHẤT LƯỢNG ẢNH THIẾU SÁNG",
            font=("Segoe UI", 16, "bold"),
            foreground="#1a365d"
        )
        title.pack(side=tk.LEFT)

        subtitle = ttk.Label(
            header,
            text="(Cài đặt From-scratch bằng NumPy - Bảo toàn màu sắc YCbCr)",
            font=("Segoe UI", 10, "italic"),
            foreground="#4a5568"
        )
        subtitle.pack(side=tk.LEFT, padx=12, pady=5)

        # 2. Control Toolbar
        ctrl_frame = ttk.LabelFrame(self, text=" BẢNG ĐIỀU KHIỂN & THAM SỐ ", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=15, pady=4)

        # Hàng nút tác vụ
        action_box = ttk.Frame(ctrl_frame)
        action_box.pack(fill=tk.X)

        self.btn_load = ttk.Button(action_box, text="📂 Mở ảnh...", command=self._on_load_image)
        self.btn_load.pack(side=tk.LEFT, padx=4)

        self.btn_save = ttk.Button(action_box, text="💾 Lưu ảnh kết quả", command=self._on_save_image, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        ttk.Separator(action_box, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Chọn thuật toán
        ttk.Label(action_box, text="Thuật toán:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=4)
        self.algo_var = tk.StringVar(value="gamma")
        self.combo_algo = ttk.Combobox(
            action_box,
            textvariable=self.algo_var,
            values=["gamma", "he", "clahe"],
            state="readonly",
            width=10
        )
        self.combo_algo.pack(side=tk.LEFT, padx=4)
        self.combo_algo.bind("<<ComboboxSelected>>", self._on_algo_changed)

        # Khung chứa slider động
        self.param_box = ttk.Frame(action_box)
        self.param_box.pack(side=tk.LEFT, padx=12)

        # Slider Gamma
        self.gamma_lbl = ttk.Label(self.param_box, text="Gamma (γ): 0.60")
        self.gamma_scale = ttk.Scale(
            self.param_box,
            from_=0.10,
            to=2.00,
            value=0.60,
            command=self._on_gamma_slider
        )
        self.gamma_lbl.pack(side=tk.LEFT, padx=4)
        self.gamma_scale.pack(side=tk.LEFT, padx=4)

        # Slider CLAHE (ẩn mặc định)
        self.clip_lbl = ttk.Label(self.param_box, text="Clip Factor: 2.0")
        self.clip_scale = ttk.Scale(
            self.param_box,
            from_=1.0,
            to=5.0,
            value=2.0,
            command=self._on_clip_slider
        )

        # Checkbox cập nhật tức thì (Realtime)
        self.realtime_var = tk.BooleanVar(value=True)
        self.chk_realtime = ttk.Checkbutton(
            action_box,
            text="⚡ Cập nhật tức thì (Real-time)",
            variable=self.realtime_var
        )
        self.chk_realtime.pack(side=tk.LEFT, padx=8)

        # Nút thực thi thủ công
        self.btn_run = ttk.Button(
            action_box,
            text="▶ Xử lý ảnh",
            command=self._on_process,
            state=tk.DISABLED
        )
        self.btn_run.pack(side=tk.RIGHT, padx=6)

        # 3. Main Workspace Notebook (Tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)

        # Tab 1: So sánh trực tiếp Before / After
        self.tab_compare = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_compare, text=" 🖼️ So sánh trực tiếp (Before / After) ")
        self._build_tab_compare()

        # Tab 2: Phân tích Lược đồ Histogram
        self.tab_hist = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_hist, text=" 📊 Phân tích Lược đồ (Histograms) ")
        self._build_tab_hist()

        # Tab 3: So sánh đồng thời bộ 3 thuật toán
        self.tab_multi = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_multi, text=" 🔍 So sánh cả 3 thuật toán (Overview) ")
        self._build_tab_multi()

        # Tab 4: Đối chứng trực tiếp với OpenCV
        self.tab_opencv = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_opencv, text=" ⚖️ Đối chứng OpenCV (Benchmark) ")
        self._build_tab_opencv()

        # 4. Status Bar
        self.status_var = tk.StringVar(value="Sẵn sàng. Vui lòng mở ảnh chụp thiếu sáng.")
        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(10, 4),
            font=("Segoe UI", 9)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # -------------------------------------------------------------
    # BUILD TAB 1: SO SÁNH TRỰC TIẾP BEFORE / AFTER
    # -------------------------------------------------------------
    def _build_tab_compare(self):
        panel = ttk.Frame(self.tab_compare, padding=8)
        panel.pack(fill=tk.BOTH, expand=True)

        # Hàng nút tương tác xem Before / After
        top_bar = ttk.Frame(panel)
        top_bar.pack(fill=tk.X, pady=(0, 6))

        self.btn_hold_compare = ttk.Button(top_bar, text="👁️ Nhấn giữ chuột để xem ảnh gốc (Before / After)")
        self.btn_hold_compare.pack(side=tk.LEFT, padx=4)
        self.btn_hold_compare.bind("<ButtonPress-1>", self._on_peek_original_press)
        self.btn_hold_compare.bind("<ButtonRelease-1>", self._on_peek_original_release)

        self.lbl_compare_hint = ttk.Label(
            top_bar,
            text="Mẹo: Nhấn giữ nút bên trái để so sánh trực tiếp trước/sau.",
            font=("Segoe UI", 9, "italic"),
            foreground="#555"
        )
        self.lbl_compare_hint.pack(side=tk.LEFT, padx=8)

        # Điều khiển kích thước hiển thị ảnh
        zoom_frame = ttk.Frame(top_bar)
        zoom_frame.pack(side=tk.RIGHT, padx=4)

        ttk.Label(zoom_frame, text="🔍 Cỡ hiển thị:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=3)
        self.zoom_lbl = ttk.Label(zoom_frame, text="460 px", width=7)
        self.zoom_lbl.pack(side=tk.LEFT, padx=2)

        self.zoom_scale = ttk.Scale(
            zoom_frame,
            from_=200,
            to=900,
            value=460,
            command=self._on_zoom_change
        )
        self.zoom_scale.pack(side=tk.LEFT, padx=4)

        self.btn_zoom_fit = ttk.Button(zoom_frame, text="Mặc định", width=8, command=self._on_zoom_reset)
        self.btn_zoom_fit.pack(side=tk.LEFT, padx=2)

        # Khung hiển thị 2 ảnh
        img_container = ttk.Frame(panel)
        img_container.pack(fill=tk.BOTH, expand=True)

        # Cột ảnh gốc
        self.left_box = ttk.LabelFrame(img_container, text=" ẢNH GỐC (ORIGINAL) ", padding=6)
        self.left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.lbl_orig_img = ttk.Label(self.left_box, text="Chưa tải ảnh", anchor="center")
        self.lbl_orig_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_orig_metrics = ttk.Label(
            self.left_box,
            text="Kích thước: -- | Độ sáng TB: -- | Độ tương phản: --",
            font=("Segoe UI", 9),
            anchor="center"
        )
        self.lbl_orig_metrics.pack(fill=tk.X, pady=4)

        # Cột ảnh kết quả
        self.right_box = ttk.LabelFrame(img_container, text=" ẢNH SAU TĂNG CƯỜNG (ENHANCED) ", padding=6)
        self.right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4)

        self.lbl_enh_img = ttk.Label(self.right_box, text="Chưa có kết quả xử lý", anchor="center")
        self.lbl_enh_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_enh_metrics = ttk.Label(
            self.right_box,
            text="Thuật toán: -- | Thời gian: -- ms | Độ sáng TB: -- | Độ tương phản: --",
            font=("Segoe UI", 9, "bold"),
            anchor="center",
            foreground="#1b4d3e"
        )
        self.lbl_enh_metrics.pack(fill=tk.X, pady=4)

    # -------------------------------------------------------------
    # BUILD TAB 2: BIỂU ĐỒ HISTOGRAM MATPLOTLIB
    # -------------------------------------------------------------
    def _build_tab_hist(self):
        panel = ttk.Frame(self.tab_hist, padding=8)
        panel.pack(fill=tk.BOTH, expand=True)

        # Tạo figure Matplotlib
        self.fig, (self.ax_orig, self.ax_enh) = plt.subplots(
            nrows=1, ncols=2, figsize=(9.5, 4.5), dpi=95, constrained_layout=True
        )
        self.fig.patch.set_facecolor("#f7fafc")

        # Thiết lập ban đầu cho 2 biểu đồ
        self._init_empty_hist_axes()

        # Nhúng vào Tkinter canvas
        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=panel)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _init_empty_hist_axes(self):
        for ax, title, color in [
            (self.ax_orig, "Lược đồ mức xám Y - Ảnh gốc (Original)", "#4a5568"),
            (self.ax_enh, "Lược đồ mức xám Y - Sau tăng cường (Enhanced)", "#2b6cb0")
        ]:
            ax.clear()
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel("Mức xám [0 .. 255]", fontsize=9)
            ax.set_ylabel("Tần suất pixel", fontsize=9)
            ax.set_xlim([0, 255])
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_facecolor("#ffffff")
        self.fig.canvas.draw_idle()

    def _render_histograms(self, y_orig: np.ndarray, y_enh: np.ndarray):
        """Vẽ biểu đồ histogram 256 mức cho kênh Y của ảnh gốc và ảnh sau xử lý."""
        hist_orig = np.bincount(y_orig.ravel(), minlength=256)
        hist_enh = np.bincount(y_enh.ravel(), minlength=256)

        bins = np.arange(256)
        mean_orig = float(np.mean(y_orig))
        mean_enh = float(np.mean(y_enh))

        # 1. Vẽ Histogram ảnh gốc
        self.ax_orig.clear()
        self.ax_orig.bar(bins, hist_orig, width=1.0, color="#718096", edgecolor="none", alpha=0.85)
        self.ax_orig.axvline(mean_orig, color="#e53e3e", linestyle="--", linewidth=1.5,
                             label=f"TB (Mean): {mean_orig:.1f}")
        self.ax_orig.set_title("Lược đồ Y - Ảnh gốc (Tập trung vùng tối)", fontsize=11, fontweight="bold")
        self.ax_orig.set_xlabel("Mức xám [0 .. 255]", fontsize=9)
        self.ax_orig.set_ylabel("Số lượng pixel", fontsize=9)
        self.ax_orig.set_xlim([0, 255])
        self.ax_orig.legend(loc="upper right", fontsize=9)
        self.ax_orig.grid(True, linestyle="--", alpha=0.4)
        self.ax_orig.set_facecolor("#ffffff")

        # 2. Vẽ Histogram ảnh sau tăng cường
        self.ax_enh.clear()
        self.ax_enh.bar(bins, hist_enh, width=1.0, color="#3182ce", edgecolor="none", alpha=0.85)
        self.ax_enh.axvline(mean_enh, color="#e53e3e", linestyle="--", linewidth=1.5,
                            label=f"TB (Mean): {mean_enh:.1f}")
        self.ax_enh.set_title("Lược đồ Y - Sau tăng cường (Dải sáng kéo giãn)", fontsize=11, fontweight="bold")
        self.ax_enh.set_xlabel("Mức xám [0 .. 255]", fontsize=9)
        self.ax_enh.set_ylabel("Số lượng pixel", fontsize=9)
        self.ax_enh.set_xlim([0, 255])
        self.ax_enh.legend(loc="upper right", fontsize=9)
        self.ax_enh.grid(True, linestyle="--", alpha=0.4)
        self.ax_enh.set_facecolor("#ffffff")

        self.hist_canvas.draw()

    # -------------------------------------------------------------
    # BUILD TAB 3: SO SÁNH ĐỒNG THỜI BỘ 3 THUẬT TOÁN
    # -------------------------------------------------------------
    def _build_tab_multi(self):
        panel = ttk.Frame(self.tab_multi, padding=8)
        panel.pack(fill=tk.BOTH, expand=True)

        top_ctrl = ttk.Frame(panel)
        top_ctrl.pack(fill=tk.X, pady=(0, 6))

        self.btn_run_all = ttk.Button(
            top_ctrl,
            text="⚡ Chạy đối sánh đồng thời cả 3 thuật toán (Gamma, HE, CLAHE)",
            command=self._on_run_all_algorithms,
            state=tk.DISABLED
        )
        self.btn_run_all.pack(side=tk.LEFT, padx=4)

        ttk.Label(
            top_ctrl,
            text="(4 khung so sánh trực tiếp)",
            font=("Segoe UI", 9, "italic"),
            foreground="#555"
        ).pack(side=tk.LEFT, padx=6)

        # Điều khiển kích thước hiển thị ảnh Tab 3
        zoom_frame_tab3 = ttk.Frame(top_ctrl)
        zoom_frame_tab3.pack(side=tk.RIGHT, padx=4)

        ttk.Label(zoom_frame_tab3, text="🔍 Cỡ hiển thị:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=3)
        self.zoom_lbl_tab3 = ttk.Label(zoom_frame_tab3, text="280 px", width=7)
        self.zoom_lbl_tab3.pack(side=tk.LEFT, padx=2)

        self.zoom_scale_tab3 = ttk.Scale(
            zoom_frame_tab3,
            from_=160,
            to=600,
            value=280,
            command=self._on_zoom_change_tab3
        )
        self.zoom_scale_tab3.pack(side=tk.LEFT, padx=4)

        self.btn_zoom_fit_tab3 = ttk.Button(zoom_frame_tab3, text="Mặc định", width=8, command=self._on_zoom_reset_tab3)
        self.btn_zoom_fit_tab3.pack(side=tk.LEFT, padx=2)

        # Vùng cuộn (Scrollable Canvas) cho Tab 3
        canvas_container = ttk.Frame(panel)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas_tab3 = tk.Canvas(canvas_container, highlightthickness=0)
        v_scrollbar_tab3 = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas_tab3.yview)
        self.canvas_tab3.configure(yscrollcommand=v_scrollbar_tab3.set)

        v_scrollbar_tab3.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_tab3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        grid_frame = ttk.Frame(self.canvas_tab3)
        self.canvas_window_tab3 = self.canvas_tab3.create_window((0, 0), window=grid_frame, anchor="nw")

        def _on_tab3_frame_configure(event=None):
            req_h = grid_frame.winfo_reqheight()
            canvas_h = self.canvas_tab3.winfo_height()
            final_h = max(req_h, canvas_h)
            self.canvas_tab3.itemconfig(self.canvas_window_tab3, height=final_h)
            self.canvas_tab3.configure(scrollregion=(0, 0, self.canvas_tab3.winfo_width(), req_h))

        def _on_tab3_canvas_configure(event):
            # Luôn lấp đầy chiều rộng của màn hình
            self.canvas_tab3.itemconfig(self.canvas_window_tab3, width=event.width)
            req_h = grid_frame.winfo_reqheight()
            final_h = max(req_h, event.height)
            self.canvas_tab3.itemconfig(self.canvas_window_tab3, height=final_h)
            self.canvas_tab3.configure(scrollregion=(0, 0, event.width, req_h))

        grid_frame.bind("<Configure>", _on_tab3_frame_configure)
        self.canvas_tab3.bind("<Configure>", _on_tab3_canvas_configure)

        def _on_mousewheel(event):
            try:
                # Đơn vị cuộn chuẩn trên Windows
                delta = int(-1 * (event.delta / 120))
                current_tab = self.notebook.index(self.notebook.select())
                if current_tab == 2:
                    self.canvas_tab3.yview_scroll(delta, "units")
                elif current_tab == 3:
                    self.canvas_tab4.yview_scroll(delta, "units")
            except Exception:
                pass

        self.bind_all("<MouseWheel>", _on_mousewheel)

        # 4 ô lưới 2x2
        self.multi_boxes = {}
        box_titles = [
            ("orig", "1. Ảnh gốc (Original)"),
            ("gamma", "2. Gamma Correction (TV1)"),
            ("he", "3. Global Histogram Equalization (TV2)"),
            ("clahe", "4. CLAHE (TV3)")
        ]

        for idx, (key, title) in enumerate(box_titles):
            row = idx // 2
            col = idx % 2
            box = ttk.LabelFrame(grid_frame, text=f" {title} ", padding=4)
            box.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            grid_frame.rowconfigure(row, weight=1)
            grid_frame.columnconfigure(col, weight=1)

            lbl_img = ttk.Label(box, text="Chưa có dữ liệu", anchor="center")
            lbl_img.pack(fill=tk.BOTH, expand=True)

            lbl_info = ttk.Label(box, text="", font=("Segoe UI", 8), anchor="center")
            lbl_info.pack(fill=tk.X, pady=2)

            self.multi_boxes[key] = {"lbl_img": lbl_img, "lbl_info": lbl_info}

    # -------------------------------------------------------------
    # BUILD TAB 4: ĐỐI CHỨNG TRỰC TIẾP VỚI THƯ VIỆN OPENCV
    # -------------------------------------------------------------
    def _build_tab_opencv(self):
        panel = ttk.Frame(self.tab_opencv, padding=8)
        panel.pack(fill=tk.BOTH, expand=True)

        top_ctrl = ttk.Frame(panel)
        top_ctrl.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(top_ctrl, text="Chọn thuật toán đối chứng:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=4)
        self.cv_algo_var = tk.StringVar(value="gamma")
        self.combo_cv_algo = ttk.Combobox(
            top_ctrl,
            textvariable=self.cv_algo_var,
            values=["gamma", "he", "clahe"],
            state="readonly",
            width=10
        )
        self.combo_cv_algo.pack(side=tk.LEFT, padx=4)

        self.btn_run_opencv_compare = ttk.Button(
            top_ctrl,
            text="⚡ Chạy đối chứng với OpenCV",
            command=self._on_run_opencv_comparison,
            state=tk.DISABLED
        )
        self.btn_run_opencv_compare.pack(side=tk.LEFT, padx=8)

        ttk.Label(
            top_ctrl,
            text="(So sánh pixel-by-pixel)",
            font=("Segoe UI", 9, "italic"),
            foreground="#555"
        ).pack(side=tk.LEFT, padx=4)

        # Điều khiển kích thước hiển thị ảnh Tab 4
        zoom_frame_tab4 = ttk.Frame(top_ctrl)
        zoom_frame_tab4.pack(side=tk.RIGHT, padx=4)

        ttk.Label(zoom_frame_tab4, text="🔍 Cỡ hiển thị:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=3)
        self.zoom_lbl_tab4 = ttk.Label(zoom_frame_tab4, text="460 px", width=7)
        self.zoom_lbl_tab4.pack(side=tk.LEFT, padx=2)

        self.zoom_scale_tab4 = ttk.Scale(
            zoom_frame_tab4,
            from_=200,
            to=900,
            value=460,
            command=self._on_zoom_change_tab4
        )
        self.zoom_scale_tab4.pack(side=tk.LEFT, padx=4)

        self.btn_zoom_fit_tab4 = ttk.Button(zoom_frame_tab4, text="Mặc định", width=8, command=self._on_zoom_reset_tab4)
        self.btn_zoom_fit_tab4.pack(side=tk.LEFT, padx=2)

        # Vùng cuộn (Scrollable Canvas) cho Tab 4
        canvas_container_tab4 = ttk.Frame(panel)
        canvas_container_tab4.pack(fill=tk.BOTH, expand=True)

        self.canvas_tab4 = tk.Canvas(canvas_container_tab4, highlightthickness=0)
        v_scrollbar_tab4 = ttk.Scrollbar(canvas_container_tab4, orient=tk.VERTICAL, command=self.canvas_tab4.yview)
        self.canvas_tab4.configure(yscrollcommand=v_scrollbar_tab4.set)

        v_scrollbar_tab4.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_tab4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        container = ttk.Frame(self.canvas_tab4)
        self.canvas_window_tab4 = self.canvas_tab4.create_window((0, 0), window=container, anchor="nw")

        def _on_tab4_frame_configure(event=None):
            req_h = container.winfo_reqheight()
            canvas_h = self.canvas_tab4.winfo_height()
            final_h = max(req_h, canvas_h)
            self.canvas_tab4.itemconfig(self.canvas_window_tab4, height=final_h)
            self.canvas_tab4.configure(scrollregion=(0, 0, self.canvas_tab4.winfo_width(), req_h))

        def _on_tab4_canvas_configure(event):
            # Luôn lấp đầy chiều rộng của màn hình
            self.canvas_tab4.itemconfig(self.canvas_window_tab4, width=event.width)
            req_h = container.winfo_reqheight()
            final_h = max(req_h, event.height)
            self.canvas_tab4.itemconfig(self.canvas_window_tab4, height=final_h)
            self.canvas_tab4.configure(scrollregion=(0, 0, event.width, req_h))

        container.bind("<Configure>", _on_tab4_frame_configure)
        self.canvas_tab4.bind("<Configure>", _on_tab4_canvas_configure)

        # Cột trái: Nhóm tự cài đặt (NumPy)
        self.cv_left_box = ttk.LabelFrame(container, text=" 1. NHÓM TỰ CÀI ĐẶT (NUMPY FROM-SCRATCH) ", padding=6)
        self.cv_left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.lbl_cv_our_img = ttk.Label(self.cv_left_box, text="Chưa chạy đối chứng", anchor="center")
        self.lbl_cv_our_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_cv_our_metrics = ttk.Label(
            self.cv_left_box,
            text="Thời gian: -- ms | Độ sáng TB: -- | Độ tương phản: --",
            font=("Segoe UI", 9, "bold"),
            anchor="center",
            foreground="#1b4d3e"
        )
        self.lbl_cv_our_metrics.pack(fill=tk.X, pady=4)

        # Cột phải: OpenCV thư viện chuẩn
        self.cv_right_box = ttk.LabelFrame(container, text=" 2. THƯ VIỆN OPENCV (CV2 STANDARD) ", padding=6)
        self.cv_right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4)

        self.lbl_cv_lib_img = ttk.Label(self.cv_right_box, text="Chưa chạy đối chứng", anchor="center")
        self.lbl_cv_lib_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_cv_lib_metrics = ttk.Label(
            self.cv_right_box,
            text="Thời gian: -- ms | Độ sáng TB: -- | Độ tương phản: --",
            font=("Segoe UI", 9, "bold"),
            anchor="center",
            foreground="#2b6cb0"
        )
        self.lbl_cv_lib_metrics.pack(fill=tk.X, pady=4)

        # Khung kết luận định lượng ở đáy Tab 4
        self.lbl_cv_conclusion = ttk.Label(
            panel,
            text="Bấm 'Chạy đối chứng với OpenCV' để phân tích sai số định lượng (MAE, chênh lệch tốc độ).",
            font=("Segoe UI", 10, "bold"),
            relief=tk.GROOVE,
            padding=6,
            anchor="center",
            foreground="#1a365d"
        )
        self.lbl_cv_conclusion.pack(fill=tk.X, pady=(6, 0))

    # -------------------------------------------------------------
    # LOGIC XỬ LÝ SỰ KIỆN
    # -------------------------------------------------------------
    def _on_zoom_change(self, val):
        self.zoom_size = int(float(val))
        self.zoom_lbl.config(text=f"{self.zoom_size} px")
        self._refresh_tab1_display()

    def _on_zoom_reset(self):
        self.zoom_size = 460
        self.zoom_scale.set(460)
        self.zoom_lbl.config(text="460 px")
        self._refresh_tab1_display()

    def _refresh_tab1_display(self):
        """Vẽ lại ảnh gốc và ảnh kết quả trên Tab 1 theo zoom_size hiện thời."""
        if self.original_image is not None:
            self._display_image(self.original_image, self.lbl_orig_img, max_w=self.zoom_size, max_h=self.zoom_size)
        if self.enhanced_image is not None:
            self._display_image(self.enhanced_image, self.lbl_enh_img, max_w=self.zoom_size, max_h=self.zoom_size)

    def _on_zoom_change_tab3(self, val):
        self.zoom_size_tab3 = int(float(val))
        self.zoom_lbl_tab3.config(text=f"{self.zoom_size_tab3} px")
        self._refresh_tab3_display()

    def _on_zoom_reset_tab3(self):
        self.zoom_size_tab3 = 280
        self.zoom_scale_tab3.set(280)
        self.zoom_lbl_tab3.config(text="280 px")
        self._refresh_tab3_display()

    def _refresh_tab3_display(self):
        """Vẽ lại cả 4 ảnh trong Tab 3 theo zoom_size_tab3."""
        for key, img in self.multi_compare_images.items():
            if key in self.multi_boxes:
                lbl_img = self.multi_boxes[key]["lbl_img"]
                self._display_image(img, lbl_img, max_w=self.zoom_size_tab3, max_h=int(self.zoom_size_tab3 * 0.8))
        self.update_idletasks()
        if hasattr(self, "canvas_tab3"):
            req_h = self.canvas_tab3.winfo_children()[0].winfo_reqheight() if self.canvas_tab3.winfo_children() else 0
            final_h = max(req_h, self.canvas_tab3.winfo_height())
            self.canvas_tab3.itemconfig(self.canvas_window_tab3, height=final_h)
            self.canvas_tab3.configure(scrollregion=(0, 0, self.canvas_tab3.winfo_width(), req_h))

    def _on_zoom_change_tab4(self, val):
        self.zoom_size_tab4 = int(float(val))
        self.zoom_lbl_tab4.config(text=f"{self.zoom_size_tab4} px")
        self._refresh_tab4_display()

    def _on_zoom_reset_tab4(self):
        self.zoom_size_tab4 = 460
        self.zoom_scale_tab4.set(460)
        self.zoom_lbl_tab4.config(text="460 px")
        self._refresh_tab4_display()

    def _refresh_tab4_display(self):
        """Vẽ lại 2 ảnh trong Tab 4 theo zoom_size_tab4."""
        if "our" in self.opencv_compare_images:
            self._display_image(self.opencv_compare_images["our"], self.lbl_cv_our_img, max_w=self.zoom_size_tab4, max_h=self.zoom_size_tab4)
        if "cv" in self.opencv_compare_images:
            self._display_image(self.opencv_compare_images["cv"], self.lbl_cv_lib_img, max_w=self.zoom_size_tab4, max_h=self.zoom_size_tab4)
        self.update_idletasks()
        if hasattr(self, "canvas_tab4"):
            req_h = self.canvas_tab4.winfo_children()[0].winfo_reqheight() if self.canvas_tab4.winfo_children() else 0
            final_h = max(req_h, self.canvas_tab4.winfo_height())
            self.canvas_tab4.itemconfig(self.canvas_window_tab4, height=final_h)
            self.canvas_tab4.configure(scrollregion=(0, 0, self.canvas_tab4.winfo_width(), req_h))

    def _on_algo_changed(self, event=None):
        method = self.algo_var.get()
        if method == "gamma":
            self.clip_lbl.pack_forget()
            self.clip_scale.pack_forget()
            self.gamma_lbl.pack(side=tk.LEFT, padx=4)
            self.gamma_scale.pack(side=tk.LEFT, padx=4)
        elif method == "clahe":
            self.gamma_lbl.pack_forget()
            self.gamma_scale.pack_forget()
            self.clip_lbl.pack(side=tk.LEFT, padx=4)
            self.clip_scale.pack(side=tk.LEFT, padx=4)
        else:  # he
            self.gamma_lbl.pack_forget()
            self.gamma_scale.pack_forget()
            self.clip_lbl.pack_forget()
            self.clip_scale.pack_forget()

        if self.realtime_var.get() and self.original_image is not None:
            self._on_process()

    def _on_gamma_slider(self, val):
        self.gamma_lbl.config(text=f"Gamma (γ): {float(val):.2f}")
        if self.realtime_var.get() and self.original_image is not None:
            self._on_process()

    def _on_clip_slider(self, val):
        self.clip_lbl.config(text=f"Clip Factor: {float(val):.1f}")
        if self.realtime_var.get() and self.original_image is not None:
            self._on_process()

    def _on_load_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
        chosen = filedialog.askopenfilename(title="Chọn ảnh thiếu sáng", filetypes=filetypes)
        if not chosen:
            return

        try:
            path = Path(chosen)
            self.original_image = load_image(path)
            self.input_image_path = path

            # Hiển thị ảnh gốc trên Tab 1
            self._display_image(self.original_image, self.lbl_orig_img, max_w=self.zoom_size, max_h=self.zoom_size)

            H, W = self.original_image.shape[:2]
            mode = "RGB" if self.original_image.ndim == 3 else "Grayscale"
            mean_b = float(np.mean(self.original_image))
            std_b = float(np.std(self.original_image))

            self.lbl_orig_metrics.config(
                text=f"{path.name} ({W}x{H}, {mode}) | Độ sáng TB: {mean_b:.1f} | Độ tương phản: {std_b:.1f}"
            )

            # Cập nhật ô gốc trong Tab 3
            self.multi_compare_images["orig"] = self.original_image
            self._display_image(self.original_image, self.multi_boxes["orig"]["lbl_img"], max_w=self.zoom_size_tab3, max_h=int(self.zoom_size_tab3 * 0.8))
            self.multi_boxes["orig"]["lbl_info"].config(text=f"Độ sáng TB: {mean_b:.1f} | Std: {std_b:.1f}")

            self.btn_run.config(state=tk.NORMAL)
            self.btn_run_all.config(state=tk.NORMAL)
            self.btn_run_opencv_compare.config(state=tk.NORMAL)
            self.status_var.set(f"Đã mở: {path.name} ({W}x{H}). Đang thực thi thuật toán...")

            # Tự động chạy thuật toán được chọn
            self._on_process()

        except Exception as e:
            messagebox.showerror("Lỗi mở ảnh", f"Không thể tải ảnh: {str(e)}")

    def _on_process(self):
        if self.original_image is None:
            return

        method = self.algo_var.get()
        params = {}
        if method == "gamma":
            params["gamma"] = float(self.gamma_scale.get())
        elif method == "clahe":
            params["clip_factor"] = float(self.clip_scale.get())
            params["grid"] = (8, 8)

        try:
            res = enhance(self.original_image, method=method, params=params)
            self.last_pipeline_result = res
            self.enhanced_image = res["image"]

            # Hiển thị ảnh sau tăng cường trên Tab 1
            self._display_image(self.enhanced_image, self.lbl_enh_img, max_w=self.zoom_size, max_h=self.zoom_size)

            t_ms = res["algorithm_ms"]
            mean_enh = float(np.mean(self.enhanced_image))
            std_enh = float(np.std(self.enhanced_image))
            is_mock_str = " [MOCK - Chờ TV]" if res.get("is_mock") else ""

            self.lbl_enh_metrics.config(
                text=f"Thuật toán: {method.upper()}{is_mock_str} | Thời gian: {t_ms:.2f} ms | "
                     f"Độ sáng TB: {mean_enh:.1f} | Độ tương phản: {std_enh:.1f}"
            )

            # Cập nhật Tab 2: Vẽ lại biểu đồ Histogram
            y_orig = res["y_original"]
            y_result = res["y_result"]
            self._render_histograms(y_orig, y_result)

            self.btn_save.config(state=tk.NORMAL)
            if res.get("is_mock"):
                self.status_var.set(f"[{method.upper()}] đang dùng hàm giả lập (mock). Cần đợi TV phụ trách bàn giao code.")
            else:
                self.status_var.set(f"Đã xử lý xong bằng [{method.upper()}] trong {t_ms:.2f} ms!")

        except Exception as e:
            messagebox.showerror("Lỗi thuật toán", f"Quá trình xử lý thất bại: {str(e)}")
            self.status_var.set("Lỗi thực thi thuật toán.")

    def _on_peek_original_press(self, event=None):
        """Khi nhấn giữ chuột, tạm thời hiển thị ảnh gốc ở khung kết quả."""
        if self.original_image is not None and self.enhanced_image is not None:
            self._display_image(self.original_image, self.lbl_enh_img, max_w=self.zoom_size, max_h=self.zoom_size)
            self.lbl_enh_metrics.config(text="[ĐANG HIỂN THỊ ẢNH GỐC ĐỂ SO SÁNH]")

    def _on_peek_original_release(self, event=None):
        """Khi nhả chuột, hiển thị lại ảnh sau xử lý."""
        if self.enhanced_image is not None and self.last_pipeline_result is not None:
            self._display_image(self.enhanced_image, self.lbl_enh_img, max_w=self.zoom_size, max_h=self.zoom_size)
            res = self.last_pipeline_result
            t_ms = res["algorithm_ms"]
            mean_enh = float(np.mean(self.enhanced_image))
            std_enh = float(np.std(self.enhanced_image))
            is_mock_str = " [MOCK - Chờ TV]" if res.get("is_mock") else ""
            self.lbl_enh_metrics.config(
                text=f"Thuật toán: {res['method'].upper()}{is_mock_str} | Thời gian: {t_ms:.2f} ms | "
                     f"Độ sáng TB: {mean_enh:.1f} | Độ tương phản: {std_enh:.1f}"
            )

    def _on_run_all_algorithms(self):
        """Chạy cả 3 thuật toán và hiển thị so sánh tại Tab 3."""
        if self.original_image is None:
            return

        self.status_var.set("Đang chạy đối sánh cả 3 thuật toán...")
        self.update_idletasks()

        gamma_val = float(self.gamma_scale.get())

        configs = [
            ("gamma", {"gamma": gamma_val}),
            ("he", {}),
            ("clahe", {"grid": (8, 8), "clip_factor": 2.0})
        ]

        for method, params in configs:
            res = enhance(self.original_image, method=method, params=params)
            img = res["image"]
            t_ms = res["algorithm_ms"]
            mean_v = float(np.mean(img))
            std_v = float(np.std(img))
            is_mock = res.get("is_mock", False)

            lbl_img = self.multi_boxes[method]["lbl_img"]
            lbl_info = self.multi_boxes[method]["lbl_info"]

            self.multi_compare_images[method] = img
            self._display_image(img, lbl_img, max_w=self.zoom_size_tab3, max_h=int(self.zoom_size_tab3 * 0.8))
            mock_note = " (MOCK)" if is_mock else ""
            lbl_info.config(
                text=f"Thời gian: {t_ms:.2f} ms{mock_note} | TB: {mean_v:.1f} | Std: {std_v:.1f}"
            )

        self.status_var.set("Đã hoàn thành so sánh đồng thời 3 thuật toán!")
        self.notebook.select(self.tab_multi)

    def _on_run_opencv_comparison(self):
        """Chạy đối chứng thuật toán được chọn với thư viện OpenCV và cập nhật Tab 4."""
        if self.original_image is None:
            return

        method = self.cv_algo_var.get()
        self.status_var.set(f"Đang chạy đối chứng [{method.upper()}] với OpenCV...")
        self.update_idletasks()

        is_color = (self.original_image.ndim == 3 and self.original_image.shape[2] == 3)
        if is_color:
            y_orig, cb, cr = rgb_to_ycbcr(self.original_image)
        else:
            y_orig = self.original_image.copy()

        params = {}
        if method == "gamma":
            params = {"gamma": float(self.gamma_scale.get())}
        elif method == "clahe":
            params = {
                "grid": (8, 8),
                "clip_factor": float(self.clip_scale.get())
            }

        try:
            res = compare_with_opencv(y_orig, method=method, params=params)
            our_y = res["our_result"]
            cv_y = res["cv_result"]
            mae = res["mae"]
            our_ms = res["our_ms"]
            cv_ms = res["cv_ms"]

            # Khôi phục ảnh RGB nếu đầu vào là ảnh màu
            if is_color:
                our_img = ycbcr_to_rgb(our_y, cb, cr)
                cv_img = ycbcr_to_rgb(cv_y, cb, cr)
            else:
                our_img = our_y
                cv_img = cv_y

            # Hiển thị 2 ảnh trên Tab 4
            self.opencv_compare_images["our"] = our_img
            self.opencv_compare_images["cv"] = cv_img
            self._display_image(our_img, self.lbl_cv_our_img, max_w=self.zoom_size_tab4, max_h=self.zoom_size_tab4)
            self._display_image(cv_img, self.lbl_cv_lib_img, max_w=self.zoom_size_tab4, max_h=self.zoom_size_tab4)

            # Cập nhật thông số
            our_mean = float(np.mean(our_img))
            our_std = float(np.std(our_img))
            cv_mean = float(np.mean(cv_img))
            cv_std = float(np.std(cv_img))

            self.lbl_cv_our_metrics.config(
                text=f"Thời gian: {our_ms:.2f} ms | Độ sáng TB: {our_mean:.1f} | Độ tương phản: {our_std:.1f}"
            )
            self.lbl_cv_lib_metrics.config(
                text=f"Thời gian: {cv_ms:.2f} ms | Độ sáng TB: {cv_mean:.1f} | Độ tương phản: {cv_std:.1f}"
            )

            # Đánh giá kết luận
            if mae < 1e-4:
                eval_text = f"Độ khớp tuyệt đối: MAE = {mae:.4f} (Trùng khớp 100% với OpenCV!)"
                eval_color = "#22543d"
            else:
                eval_text = f"Sai số trung bình: MAE = {mae:.4f} mức xám | Chênh lệch tốc độ: Nhóm {our_ms:.1f}ms vs CV {cv_ms:.1f}ms"
                eval_color = "#1a365d"

            self.lbl_cv_conclusion.config(
                text=f"Thuật toán [{method.upper()}]: {eval_text}",
                foreground=eval_color
            )

            self.status_var.set(f"Đã hoàn thành đối chứng [{method.upper()}] với OpenCV! MAE = {mae:.4f}")
            self.notebook.select(self.tab_opencv)

        except Exception as e:
            messagebox.showerror("Lỗi đối chứng", f"Quá trình đối chứng với OpenCV thất bại: {str(e)}")
            self.status_var.set("Lỗi thực thi đối chứng OpenCV.")

    def _on_save_image(self):
        if self.enhanced_image is None:
            return

        method = self.algo_var.get()
        orig_name = self.input_image_path.name if self.input_image_path else "out.png"
        init_name = f"enhanced_{method}_{orig_name}"

        save_path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            initialfile=init_name,
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
        )
        if not save_path:
            return

        try:
            save_image(self.enhanced_image, save_path)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh kết quả thành công:\n{save_path}")
            self.status_var.set(f"Đã lưu ảnh: {Path(save_path).name}")
        except Exception as e:
            messagebox.showerror("Lỗi lưu ảnh", f"Không thể lưu file: {str(e)}")

    def _display_image(self, img_arr: np.ndarray, label: ttk.Label, max_w: int = 460, max_h: int = 460):
        """Thay đổi kích thước tỷ lệ ảnh và vẽ lên label Tkinter."""
        h, w = img_arr.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        pil_img = Image.fromarray(img_arr)
        if (new_w, new_h) != (w, h):
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(pil_img)
        label.config(image=photo, text="")
        label.image = photo


def start_app():
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    start_app()
