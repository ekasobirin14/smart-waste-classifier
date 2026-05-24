import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from datetime import datetime
import time

# ======================================
# APPEARANCE
# ======================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ======================================
# LOAD MODEL
# ======================================

model = load_model("model/keras_model.h5", compile=False)

class_names = open(
    "model/labels.txt",
    "r"
).readlines()

# ======================================
# SETTINGS
# ======================================

CONFIDENCE_THRESHOLD = 0.90

# delay scan setelah tombol reset
SCAN_DELAY_SECONDS = 2

# ======================================
# CLASS COLORS
# ======================================

class_colors = {
    "plastic": "#00BFFF",
    "paper": "#F5C542",
    "glass": "#00FFD1",
    "metal": "#C0C0C0",
    "cardboard": "#C68642",
    "organic": "#39FF14",
    "ewaste": "#B026FF",
    "battery": "#FF3131"
}

# ======================================
# VARIABLES
# ======================================

detected_class = "WAITING..."
detected_confidence = 0

scan_active = True

# timer delay scan
scan_start_time = time.time()

# history list
detection_history = []

# ======================================
# MAIN APP
# ======================================

app = ctk.CTk()

app.title("Smart Waste Classifier")

app.geometry("1700x950")

app.minsize(1300, 750)

# ======================================
# MAIN FRAME
# ======================================

main_frame = ctk.CTkFrame(
    app,
    fg_color="#0A0F1F",
    corner_radius=20,
    border_width=2,
    border_color="#00BFFF"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=25
)

# ======================================
# HEADER
# ======================================

header_frame = ctk.CTkFrame(
    main_frame,
    fg_color="transparent"
)

header_frame.pack(
    fill="x",
    padx=25,
    pady=(10, 5)
)

title_label = ctk.CTkLabel(
    header_frame,
    text="SMART WASTE CLASSIFIER",
    font=("Orbitron", 40, "bold"),
    text_color="#00E5FF"
)

title_label.pack()

name_label = ctk.CTkLabel(
    header_frame,
    text="Muhammad Eka Sobirin",
    font=("Arial", 20, "bold"),
    text_color="white"
)

name_label.pack(pady=(5, 0))

nim_label = ctk.CTkLabel(
    header_frame,
    text="NIM : 3.34.23.2.14",
    font=("Arial", 16),
    text_color="#8FA8C9"
)

nim_label.pack()

# ======================================
# CONTENT FRAME
# ======================================

content_frame = ctk.CTkFrame(
    main_frame,
    fg_color="transparent"
)

content_frame.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)

# responsive layout
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=4)
content_frame.grid_columnconfigure(2, weight=2)

content_frame.grid_rowconfigure(0, weight=1)

# ======================================
# LEFT PANEL - HISTORY
# ======================================

history_panel = ctk.CTkFrame(
    content_frame,
    fg_color="#111827",
    corner_radius=20,
    border_width=2,
    border_color="#00BFFF"
)

history_panel.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 10)
)

history_title = ctk.CTkLabel(
    history_panel,
    text="DETECTION\nHISTORY",
    font=("Arial", 24, "bold"),
    text_color="#00E5FF"
)

history_title.pack(pady=(20, 15))

history_textbox = ctk.CTkTextbox(
    history_panel,
    width=250,
    fg_color="#0F172A",
    text_color="white",
    font=("Consolas", 14),
    corner_radius=15
)

history_textbox.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=(0, 20)
)

history_textbox.insert(
    "end",
    "No Detection History...\n"
)

history_textbox.configure(state="disabled")

# ======================================
# CENTER PANEL - CAMERA
# ======================================

camera_panel = ctk.CTkFrame(
    content_frame,
    fg_color="#111827",
    corner_radius=20,
    border_width=2,
    border_color="#00BFFF"
)

camera_panel.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(0, 10)
)

camera_panel.grid_rowconfigure(1, weight=1)
camera_panel.grid_columnconfigure(0, weight=1)

# ======================================
# CAMERA TITLE
# ======================================

camera_title = ctk.CTkLabel(
    camera_panel,
    text="LIVE SCANNING CAMERA",
    font=("Arial", 30, "bold"),
    text_color="#00E5FF"
)

camera_title.grid(
    row=0,
    column=0,
    pady=(20, 10)
)

# ======================================
# VIDEO LABEL
# ======================================

video_label = ctk.CTkLabel(
    camera_panel,
    text=""
)

video_label.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=20,
    pady=10
)

# ======================================
# BUTTON FRAME
# ======================================

button_frame = ctk.CTkFrame(
    camera_panel,
    fg_color="transparent"
)

button_frame.grid(
    row=2,
    column=0,
    pady=(5, 20)
)

# ======================================
# RESET FUNCTION
# ======================================

def reset_detection():

    global scan_active
    global detected_class
    global detected_confidence
    global scan_start_time

    detected_class = "WAITING..."
    detected_confidence = 0

    result_label.configure(
        text="WAITING...",
        text_color="#39FF14"
    )

    confidence_label.configure(
        text="Confidence : 0%"
    )

    confidence_bar.set(0)

    scan_status_label.configure(
        text="PREPARING SCAN...",
        text_color="#00E5FF"
    )

    bin_label.configure(image=None)

    # reset color
    history_panel.configure(
        border_color="#00BFFF"
    )

    camera_panel.configure(
        border_color="#00BFFF"
    )

    result_panel.configure(
        border_color="#00BFFF"
    )

    main_frame.configure(
        border_color="#00BFFF"
    )

    # aktif scan
    scan_active = True

    # reset timer
    scan_start_time = time.time()

# ======================================
# RESET BUTTON
# ======================================

reset_button = ctk.CTkButton(
    button_frame,
    text="DETEKSI ULANG",
    width=280,
    height=55,
    font=("Arial", 20, "bold"),
    fg_color="#00BFFF",
    hover_color="#0099CC",
    corner_radius=15,
    command=reset_detection
)

reset_button.pack()

# ======================================
# RIGHT PANEL - RESULT
# ======================================

result_panel = ctk.CTkFrame(
    content_frame,
    fg_color="#111827",
    corner_radius=20,
    border_width=2,
    border_color="#00BFFF"
)

result_panel.grid(
    row=0,
    column=2,
    sticky="nsew"
)

# ======================================
# RESULT TITLE
# ======================================

info_title = ctk.CTkLabel(
    result_panel,
    text="DETECTION RESULT",
    font=("Arial", 30, "bold"),
    text_color="#00E5FF"
)

info_title.pack(
    pady=(25, 20)
)

# ======================================
# RESULT LABEL
# ======================================

result_label = ctk.CTkLabel(
    result_panel,
    text="WAITING...",
    font=("Arial", 38, "bold"),
    text_color="#39FF14"
)

result_label.pack(pady=10)

# ======================================
# CONFIDENCE LABEL
# ======================================

confidence_label = ctk.CTkLabel(
    result_panel,
    text="Confidence : 0%",
    font=("Arial", 22),
    text_color="white"
)

confidence_label.pack(pady=(5, 10))

# ======================================
# PROGRESS BAR
# ======================================

confidence_bar = ctk.CTkProgressBar(
    result_panel,
    width=320,
    height=20,
    corner_radius=10,
    progress_color="#00E5FF"
)

confidence_bar.pack(pady=(0, 25))

confidence_bar.set(0)

# ======================================
# STATUS LABEL
# ======================================

scan_status_label = ctk.CTkLabel(
    result_panel,
    text="SCANNING OBJECT...",
    font=("Arial", 20, "bold"),
    text_color="#00E5FF"
)

scan_status_label.pack(pady=10)

# ======================================
# BIN IMAGE
# ======================================

bin_label = ctk.CTkLabel(
    result_panel,
    text=""
)

bin_label.pack(
    pady=20
)

# ======================================
# FOOTER
# ======================================

footer_label = ctk.CTkLabel(
    result_panel,
    text="AI Recyclable Waste Detection System",
    font=("Arial", 14),
    text_color="#8FA8C9"
)

footer_label.pack(
    side="bottom",
    pady=20
)

# ======================================
# CAMERA
# ======================================

camera = cv2.VideoCapture(0)

# ======================================
# SCAN ANIMATION
# ======================================

scan_line_position = 0
scan_direction = 1

# ======================================
# UPDATE HISTORY
# ======================================

def add_history(text):

    history_textbox.configure(state="normal")

    # hapus tulisan awal
    current_text = history_textbox.get(
        "1.0",
        "end"
    ).strip()

    if current_text == "No Detection History...":
        history_textbox.delete("1.0", "end")

    history_textbox.insert(
        "1.0",
        text + "\n"
    )

    history_textbox.configure(state="disabled")

# ======================================
# UPDATE FRAME
# ======================================

def update_frame():

    global scan_line_position
    global scan_direction
    global detected_class
    global detected_confidence
    global scan_active

    ret, frame = camera.read()

    if ret:

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # ======================================
        # DETECTION
        # ======================================

        if scan_active:

            # delay scan 2 detik
            elapsed_time = (
                time.time() - scan_start_time
            )

            if elapsed_time < SCAN_DELAY_SECONDS:

                remaining = int(
                    SCAN_DELAY_SECONDS - elapsed_time
                ) + 1

                scan_status_label.configure(
                    text=f"PREPARING CAMERA... {remaining}"
                )

            else:

                scan_status_label.configure(
                    text="SCANNING OBJECT...",
                    text_color="#00E5FF"
                )

                image_resized = cv2.resize(
                    frame,
                    (224, 224)
                )

                image_array = np.asarray(
                    image_resized
                )

                normalized_image_array = (
                    image_array.astype(np.float32)
                    / 127.5
                ) - 1

                data = np.ndarray(
                    shape=(1, 224, 224, 3),
                    dtype=np.float32
                )

                data[0] = normalized_image_array

                prediction = model.predict(
                    data,
                    verbose=0
                )

                index = np.argmax(prediction)

                class_name = class_names[index].strip()

                confidence_score = prediction[0][index]

                # ======================================
                # OBJECT DETECTED
                # ======================================

                if confidence_score > CONFIDENCE_THRESHOLD:

                    detected_class = (
                        class_name.split(" ")[1]
                    )

                    detected_confidence = confidence_score

                    color = class_colors.get(
                        detected_class,
                        "#00E5FF"
                    )

                    # result
                    result_label.configure(
                        text=detected_class.upper(),
                        text_color=color
                    )

                    # confidence
                    confidence_label.configure(
                        text=(
                            f"Confidence : "
                            f"{detected_confidence*100:.2f}%"
                        )
                    )

                    confidence_bar.set(
                        detected_confidence
                    )

                    confidence_bar.configure(
                        progress_color=color
                    )

                    # status
                    scan_status_label.configure(
                        text="OBJECT DETECTED",
                        text_color=color
                    )

                    # border color
                    history_panel.configure(
                        border_color=color
                    )

                    camera_panel.configure(
                        border_color=color
                    )

                    result_panel.configure(
                        border_color=color
                    )

                    main_frame.configure(
                        border_color=color
                    )

                    # ======================================
                    # SAVE HISTORY
                    # ======================================

                    current_time = datetime.now().strftime(
                        "%H:%M:%S"
                    )

                    history_text = (
                        f"[{current_time}] "
                        f"{detected_class.upper()} "
                        f"({detected_confidence*100:.1f}%)"
                    )

                    detection_history.append(
                        history_text
                    )

                    add_history(history_text)

                    # ======================================
                    # BIN IMAGE
                    # ======================================

                    try:

                        bin_image = Image.open(
                            f"../assets/bins/"
                            f"{detected_class}_bin.png"
                        )

                        bin_image = bin_image.resize(
                            (220, 220)
                        )

                        bin_photo = ImageTk.PhotoImage(
                            bin_image
                        )

                        bin_label.configure(
                            image=bin_photo
                        )

                        bin_label.image = bin_photo

                    except:
                        pass

                    # stop scan
                    scan_active = False

        # ======================================
        # SCAN ANIMATION
        # ======================================

        if scan_active:

            scan_line_position += (
                5 * scan_direction
            )

            if scan_line_position >= h - 20:
                scan_direction = -1

            if scan_line_position <= 20:
                scan_direction = 1

            cv2.line(
                frame,
                (0, scan_line_position),
                (w, scan_line_position),
                (0, 255, 255),
                2
            )

        # ======================================
        # CAMERA BORDER
        # ======================================

        cv2.rectangle(
            frame,
            (30, 30),
            (w - 30, h - 30),
            (0, 255, 255),
            2
        )

        # ======================================
        # CONVERT FRAME
        # ======================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(frame_rgb)

        # ======================================
        # RESPONSIVE CAMERA
        # ======================================

        label_width = video_label.winfo_width()
        label_height = video_label.winfo_height()

        if label_width > 50 and label_height > 50:

            img_width, img_height = img.size

            aspect_ratio = (
                img_width / img_height
            )

            new_width = label_width
            new_height = int(
                new_width / aspect_ratio
            )

            if new_height > label_height:

                new_height = label_height

                new_width = int(
                    new_height * aspect_ratio
                )

            img = img.resize(
                (new_width, new_height)
            )

        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk

        video_label.configure(image=imgtk)

    video_label.after(10, update_frame)

# ======================================
# START APP
# ======================================

update_frame()

app.mainloop()

camera.release()

cv2.destroyAllWindows()