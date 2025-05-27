import sys
import os

# Redirect sys.stderr to a file if it's None
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')
import time
from tkinter.tix import Meter
import numpy as np
import torch
from PIL import Image, ImageTk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, Canvas, Frame, Scrollbar
from GaussianTrainer import GaussianTrainer
import threading

class GUITrainer:
    """
    ===============================================================
    GUITrainer: A Tkinter-based Interface for 3D Gaussian Training
    ===============================================================

    1. Purpose:
    -----------
    This file defines a GUI class (`GUITrainer`) that integrates with a 
    GaussianTrainer model to allow users to interactively:
        - Input prompts for 3D model generation
        - Start and stop training
        - View generated image previews in real-time
        - Export trained 3D models (geometry and texture)
    The GUI is built using ttkbootstrap for theming and enhanced styling.

    2. Functions Overview:
    ----------------------

    __init__(self, prompt, display_interval=10)
        - Initializes the GUI, theme, trainer, and main layout.
        - Loads icons and schedules initial training preparation.

    prepare_initial_training(self)
        - Starts a background thread to call `self.trainer.prepare_train()` with a loading overlay.

    load_icons(self)
        - Loads and resizes icon images used in the GUI from `assets/icons`.

    create_ui(self)
        - Builds the full user interface including:
            * loading and success overlays
            * generation and history tabs (Notebook)

    create_generation_tab(self)
        - Creates the "Generation" tab with:
            * A 2x2 image preview area
            * Prompt input
            * Start/stop and export buttons

    create_history_tab(self)
        - Sets up the "History" tab with a scrollable frame.

    update_images(self)
        - Updates the displayed images with trainer outputs every few milliseconds.
        - Renders views using `_render_training_views`.

    start_training(self)
        - Disables widgets and starts training in a separate thread using `trainer.prepare_train()`.

    stop_training(self)
        - Gracefully stops the training and re-enables widgets.

    toggle_training(self)
        - Toggles between start/stop training and updates the icon accordingly.

    show_loading(self, message="Loading...")
        - Displays a loading overlay with a spinning meter and disables interaction.

    hide_loading(self)
        - Hides the loading overlay and re-enables interaction.

    show_success(self, message="Export complete!")
        - Shows a success overlay with a message for a few seconds.

    hide_success_message(self)
        - Hides the success overlay and re-enables interaction.

    set_interactive_widgets_state(self, state)
        - Enables or disables all user-interactive widgets (buttons, entry).

    ask_folder_name_popup(self, base_path)
        - Opens a popup to ask the user for a model folder name.
        - Returns folder name and confirmation status.

    export_images(self)
        - Opens file dialog and prompts for a model name.
        - Exports the trained model using `trainer.save_model()` and shows a success overlay.

    run(self)
        - Starts the Tkinter event loop (`mainloop`) for the GUI.
    """
    def __init__(self, prompt="", display_interval=10):
        """
            Initializes the GUITrainer instance, sets up the main application window, styles, and widgets.
            This includes setting up the trainer backend, configuring the layout, applying styles,
            loading icons, and initializing the UI.
        """
        self.trainer = GaussianTrainer(prompt)
        window_width = int((800 * 2 + 200) * 0.5)
        window_height = int((800* 1.5) * 0.5)
        
        self.root = tb.Window(themename="darkly")
        self.root.title("Gaussian 3D Generator")
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(600, 800)
        
        self.root.after(100, self.prepare_initial_training)  # Schedule after UI renders

        self.display_interval = display_interval
        self.prompt = prompt
        
        self.history = []
        self.interactive_widgets = []

        # Define custom icon button style
        style = tb.Style()
        style.configure(
            "Icon.TButton",
            background="#222222",
            borderwidth=0,
            relief="flat",
            padding=2,
            width=32,
            height=32
        )
        style.map(
            "Icon.TButton",
            background=[("active", "#222222")],
            relief=[("pressed", "flat"), ("!pressed", "flat")]
        )

        self.load_icons()
        self.create_ui()
    
    def prepare_initial_training(self):
        """
            Prepares the initial training setup in a separate thread to avoid blocking the UI.
            Displays a loading overlay while preparing the training environment.
        """
        def task():
            self.show_loading("Preparing initial training...")
            self.trainer.prepare_train()
            self.hide_loading()
            print("Initial training preparation complete.")

        threading.Thread(target=task, daemon=True).start()

    def load_icons(self):
        """
            Loads and resizes icon images used in the GUI from `assets/icons`.
            Icons include start, stop, and export buttons.
            Also loads a success image for the success overlay.
        """
        # Get the base path for assets
        if hasattr(sys, "_MEIPASS"):
            base_path = os.path.join(sys._MEIPASS, "assets/icons")
        else:
            base_path = "assets/icons"

        self.icon_start = ImageTk.PhotoImage(Image.open(os.path.join(base_path, "start.png")).resize((27, 28)))
        self.icon_stop = ImageTk.PhotoImage(Image.open(os.path.join(base_path, "stop.webp")).resize((28, 28)))
        self.icon_export = ImageTk.PhotoImage(Image.open(os.path.join(base_path, "export.png")).resize((28, 28))) 
        success_img = Image.open(os.path.join(base_path, "success.webp")).resize((64, 64))
        self.success_image_tk = ImageTk.PhotoImage(success_img)

    def create_ui(self):
        """
            Creates the main user interface layout including:
            - Loading overlay with a spinning meter
            - Success overlay with a success message and image
            - Notebook with two tabs: Generation and History
            - Generation tab with a prompt bar, image previews, and action buttons
            - History tab with a scrollable frame (currently placeholder)
        """

        self.loading_overlay = tb.Frame(self.root, style="dark", bootstyle="dark")
        self.loading_overlay.place(relx=0.5, rely=0.5, anchor="center")

        # Meter only, with padded frame and unified style
        self.loading_meter = tb.Meter(
            self.loading_overlay,
            metersize=160,
            amountused=0,
            amounttotal=100,
            metertype="full",
            bootstyle=PRIMARY,
            interactive=False,
            subtext="Loading...",
            subtextfont=("Arial", 15),
            stripethickness=12,
            padding=20,
            showtext=False,
        )
        self.loading_meter.pack(padx=20, pady=20)
        
        # Loading text
        self.loading_subtext = tb.Label(
            self.loading_overlay,
            text="Loading...",
            font=("Arial", 12),
            background="#303030",
            padding=(10, 10),
        )
        self.loading_subtext.pack()

        # Hide overlay initially
        self.loading_overlay.lower()

        # === Success Overlay ===
        self.success_overlay = tb.Frame(self.root, style="dark", bootstyle="dark")
        self.success_overlay.place(relx=0.5, rely=0.5, anchor="center")    

        self.success_image_label = tb.Label(
            self.success_overlay,
            image=self.success_image_tk,
            background="#303030"
        )
        self.success_image_label.pack(pady=(10, 5))

        self.success_text = tb.Label(
            self.success_overlay,
            text="Export successful!",
            font=("Arial", 13),
            background="#303030",
            padding=(10, 10)
        )
        self.success_text.pack()

        # Initially hide success overlay
        self.success_overlay.lower()

        self.notebook = tb.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True)

        self.gen_frame = tb.Frame(self.notebook)
        self.hist_frame = tb.Frame(self.notebook)

        self.notebook.add(self.gen_frame, text="Generation")
        self.notebook.add(self.hist_frame, text="History")

        self.create_generation_tab()
        self.create_history_tab()


    def create_generation_tab(self):
        """
            Creates the "Generation" tab with:
            - A 2x2 image preview area
            - Prompt input field
            - Start/stop and export buttons
        """
        canvas = Canvas(self.gen_frame, bg="#2b2b2b", highlightthickness=0)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)

        outer_frame = tb.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=outer_frame, anchor="nw")

        self.scrollable_frame = tb.Frame(outer_frame)
        self.scrollable_frame.place(relx=0.5, rely=0.5, anchor="center")

        def on_canvas_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(window_id, width=event.width, height=event.height)

        canvas.bind("<Configure>", on_canvas_configure)

        content_frame = tb.Frame(self.scrollable_frame)
        content_frame.pack()

        image_frame = tb.Frame(content_frame)
        image_frame.pack(pady=10)

        self.image_labels = []
        self.image_containers = []

        image_size = (256, 256)
        gray_image = Image.new("RGB", image_size, "#333")
        gray_image_tk = ImageTk.PhotoImage(gray_image)

        for i in range(2):
            row = tb.Frame(image_frame)
            row.pack()
            for j in range(2):
                index = i * 2 + j
                container = tb.Frame(row, width=image_size[0], height=image_size[1])
                container.pack(side=LEFT, padx=10, pady=10)

                img_label = tb.Label(container, image=gray_image_tk)
                img_label.image = gray_image_tk
                img_label.pack()
                self.image_labels.append(img_label)
                self.image_containers.append(container)

        # === Prompt Bar ===
        control_frame = tb.Frame(content_frame)
        control_frame.pack(fill=X, pady=15, padx=10)

        # Create a labelframe container for better styling
        prompt_container = tb.Labelframe(
            control_frame,
            text="  Prompt Bar  ",
            bootstyle="light"
        )
        prompt_container.pack(fill=X, expand=True)

        # Create inner frame for layout
        inner_frame = tb.Frame(prompt_container, padding=10)
        inner_frame.pack(fill=X, expand=True)

        # Create the prompt entry with custom styling
        self.prompt_entry = tb.Entry(
            inner_frame,
            font=("Arial", 10),
            width=50
        )
        self.prompt_entry.insert(0, self.prompt)
        self.prompt_entry.pack(side=LEFT, fill=X, expand=True, padx=5)

        # Create button frame for the action buttons
        button_frame = tb.Frame(inner_frame)
        button_frame.pack(side=LEFT, padx=(5, 5))

        # Toggle button
        self.toggle_button = tb.Button(
            button_frame,
            image=self.icon_start,
            command=self.toggle_training,
            style="Icon.TButton"
        )
        self.toggle_button.pack(side=LEFT, padx=2)

        # Export button
        self.export_button = tb.Button(
            button_frame,
            image=self.icon_export,
            command=self.export_images,
            style="Icon.TButton"
        )
        self.export_button.pack(side=LEFT, padx=2)

        self.interactive_widgets = [
            self.toggle_button,
            self.export_button,
            self.prompt_entry,
        ]

        self.update_images()

    def create_history_tab(self):
        """
            Sets up the "History" tab with a scrollable frame.
        """
        canvas = Canvas(self.hist_frame)
        scrollbar = Scrollbar(self.hist_frame, orient=VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)

        self.scrollable_frame_history = tb.Frame(canvas)
        self.scrollable_frame_history.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame_history, anchor="nw")

        empty_label = tb.Label(self.scrollable_frame_history, text="History will appear here.", font=("Arial", 12))
        empty_label.pack(pady=20)

    def update_images(self):
        """
            Updates the displayed images with trainer outputs every few milliseconds.
            Renders views using `_render_training_views`.
            This function is called recursively using `after` to create a loop.
        """
        if self.trainer.training:
            t, loss = self.trainer.optimizaiton_iteration()

            step_ratio = min(1, self.trainer.step / 500)
            print(f"Step: {self.trainer.step}, Loss: {loss:.4f}")
            resolution = self.trainer.get_resolution_for_step(step_ratio)
            _, _, _, images = self.trainer.render_training_views(resolution)

            for i, img in enumerate(images):
                img = img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
                img = Image.fromarray((img * 255).astype(np.uint8))
                img_tk = ImageTk.PhotoImage(img)
                self.image_labels[i].configure(image=img_tk)
                self.image_labels[i].image = img_tk

        self.root.after(self.display_interval, self.update_images)

    def start_training(self):
        """
            Starts the training process in a separate thread to avoid blocking the UI.
            Disables interactive widgets during training preparation.
        """
        def task():
            self.set_interactive_widgets_state(DISABLED)
            self.trainer.prompt = self.prompt_entry.get().strip()
            self.trainer.prepare_train()
            self.trainer.training = True
            time.sleep(1)  # Simulate some delay for starting
            print("Training preparation complete.")
            self.set_interactive_widgets_state(NORMAL)

        print("Training started...")
        threading.Thread(target=task, daemon=True).start()

    def stop_training(self):
        """
            Gracefully stops the training process and re-enables interactive widgets.
            This function is called when the user clicks the stop button.
        """
        def task():
            self.set_interactive_widgets_state(DISABLED)
            print("Stopping training...")
            self.trainer.training = False
            time.sleep(1)  # Simulate some delay for stopping
            print("Training stopped.")
            self.set_interactive_widgets_state(NORMAL)

        threading.Thread(target=task, daemon=True).start()


    def show_loading(self, message="Loading..."):
        """
            Displays a loading overlay with a spinning meter and disables interaction.
            The message can be customized.
        """
        self.loading_subtext.configure(text=message)
        self.set_interactive_widgets_state(DISABLED)
        self.loading_overlay.lift()

        def spin_meter():
            i = 0
            print(f"message: {message}")
            while self.loading_overlay.winfo_ismapped():
                i = (i + 5) % 105
                self.loading_meter.configure(amountused=i)
                self.loading_overlay.update_idletasks()
                time.sleep(0.05)

        threading.Thread(target=spin_meter, daemon=True).start()


    def hide_loading(self):
        """
            Hides the loading overlay and re-enables interaction.
        """
        self.set_interactive_widgets_state(NORMAL)
        self.loading_overlay.lower()

    def show_success(self, message="Export complete!"):
        """
            Shows a success overlay with a message for a few seconds.
            The message can be customized.
        """
        self.set_interactive_widgets_state(DISABLED)
        self.loading_overlay.lower()
        self.success_overlay.lift()
        self.success_text.configure(text=message)
        self.success_overlay.update_idletasks()

        # Auto-hide after 3 seconds
        self.root.after(3000, self.hide_success_message)


    def hide_success_message(self):
        """
            Hides the success overlay and re-enables interaction.
        """
        self.success_overlay.lower()
        self.set_interactive_widgets_state(NORMAL)

    def set_interactive_widgets_state(self, state):
        """
            Enables or disables all user-interactive widgets (buttons, entry).
            This is used to prevent user interaction during loading or training.
        """
        for widget in self.interactive_widgets:
            widget.configure(state=state)

    def toggle_training(self):
        """
            Toggles between starting and stopping the training process.
            Updates the icons accordingly.
        """
        if not self.trainer.training:
            self.start_training()
            self.toggle_button.configure(image=self.icon_stop)
        else:
            self.stop_training()
            self.toggle_button.configure(image=self.icon_start)

    def ask_folder_name_popup(self, base_path):
        """
            Opens a popup to ask the user for a model folder name.
            Returns the folder name and confirmation status.
        """
        result = {"confirmed": False, "folder_name": ""}

        def on_confirm():
            name = name_entry.get().strip()
            if os.path.exists(os.path.join(base_path, name)):
                error_label.configure(text="Model already exists. Choose a different name.") 
            elif 1 <= len(name) <= 10:
                result["confirmed"] = True
                result["folder_name"] = name
                popup.destroy()
            else:
                error_label.configure(text="Model name must be 1–10 characters.")

        popup = tb.Toplevel(self.root)
        popup.title("Enter Model Name")
        popup.geometry("300x150")
        popup.grab_set()

        tb.Label(popup, text="Enter a model name (max 10 chars):").pack(pady=(10, 5))
        name_entry = tb.Entry(popup, width=30)
        name_entry.pack()

        error_label = tb.Label(popup, text="", foreground="red")
        error_label.pack(pady=(5, 0))

        tb.Button(popup, text="Confirm", command=on_confirm).pack(pady=10)

        self.root.wait_window(popup)
        return result

    def export_images(self):
        """
            Opens a file dialog and prompts for a model name.
            Exports the trained model using `trainer.save_model()` and shows a success overlay.
            This function is called when the user clicks the export button.
        """
        if self.trainer.training:
            self.toggle_training()
        def task():
            # Step 1: User selects base directory
            base_path = filedialog.askdirectory(title="Select Directory to Save Model In")
            if not base_path:
                return  # User canceled

            # Step 2: Prompt user to enter a folder name
            folder_info = self.ask_folder_name_popup(base_path)
            if not folder_info["confirmed"]:
                return  # User canceled

            # Step 3: Create full path
            folder_name = folder_info["folder_name"]
            full_export_path = os.path.join(base_path, folder_name)
            os.makedirs(full_export_path, exist_ok=True)

            self.show_loading("Exporting mesh & texture...")

            try:
                self.trainer.save_model(mode=3,user_save=True,
                    model_name=folder_info["folder_name"],save_dir=full_export_path)
                print(f"Model exported to: {full_export_path}")
            except Exception as e:
                print("Export failed:", e)
            finally:
                self.hide_loading()
                self.show_success("Export successful!")

        threading.Thread(target=task, daemon=True).start()

    def run(self):
        self.root.mainloop()
