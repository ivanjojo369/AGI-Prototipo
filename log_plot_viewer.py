import os
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

PLOTS_DIR = "logs/plots"

class LogPlotViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visor de Gráficos de Logs AGI")
        self.root.geometry("900x700")
        
        self.images = []
        self.index = 0

        self.label = Label(root)
        self.label.pack(pady=10)

        self.btn_prev = Button(root, text="⬅️ Anterior", command=self.prev_image)
        self.btn_prev.pack(side="left", padx=20)

        self.btn_next = Button(root, text="➡️ Siguiente", command=self.next_image)
        self.btn_next.pack(side="right", padx=20)

        self.btn_refresh = Button(root, text="🔄 Refrescar", command=self.load_images)
        self.btn_refresh.pack(side="bottom", pady=10)

        self.load_images()

    def load_images(self):
        """Carga todas las imágenes .png del directorio."""
        self.images = [os.path.join(PLOTS_DIR, f) for f in os.listdir(PLOTS_DIR) if f.endswith(".png")]
        self.images.sort()
        self.index = 0
        if self.images:
            self.show_image()
        else:
            self.label.config(text="⚠️ No se encontraron gráficos.")

    def show_image(self):
        """Muestra la imagen actual en el visor."""
        img_path = self.images[self.index]
        img = Image.open(img_path)
        img = img.resize((850, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        self.label.config(image=photo, text="")
        self.label.image = photo  # evitar que Python lo limpie

        self.root.title(f"Visor de Gráficos - {os.path.basename(img_path)}")

    def next_image(self):
        if self.images:
            self.index = (self.index + 1) % len(self.images)
            self.show_image()

    def prev_image(self):
        if self.images:
            self.index = (self.index - 1) % len(self.images)
            self.show_image()

if __name__ == "__main__":
    if not os.path.exists(PLOTS_DIR):
        print("⚠️ No existe la carpeta de gráficos. Ejecuta analyze_logs.py primero.")
    else:
        root = tk.Tk()
        app = LogPlotViewer(root)
        root.mainloop()
