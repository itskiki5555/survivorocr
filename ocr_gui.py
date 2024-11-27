from tkinterdnd2 import TkinterDnD, DND_FILES
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import os
from ocr_pipeline import process_images  # Import the OCR pipeline

class OCRApp:
    def __init__(self, root):
        style = ttk.Style()
        style.theme_use('darkly')
        self.root = root
        self.root.title("OCR Processor")
        self.root.geometry("800x600")
        self.root.minsize(800, 400)  # Set minimum window size to 800x600
        self.root.resizable(True, True)

        # Initialize drag-and-drop functionality for the root window
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)

        # Initialize the default save location
        self.save_location = self.get_default_folder()

        # Section 1: Input and Output Controls
        self.create_section1()

        # Section 2: Data Editing and Logs
        self.create_section2()

    def create_section1(self):
        section1 = ttk.Frame(self.root, padding=10)
        section1.pack(fill=X, padx=10, pady=(10, 5))
        
        # Row 1: File/Folder Selector
        row1 = ttk.Frame(section1)
        row1.pack(fill=X, pady=5)

        self.file_label = ttk.Label(row1, text="No files selected", anchor="w")
        self.file_label.pack(side=LEFT, fill=X, expand=True)

        select_button = ttk.Button(row1, text="Select Images", command=self.select_files, width=15)
        select_button.pack(side=RIGHT, padx=5)

        # New label to show selected filenames
        self.selected_files_label = ttk.Label(row1, text="", anchor="w")
        self.selected_files_label.pack(side=RIGHT, padx=5)

        # Row 2: Output File Name and Location
        row2 = ttk.Frame(section1)
        row2.pack(fill=X, pady=5)

        self.output_label = ttk.Label(row2, text="Save Location:", anchor="w")
        self.output_label.pack(side=LEFT, fill=X, expand=True)

        self.output_name = ttk.Entry(row2)
        self.output_name.insert(0, "output.csv")
        self.output_name.pack(side=LEFT, padx=5)

        save_button = ttk.Button(row2, text="Change Location", command=self.set_save_location, width=15)
        save_button.pack(side=RIGHT, padx=5)

        # Row 3: Action Buttons
        row3 = ttk.Frame(section1)
        row3.pack(fill=X, pady=(45, 5))

        run_button = ttk.Button(row3, text="Run OCR", command=self.run_ocr, width=15)
        run_button.pack(side=LEFT, padx=5, expand=True)

        save_button = ttk.Button(row3, text="Save Output", command=self.save_output, width=15)
        save_button.pack(side=RIGHT, padx=5, expand=True)

        # Row 4: Logs
        row4 = ttk.Frame(section1)
        row4.pack(fill=X, pady=(10, 0))

        self.log_label = ttk.Label(row4, text="", anchor="center", wraplength=600)
        self.log_label.pack(fill=X, expand=True)

    def create_section2(self):
        section2 = ttk.Frame(self.root, padding=10)
        section2.pack(fill=BOTH, expand=True, padx=10, pady=0)

        # Data Editing Column with Scrollbar
        data_frame = ttk.Frame(section2)
        data_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        self.data_table = ttk.Treeview(
            data_frame,
            columns=("Members", "Medals"),
            show="headings",
            selectmode="browse",
        )
        self.data_table.heading("Members", text="Members")
        self.data_table.heading("Medals", text="Medals")
        self.data_table.column("Members", width=300)
        self.data_table.column("Medals", width=100, anchor="center")
        self.data_table.pack(fill=BOTH, expand=True, side=LEFT)
        self.data_table.bind("<Double-1>", self.on_cell_double_click)

        # Scrollbar for Data Table 
        self.data_scrollbar = ttk.Scrollbar(data_frame, orient=VERTICAL, command=self.data_table.yview)
        self.data_table.configure(yscrollcommand=self.data_scrollbar.set)
        self.data_scrollbar.configure(command=self.data_table.yview)
        self.data_scrollbar.pack(side=RIGHT, fill=Y)

        # CSV Textbox with Scrollbar
        csv_frame = ttk.Frame(section2)
        csv_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # Scrollbar for CSV Textbox
        self.csv_scrollbar = ttk.Scrollbar(csv_frame, orient=VERTICAL)
        self.csv_scrollbar.pack(side=RIGHT, fill=Y)

        # Create Text Widget for CSV content
        self.csv_text = tk.Text(csv_frame, wrap=WORD, font=("Arial", 10), state="disabled", yscrollcommand=self.csv_scrollbar.set)
        self.csv_text.pack(fill=BOTH, expand=True)
        self.csv_scrollbar.config(command=self.csv_text.yview)

        # Create the Copy Button for the CSV Textbox
        copy_button = ttk.Button(
            csv_frame, 
            text="Copy", 
            width=8, 
            command=lambda: self.root.clipboard_clear() or self.root.clipboard_append(self.csv_text.get(1.0, tk.END)) or self.root.update(),
            bootstyle="secondary"
        )
        copy_button.place(relx=1.0, rely=0.0, anchor="ne", x=-15, y=5)  # Top-right position with small padding

    # --- Drag and drop ---
    def handle_drop(self, event):
        """Handle file drag-and-drop into the window."""
        file_paths = self.root.tk.splitlist(event.data)  # Parse multiple files
        if file_paths:
            self.file_label.config(text=f"{len(file_paths)} file(s) selected")
            self.selected_files = file_paths
            
            # Display filenames of selected files
            file_names = [os.path.basename(file) for file in file_paths]
            self.selected_files_label.config(text=", ".join(file_names))
        else:
            self.file_label.config(text="No files selected")
            self.selected_files = []
            self.selected_files_label.config(text="")  # Clear the label if no files are selected

    # --- Editing Dataframe --
    def on_cell_double_click(self, event):
        """Allow the user to edit the value of a cell in the data table."""
        item = self.data_table.selection()
        if not item:
            return

        # Identify the column and row clicked
        region = self.data_table.identify("region", event.x, event.y)
        if region != "cell":  # Ensure the click is on a cell
            return

        col = self.data_table.identify_column(event.x)  # Get the column clicked (e.g., #1, #2)
        col = int(col.strip("#")) - 1  # Convert to 0-based index

        # Ensure the column is valid (only allow editing for "Members" (index 0) and "Medals" (index 1))
        if col not in [0, 1]:  
            return

        x, y, width, height = self.data_table.bbox(item, col)  # Get the bounding box for the cell

        current_value = self.data_table.item(item)["values"][col]

        # Create an entry widget for inline editing
        entry = tk.Entry(self.data_table, font=("Arial", 12))
        entry.insert(0, current_value)
        entry.focus()

        # Adjust placement to align with the clicked cell
        entry.place(
            x=x + self.data_table.winfo_rootx() - self.data_table.winfo_vrootx(),
            y=y + self.data_table.winfo_rooty() - self.data_table.winfo_vrooty(),
            width=width,
            height=height
        )

        # Position the Entry widget on top of the clicked cell
        entry.place(x=x, y=y, width=width, height=height)

        # Event handlers for saving and destroying the entry widget
        def save_edit(event=None):
            new_value = entry.get()
            values = list(self.data_table.item(item)["values"])
            values[col] = new_value
            self.data_table.item(item, values=values)
            self.update_dataframe_from_table()
            self.update_csv_textbox()
            entry.destroy()

        def cancel_edit(event=None):
            entry.destroy()

        entry.bind("<Return>", save_edit)  # Save on Enter key
        entry.bind("<Escape>", cancel_edit)  # Cancel on Escape key

        # Destroy the entry widget when clicking elsewhere
        self.data_table.bind("<Button-1>", lambda e: entry.destroy(), add="+")

    def update_dataframe_from_table(self):
        """Update the DataFrame from the current content of the DataTable."""
        data = []
        for row in self.data_table.get_children():
            data.append(self.data_table.item(row)["values"])

        # Convert the data to a DataFrame and store it
        self.df = pd.DataFrame(data, columns=["Members", "Medals"])    

    # --- Helper Functions ---
    def get_default_folder(self):
        # Automatically select the folder where the program is running
        return os.getcwd()

    def select_files(self):
        default_folder = self.get_default_folder()  # Get the default folder based on the OS
        filetypes = [("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(initialdir=default_folder, filetypes=filetypes)  # Set initial directory

        if paths:
            self.file_label.config(text=f"{len(paths)} file(s) selected")
            self.selected_files = paths
            
            # Update the label to show filenames of selected files
            file_names = [os.path.basename(file) for file in paths]
            self.selected_files_label.config(text=", ".join(file_names))  # Display filenames
        else:
            self.file_label.config(text="No files selected")
            self.selected_files = []
            self.selected_files_label.config(text="")  # Clear the label if no files are selected

    def set_save_location(self):
        default_folder = self.get_default_folder()
        folder_path = filedialog.askdirectory(initialdir=default_folder)

        if folder_path:
            self.save_location = folder_path
            self.output_label.config(text=f"Save Location: {folder_path}")

    def save_output(self):
        if not hasattr(self, "df") or self.df.empty:
            self.log_label.config(text="No data to save!")
            return

        file_name = self.output_name.get()
        if not file_name.endswith(".csv"):
            file_name += ".csv"  # Ensure the file ends with .csv

        save_path = os.path.join(self.save_location, file_name)
        self.df.to_csv(save_path, index=False)
        self.log_label.config(text=f"Output saved to {save_path}")

    def run_ocr(self):
        if not hasattr(self, "selected_files") or not self.selected_files:
            self.log_label.config(text="No files selected!")
            return

        # Run OCR on the selected files 
        text_data = process_images(self.selected_files)  # Process the images and get the extracted text
        self.df = pd.DataFrame(text_data, columns=["Members", "Medals"])

        # Update the data table and CSV textbox
        self.update_data_table()
        self.update_csv_textbox()

    def update_data_table(self):
        """Update the DataTable with the latest DataFrame data."""
        for row in self.data_table.get_children():
            self.data_table.delete(row)

        for _, row in self.df.iterrows():
            self.data_table.insert("", "end", values=list(row))

    def update_csv_textbox(self):
        """Update the CSV Textbox with Data from DataFrame using tabs as delimiter."""
        if not hasattr(self, 'df') or self.df.empty:
            return

        # Convert the DataFrame to CSV format with tabs separating columns
        csv_data = self.df.to_csv(index=False, header=False, sep='\t')

        # Update the CSV Textbox with the new CSV data
        self.csv_text.config(state="normal")
        self.csv_text.delete(1.0, tk.END)  # Clear the current text
        self.csv_text.insert(tk.END, csv_data)  # Insert new data
        self.csv_text.config(state="disabled")  # Set it back to disabled

# Function to launch the GUI
def launch_gui():
    root = TkinterDnD.Tk()
    app = OCRApp(root)
    root.mainloop()

# --- Run the App ---
if __name__ == "__main__":
    launch_gui()
