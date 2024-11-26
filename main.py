import os
import sys
import tkinter as tk
from ocr_gui import launch_gui

def main():
    # Ensure Tesseract can find the correct language data
    if sys.platform == "win32":
        # For Windows, set the Tesseract data path dynamically
        os.environ['TESSDATA_PREFIX'] = os.path.join(os.getcwd(), 'tessdata')
    
    # Launch the GUI from ocr_gui.py
    launch_gui()

if __name__ == "__main__":
    main()
