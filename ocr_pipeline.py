import cv2
import numpy as np
import pandas as pd
import pytesseract
from concurrent.futures import ThreadPoolExecutor

# Preprocess image for contour detection
def preprocess_for_contours(image, kernel_size=(3, 3)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 100, 200)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)

# Crop the largest contour
def crop_to_largest_contour(image, dilated):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return image[y:y+h, x:x+w]

# Preprocess image for OCR
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.GaussianBlur(thresh, (5, 5), 0)

# Generate blobs to define OCR bounding boxes
def generate_blobs(image, kernel_size=(5, 5), iterations=4):
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    dilated[dilated != 0] = 255  # Ensure blobs are white
    return dilated

# Extract bounding boxes from blobs
def get_bounding_boxes(blob_image):
    contours, _ = cv2.findContours(blob_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted([cv2.boundingRect(c) for c in contours], key=lambda b: (b[1], b[0]))

# Mask out unwanted regions based on bounding boxes
def mask_columns(image, bounding_boxes, tolerance=10):
    x_coords = sorted(set(x for x, _, _, _ in bounding_boxes))

    # Cluster x-coordinates based on tolerance
    clusters = []
    current_cluster = [x_coords[0]]
    for x in x_coords[1:]:
        if abs(x - current_cluster[-1]) <= tolerance:
            current_cluster.append(x)
        else:
            clusters.append(current_cluster)
            current_cluster = [x]
    clusters.append(current_cluster)

    if len(clusters) < 2:
        raise ValueError("Not enough columns to mask.")

    # Mask first and second-to-last column
    columns_to_mask = [clusters[0][0], clusters[-2][0]]
    for column_x in columns_to_mask:
        left_x = min(x for x, _, _, _ in bounding_boxes if abs(x - column_x) <= tolerance) - 5
        right_x = max(x + w for x, _, w, _ in bounding_boxes if abs(x - column_x) <= tolerance) + 5
        image[:, left_x:right_x] = 0
        bounding_boxes[:] = [box for box in bounding_boxes if not (left_x <= box[0] <= right_x)]

    return image, bounding_boxes

# Perform OCR using bounding boxes
def perform_ocr(image, bounding_boxes):
    text_column, number_column = [], []

    # Perform OCR for each bounding box in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda box: pytesseract.image_to_string(
                image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]],
                lang='eng+rus+chi_tra+chi_sim+kor+jpn',
                config=r'--oem 1 --psm 6'
            ).strip().replace("\n", ""),  # Replace newlines with nothing to concatenate
            bounding_boxes
        ))

    for text in results:
        if text.isnumeric():
            number_column.append(text)
        elif text:
            text_column.append(text)

    # Handle different column lengths by padding the shorter list with empty strings
    max_len = max(len(text_column), len(number_column))

    # Create DataFrame with equal length columns, padding with empty strings as needed
    df = pd.DataFrame({
        "Members": text_column + [''] * (max_len - len(text_column)),
        "Medals": number_column + [''] * (max_len - len(number_column))
    })

    # Remove rows where 'Medals' is empty or contains just spaces
    df = df[df['Medals'].str.strip().ne('')]

    return df

# Display images for debugging
'''def save_and_display_images(cropped_img, prepped_img, blob_img, masked_img):
    cv2.imshow("Cropped Image", cropped_img)
    cv2.imshow("Prepped Image", prepped_img)
    cv2.imshow("Blob Image", blob_img)
    cv2.imshow("Masked Image", masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

def process_images(img_paths, output_path="output.csv", log_callback=None):
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame for combined results

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            if log_callback:
                log_callback(f"Error: Unable to load image at {img_path}. Skipping...")
            continue

        try:
            # Crop to content and preprocess
            dilated = preprocess_for_contours(img)
            cropped_img = crop_to_largest_contour(img, dilated)
            prepped_img = preprocess_for_ocr(cropped_img)

            # Generate blobs and extract bounding boxes
            blob_img = generate_blobs(prepped_img)
            bounding_boxes = get_bounding_boxes(blob_img)

            # Mask unwanted columns
            masked_img, bounding_boxes = mask_columns(prepped_img, bounding_boxes)

            # Perform OCR
            df = perform_ocr(masked_img, bounding_boxes)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            if log_callback:
                log_callback(f"Error processing {img_path}: {e}")
            continue

    # Remove duplicate rows based on 'Members' and 'Medals' columns
    combined_df.drop_duplicates(subset=["Members", "Medals"], inplace=True)

    return combined_df

def update_log(message):
    # Append the message to the Section Two text box in the GUI
    text_box.insert(tk.END, message + "\n")
    text_box.see(tk.END)  # Scroll to the latest message
