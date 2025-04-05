import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import io
from typing import List
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Sudoku Digit Recognition API")
#cor function
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    cnn_model = load_model('public/model/mnist_cnn_model.h5')
except Exception as e:
    raise RuntimeError(f"Failed to load CNN model: {str(e)}")

# === Helper Functions ===

def read_and_preprocess_image(file_contents):
    img = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Error: Unable to load image from uploaded file")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, gray, blurred

def find_sudoku_contour(blurred):
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None

def reorder_points(pts):
    pts = pts.reshape(4, 2)
    new_pts = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    new_pts[0] = pts[np.argmin(s)]       # Top-left
    new_pts[2] = pts[np.argmax(s)]       # Bottom-right
    new_pts[1] = pts[np.argmin(diff)]    # Top-right
    new_pts[3] = pts[np.argmax(diff)]    # Bottom-left
    return new_pts

def perspective_transform(img, contour):
    reordered = reorder_points(contour)
    side = max([
        np.linalg.norm(reordered[0] - reordered[1]),
        np.linalg.norm(reordered[1] - reordered[2]),
        np.linalg.norm(reordered[2] - reordered[3]),
        np.linalg.norm(reordered[3] - reordered[0])
    ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(reordered, dst)
    warped = cv2.warpPerspective(img, M, (int(side), int(side)))
    return warped

def detect_grid_lines(gray_img):
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 50, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=20)
    return lines

def draw_grid_lines(gray_img, lines):
    grid_lines_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(grid_lines_img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    return grid_lines_img

def convert_to_binary(image_bgr):
    binary_img = cv2.adaptiveThreshold(image_bgr[:, :, 0], 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return binary_img

def convert_to_black_and_white(file_contents):
    img = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Error: Unable to load image from uploaded file")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    return sharpened

def split_image_into_cells(image, grid_size=(9, 9)):
    h, w = image.shape[:2]
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]
    cells = []
    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = image[y1:y2, x1:x2]
            row.append(cell)
        cells.append(row)
    return cells

def preprocess_cell_for_cnn(cell):
    if len(cell.shape) == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    return inverted

def is_empty_cell(binary, pixel_threshold=30):
    return cv2.countNonZero(binary) < pixel_threshold

def cnn_predict_digit(cell_img):
    resized = cv2.resize(cell_img, (28, 28))
    normalized = resized.astype("float32") / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    prediction = cnn_model.predict(reshaped, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return digit if confidence > 0.8 else 0

# === API Endpoints ===

@app.post("/process-sudoku/", response_model=List[List[int]])
async def process_sudoku_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # First, convert the image to black and white and enhance its quality
        enhanced_image = convert_to_black_and_white(contents)
        
        # Process the original image
        original, _, blurred = read_and_preprocess_image(contents)
        contour = find_sudoku_contour(blurred)
        if contour is None:
            raise HTTPException(status_code=400, detail="Sudoku grid not found in the image")
        
        warped = perspective_transform(original, contour)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        lines = detect_grid_lines(gray)
        grid_removed = draw_grid_lines(gray, lines)
        binary_img = convert_to_binary(grid_removed)
        
        # Predict digits
        resized = cv2.resize(binary_img, (450, 450))
        cells = split_image_into_cells(resized)
        predicted_grid = []

        for i, row in enumerate(cells):
            predicted_row = []
            for j, cell in enumerate(row):
                processed = preprocess_cell_for_cnn(cell)
                if is_empty_cell(processed):
                    predicted_row.append(0)
                else:
                    digit = cnn_predict_digit(processed)
                    predicted_row.append(digit)
            predicted_grid.append(predicted_row)

        return predicted_grid

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preview-processing/")
async def preview_processing(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Process the image and get intermediate steps
        original, _, blurred = read_and_preprocess_image(contents)
        contour = find_sudoku_contour(blurred)
        if contour is None:
            raise HTTPException(status_code=400, detail="Sudoku grid not found in the image")
        
        warped = perspective_transform(original, contour)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        lines = detect_grid_lines(gray)
        grid_removed = draw_grid_lines(gray, lines)
        binary_img = convert_to_binary(grid_removed)
        
        # Convert images to bytes for response
        def image_to_bytes(img):
            _, encoded_img = cv2.imencode('.png', img)
            return io.BytesIO(encoded_img.tobytes())
        
        return {
            "original": StreamingResponse(image_to_bytes(original)),
            "warped": StreamingResponse(image_to_bytes(warped)),
            "grid_removed": StreamingResponse(image_to_bytes(grid_removed)),
            "binary": StreamingResponse(image_to_bytes(binary_img))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)