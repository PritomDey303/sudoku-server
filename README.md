🧩 Sudoku Server – Digit Recognition & Solver API
The Sudoku Server is a Python-based FastAPI backend that uses a trained CNN model to recognize handwritten or printed Sudoku puzzles from images. It extracts the Sudoku grid, detects digits using computer vision techniques, and returns a 2D array representing the puzzle.

🔧 Features
Upload an image of a Sudoku puzzle and receive a digit matrix

Pretrained CNN model (trained on MNIST) for digit classification

Grid extraction and warping using OpenCV

Optional preview endpoint to view preprocessing steps

JSON output for easy frontend integration

📦 Tech Stack
FastAPI – for building the REST API

TensorFlow/Keras – for digit recognition model

OpenCV – for image preprocessing and grid extraction

Uvicorn – ASGI server for local deployment

