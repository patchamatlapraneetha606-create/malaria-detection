# Malaria Detection System

Automated malaria screening using deep learning: classify blood smear cell images as **Parasitized** (infected) or **Uninfected**. Built with Python, TensorFlow, Keras, CNN, and OpenCV for use in clinics, hospitals, and mobile health units.

## Features

- **Parasite detection** – Binary classification of cell images
- **Image classification** – Parasitized vs Uninfected
- **Automated diagnosis** – Upload an image and get a result with confidence
- **Model accuracy analysis** – Confusion matrix, ROC curve, precision/recall/F1
- **Healthcare support UI** – Streamlit app for diagnosis, training, and evaluation

## Technologies

- **Python 3.8+**
- **TensorFlow & Keras** – Deep learning
- **CNN** – Custom architecture + transfer learning (MobileNetV2, EfficientNetB0)
- **OpenCV** – Image loading and preprocessing

## Project Structure

```
praneeproject/
├── config.py          # Paths and hyperparameters
├── data_loader.py     # Dataset loading, OpenCV preprocessing
├── models.py          # Custom CNN, MobileNetV2, EfficientNetB0
├── train.py           # Training script
├── evaluate.py        # Accuracy analysis, confusion matrix, ROC
├── predict.py         # Single/batch prediction (CLI)
├── app.py             # Streamlit healthcare UI
├── download_data.py   # Dataset directory setup
├── requirements.txt
├── data/
│   └── cell_images/
│       ├── Parasitized/
│       └── Uninfected/
├── saved_models/      # Trained models
└── results/           # Reports and plots
```

## Setup

1. **Create environment and install dependencies**

   ```bash
   cd praneeproject
   python -m venv venv
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

2. **Dataset (required for training)**

   **Option A – Download sample data (recommended)**  
   Run once to download the NIH malaria cell images (~337 MB) and extract 500 images per class:

   ```bash
   python download_data.py --max-per-class 500
   ```

   To use all images (no limit):

   ```bash
   python download_data.py
   ```

   **Option B – Manual**  
   Place cell images in:

   - `data/cell_images/Parasitized/` – infected cell images  
   - `data/cell_images/Uninfected/` – uninfected cell images  

   Supported formats: `.png`, `.jpg`, `.jpeg`. You can also use the [NIH](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-dataset) or [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) datasets.

## Running the Application

### 1. Web UI (recommended)

```bash
streamlit run app.py
```

- **Diagnose** – Upload a cell image and get Parasitized/Uninfected with confidence.  
- **Accuracy Analysis** – View classification report, confusion matrix, ROC (after evaluation).  
- **Train Model** – Choose architecture (custom, MobileNetV2, EfficientNetB0) and start training.

### 2. Train from command line

```bash
python train.py --model mobilenetv2 --epochs 15
# Or: custom, efficientnet
```

Models are saved under `saved_models/<model_name>/` (e.g. `best.keras`, `final.keras`).

### 3. Evaluate and accuracy analysis

```bash
python evaluate.py --model mobilenetv2
```

Writes `results/classification_report.txt`, `confusion_matrix.png`, `roc_curve.png`.

### 4. Predict (CLI)

Single image:

```bash
python predict.py path/to/cell_image.png --model mobilenetv2
```

Batch:

```bash
python predict.py --batch img1.png img2.png --model mobilenetv2
```

## Models

| Model         | Description                    |
|---------------|--------------------------------|
| `custom`      | Custom CNN (Conv2D + BN + Dropout) |
| `mobilenetv2` | Transfer learning, ImageNet weights |
| `efficientnet`| EfficientNetB0, transfer learning  |

Default image size: **224×224**. Training uses validation split (20%), optional augmentation (flip, rotation, contrast), early stopping, and learning rate reduction.

## Citation / Dataset

If you use the NIH malaria dataset, please cite the relevant paper from the [NIH page](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-dataset).

## Disclaimer

This system is for **research and decision support** only. It is not a replacement for professional medical diagnosis. Always follow local clinical guidelines and quality assurance procedures.
