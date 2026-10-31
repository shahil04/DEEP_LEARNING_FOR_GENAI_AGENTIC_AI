# Computer Vision Demo Collection (8 topics)

A classroom-ready starter collection with **8 Jupyter notebooks** and a **single Streamlit demo app**. Designed for Google Colab GPU, but most inference demos also work on CPU.

## Included topics
1. Classical object detection with OpenCV (edges/contours)
2. Modern YOLO detection and optional transfer learning
3. Faster R-CNN detection (Torchvision)
4. Image classification (TensorFlow/Keras MobileNetV2)
5. Semantic segmentation (Torchvision DeepLabV3)
6. Facial-expression image classification (Hugging Face pretrained model)
7. Siamese network trained on MNIST image pairs (PyTorch)
8. DCGAN trained on MNIST (PyTorch)

## Quick start in Google Colab
1. Download and extract this ZIP, or upload individual `.ipynb` files to Google Colab.
2. In Colab choose **Runtime → Change runtime type → T4 GPU** (if available).
3. Open a notebook under `notebooks/` and run cells from top to bottom.
4. First runs may download datasets/model weights, so internet access is required.
5. Some notebooks are intentionally demonstrations: inference notebooks use pretrained weights; the Siamese and GAN notebooks train small models. GPU is recommended for those training notebooks.

## Run the Streamlit app locally
Use Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app loads model weights on first use and can take time. The face-expression mode requires internet and the Hugging Face `transformers` package. YOLO uses Ultralytics. The GAN mode offers a simple training action; use a GPU for reasonable speed.

## Suggested classroom flow (60 minutes)
- 0–10 min: classification vs detection vs segmentation vs generation
- 10–25 min: OpenCV contours, then YOLO detections
- 25–35 min: Faster R-CNN and transfer learning comparison
- 35–45 min: classification and segmentation
- 45–52 min: Siamese similarity and facial-expression classification caveats
- 52–60 min: show GAN generated samples and student questions

## Important notes
- Pretrained model downloads require an internet connection.
- For production, pin and test package versions, add model caching, input validation, and explicit license review.
- This is educational demo code, not a validated safety, medical, or identity-verification system.
- Facial-expression labels are visual predictions and do not establish a person's actual emotional state.
- Classical contours detect shapes, not semantic object classes.
- Detection model labels depend on the pretrained dataset. Custom classes require labeled data and training.
- GPU availability in Colab is not guaranteed and session runtimes are temporary.
