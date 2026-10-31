# Google Colab setup
1. Open https://colab.research.google.com/
2. Upload a notebook from `notebooks/`.
3. Choose Runtime → Change runtime type → T4 GPU (if available).
4. Run cells top-to-bottom. The notebooks install their main dependencies.
5. YOLO, Faster R-CNN, MobileNetV2, DeepLabV3, and facial-expression model weights are downloaded at first use.
6. Siamese and DCGAN notebooks download MNIST and train small models. Start with the included low epoch counts, then increase if time allows.
7. If the GPU is unavailable, most inference notebooks fall back to CPU; training may take longer.
