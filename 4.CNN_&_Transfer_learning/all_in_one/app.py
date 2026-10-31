import streamlit as st
from PIL import Image
import numpy as np
import io, os, time

st.set_page_config(page_title="Computer Vision Demo Lab", page_icon="🔎", layout="wide")
st.title("🔎 Computer Vision Demo Lab")
st.write("Interactive examples for eight computer-vision topics. First run may download pretrained weights.")
st.info("Educational demonstrations only. Model predictions can be wrong; facial-expression labels do not reveal a person's true internal feelings.")

topic = st.sidebar.selectbox("Choose a project", [
    "1. Classical CV — edges and contours",
    "2. YOLO — modern object detection",
    "3. Faster R-CNN — two-stage detection",
    "4. Image classification — MobileNetV2",
    "5. Semantic segmentation — DeepLabV3",
    "6. Facial-expression classification",
    "7. Siamese-style image similarity",
    "8. GAN — generate MNIST-like images"
])
uploaded = st.file_uploader("Upload an image (not required for GAN)", type=["png","jpg","jpeg","webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
else:
    image = None

if image is not None:
    st.image(image, caption="Input image", width=420)

def run_classical(img):
    import cv2
    a=np.array(img.convert("RGB"))
    bgr=cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),50,150)
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=bgr.copy()
    kept=0
    for c in contours:
        if cv2.contourArea(c)<150: continue
        x,y,w,h=cv2.boundingRect(c); cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2); kept+=1
    c1,c2=st.columns(2)
    c1.image(edges,caption="Canny edges")
    c2.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),caption=f"Contours with boxes ({kept})")
    st.caption("Contours detect shapes, not semantic categories such as 'car' or 'person'.")

if topic.startswith("1."):
    if image is None: st.warning("Upload an image to run the classical CV demo.")
    else:
        if st.button("Run OpenCV edges and contours"): run_classical(image)

elif topic.startswith("2."):
    st.write("YOLO predicts object labels and bounding boxes. This example uses a pretrained COCO model.")
    if st.button("Run YOLO detection"):
        if image is None: st.error("Upload an image first.")
        else:
            with st.spinner("Loading model and detecting objects..."):
                from ultralytics import YOLO
                import torch
                model=YOLO("yolo11n.pt")
                result=model.predict(image,conf=0.25,device=0 if torch.cuda.is_available() else "cpu",verbose=False)[0]
                st.image(result.plot()[...,::-1],caption="YOLO detections")
                rows=[]
                for b in result.boxes:
                    rows.append({"class":model.names[int(b.cls[0])],"confidence":round(float(b.conf[0]),3),"box_xyxy":[round(float(x),1) for x in b.xyxy[0]]})
                st.dataframe(rows,use_container_width=True)

elif topic.startswith("3."):
    st.write("Faster R-CNN is a two-stage detector (region proposals, then classification/refinement).")
    if st.button("Run Faster R-CNN"):
        if image is None: st.error("Upload an image first.")
        else:
            with st.spinner("Loading Torchvision model..."):
                import torch, torchvision
                from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                model=fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
                x=weights.transforms()(image).to(device)
                with torch.no_grad(): pred=model([x])[0]
                cats=weights.meta["categories"]
                fig,ax=plt.subplots(figsize=(10,7)); ax.imshow(image)
                rows=[]
                for box,label,score in zip(pred["boxes"].cpu(),pred["labels"].cpu(),pred["scores"].cpu()):
                    if float(score)<0.5: continue
                    x1,y1,x2,y2=box.tolist()
                    ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,linewidth=2))
                    ax.text(x1,y1,f"{cats[int(label)]} {float(score):.2f}",color="white",bbox=dict(facecolor="green",alpha=.7))
                    rows.append({"class":cats[int(label)],"confidence":round(float(score),3)})
                ax.axis("off"); st.pyplot(fig); plt.close(fig)
                st.dataframe(rows)

elif topic.startswith("4."):
    st.write("MobileNetV2 predicts ImageNet classes for the whole image (top-5 labels).")
    if st.button("Classify image"):
        if image is None: st.error("Upload an image first.")
        else:
            with st.spinner("Loading TensorFlow/Keras MobileNetV2..."):
                import tensorflow as tf
                from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
                model=MobileNetV2(weights="imagenet")
                arr=np.array(image.resize((224,224)),dtype=np.float32)[None,...]
                pred=model.predict(preprocess_input(arr),verbose=0)
                rows=[{"label":label,"probability":float(score)} for _,label,score in decode_predictions(pred,top=5)[0]]
                st.dataframe(rows,use_container_width=True)
                st.bar_chart({r["label"]:r["probability"] for r in rows})

elif topic.startswith("5."):
    st.write("DeepLabV3 assigns a class to each pixel, producing a segmentation mask.")
    if st.button("Segment image"):
        if image is None: st.error("Upload an image first.")
        else:
            with st.spinner("Loading DeepLabV3..."):
                import torch
                from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
                weights=DeepLabV3_ResNet50_Weights.DEFAULT
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model=deeplabv3_resnet50(weights=weights).to(device).eval()
                x=weights.transforms()(image).unsqueeze(0).to(device)
                with torch.no_grad(): mask=model(x)["out"][0].argmax(0).cpu().numpy()
                c1,c2=st.columns(2); c1.image(image,caption="Original")
                c2.image(mask.astype("uint8"),caption="Predicted class mask")
                vals,counts=np.unique(mask,return_counts=True)
                cats=weights.meta["categories"]
                st.write("Pixel counts by class:",{cats[int(v)]:int(c) for v,c in zip(vals,counts) if int(v)<len(cats)})

elif topic.startswith("6."):
    st.write("Classifies visible facial-expression patterns. Use only consented example images; predictions are not proof of a person's feelings.")
    if st.button("Predict expression"):
        if image is None: st.error("Upload a face image first.")
        else:
            with st.spinner("Loading pretrained expression model (internet required)..."):
                from transformers import pipeline
                clf=pipeline("image-classification",model="trpakov/vit-face-expression")
                pred=clf(image)
                st.dataframe([{"label":p["label"],"score":p["score"]} for p in pred[:5]],use_container_width=True)

elif topic.startswith("7."):
    st.write("Quick demonstration of image similarity using pretrained ResNet18 features. The notebook trains a true Siamese network on MNIST pairs.")
    other=st.file_uploader("Upload the second image to compare",type=["png","jpg","jpeg","webp"],key="second")
    if st.button("Compare images"):
        if image is None or other is None: st.error("Upload both images.")
        else:
            with st.spinner("Extracting embeddings..."):
                import torch
                import torch.nn.functional as F
                from torchvision.models import resnet18, ResNet18_Weights
                weights=ResNet18_Weights.DEFAULT
                model=resnet18(weights=weights)
                model.fc=torch.nn.Identity()
                model.eval()
                tfm=weights.transforms()
                a=tfm(image).unsqueeze(0); b=tfm(Image.open(other).convert("RGB")).unsqueeze(0)
                with torch.no_grad(): za=F.normalize(model(a),dim=1); zb=F.normalize(model(b),dim=1)
                sim=float(F.cosine_similarity(za,zb))
                st.metric("Cosine similarity",f"{sim:.3f}")
                st.caption("This is a pretrained-feature similarity baseline, not a validated identity or verification system.")

else:
    st.write("Train a tiny DCGAN on MNIST to generate synthetic digit-like images. Training takes time; use Colab GPU for the notebook for best speed.")
    epochs=st.slider("Training epochs",1,5,1)
    if st.button("Train mini GAN and generate samples"):
        with st.spinner("Training DCGAN on MNIST. This may take several minutes..."):
            import torch, torch.nn as nn
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ds=datasets.MNIST("./data",train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,),(.5,))]))
            loader=DataLoader(ds,batch_size=128,shuffle=True,drop_last=True,num_workers=0)
            zdim=100
            G=nn.Sequential(nn.Linear(zdim,256*7*7),nn.BatchNorm1d(256*7*7),nn.ReLU(True),nn.Unflatten(1,(256,7,7)),nn.ConvTranspose2d(256,128,4,2,1),nn.BatchNorm2d(128),nn.ReLU(True),nn.ConvTranspose2d(128,64,4,2,1),nn.BatchNorm2d(64),nn.ReLU(True),nn.Conv2d(64,1,3,1,1),nn.Tanh()).to(device)
            D=nn.Sequential(nn.Conv2d(1,64,4,2,1),nn.LeakyReLU(.2,True),nn.Conv2d(64,128,4,2,1),nn.BatchNorm2d(128),nn.LeakyReLU(.2,True),nn.Flatten(),nn.Linear(128*7*7,1)).to(device)
            og=torch.optim.Adam(G.parameters(),lr=2e-4,betas=(.5,.999)); od=torch.optim.Adam(D.parameters(),lr=2e-4,betas=(.5,.999)); lossfn=nn.BCEWithLogitsLoss()
            for ep in range(epochs):
                for real,_ in loader:
                    real=real.to(device); n=real.size(0)
                    od.zero_grad(); lr=lossfn(D(real),torch.ones(n,1,device=device)); lf=lossfn(D(G(torch.randn(n,zdim,device=device)).detach()),torch.zeros(n,1,device=device)); (lr+lf).backward(); od.step()
                    og.zero_grad(); fake=G(torch.randn(n,zdim,device=device)); lg=lossfn(D(fake),torch.ones(n,1,device=device)); lg.backward(); og.step()
                st.write(f"Epoch {ep+1}/{epochs} complete")
            with torch.no_grad(): samples=G(torch.randn(25,zdim,device=device)).cpu()
            grid=np.zeros((5*28,5*28),dtype=np.float32)
            for i,s in enumerate(samples):
                r,c=divmod(i,5); grid[r*28:(r+1)*28,c*28:(c+1)*28]=((s.squeeze().numpy()+1)/2).clip(0,1)
            st.image(grid,caption="Generated synthetic MNIST-like samples",clamp=True)
