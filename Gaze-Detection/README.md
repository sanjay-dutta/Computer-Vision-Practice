# Gaze Detection Using Computer Vision & YOLO11

This repository provides a concise overview and starter structure for understanding and implementing **gaze detection** using modern computer vision techniques. The project is inspired by the principles described in *“Gaze Detection with Computer Vision & YOLO11 | Ultralytics”* and highlights how object detection and pose estimation models can support eye-tracking and gaze-estimation tasks.

---

## 📌 Summary of Gaze Detection

Gaze detection is a computer vision method used to determine where a person is looking by analyzing **eye movements, facial features, and head orientation**. Traditional infrared-based eye-tracking systems required specialized hardware, but advancements in AI now enable accurate gaze estimation with regular cameras.

Models like **Ultralytics YOLO11** can detect key facial regions such as **eyes, pupils, and head pose**, which can then be fed into specialized gaze-estimation networks (e.g., GazeNet) to compute the direction of gaze. These techniques support applications in **driver monitoring**, **gaming analytics**, **psychology research**, and **human–computer interaction**.

Although powerful and increasingly accessible, gaze detection faces challenges including **lighting sensitivity**, **occlusions**, **privacy concerns**, and **computational requirements** for real-time predictions.

---

## 🚀 Key Features

* Detection of faces, eyes, and pupils using YOLO11
* Foundation for integrating deep gaze-estimation models
* Works with standard webcams (no IR hardware needed)
* Applicable to real-world scenarios such as:

  * Driver attention monitoring
  * Gaming and eSports performance analysis
  * Cognitive and psychological studies
  * Human–computer interaction (HCI)

---

## 📂 Project Structure (Suggested)

```
├── README.md
├── requirements.txt
├── src/
│   ├── detect_faces.py
│   ├── detect_eyes.py
│   ├── gaze_estimation.py
│   └── utils.py
├── models/
│   ├── yolov11n.pt
│   └── gazenet_pretrained.pt
├── data/
│   ├── sample_images/
│   └── videos/
└── notebooks/
    └── demo.ipynb
```

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/gaze-detection-yolo11.git
cd gaze-detection-yolo11

pip install -r requirements.txt
```

---

## ▶️ Usage

### **1. Run face + eye detection**

```bash
python src/detect_eyes.py --source data/sample_images/
```

### **2. Run gaze estimation**

```bash
python src/gaze_estimation.py --source data/videos/demo.mp4
```

### **3. Jupyter Notebook Demo**

```
notebooks/demo.ipynb
```

---

## 🧠 How It Works

1. **YOLO11** detects face, eyes, and pupils
2. **Head pose estimation** refines directional context
3. A deep gaze-estimation model (e.g., GazeNet) predicts gaze direction
4. Visual overlay shows where the user is looking

---

## 📌 Applications

* Driver monitoring & safety systems
* eSports & gaming performance analysis
* Psychology & cognitive research
* Retail & marketing attention studies
* VR/AR interaction systems

---
