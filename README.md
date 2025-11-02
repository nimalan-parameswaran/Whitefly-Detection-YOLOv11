# Whitefly Detection using YOLOv11m

An AI-powered web app that detects and counts **whiteflies on cassava leaves** using a custom-trained **YOLOv11m** model.
The model is deployed on **Hugging Face Spaces**, while the **frontend** is hosted on **GitHub Pages** for public access.

🔗 **Live App:** [Whitefly Detection Web App](https://nimalan-parameswaran.github.io/Whitefly-Detection-YOLOv11/)

🔗 **Hugging Face Space:** [nimalan/whitefly-cassava](https://huggingface.co/spaces/nimalan/whitefly-cassava)

---

## Repository Structure

| File / Folder        | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| **app.py**           | Backend inference script using Gradio and YOLOv11m ONNX model |
| **best.onnx.svg**    | Model structure visualization                                 |
| **notebook.ipynb**   | Training notebook used for fine-tuning YOLOv11m               |
| **requirements.txt** | Python dependencies for Hugging Face Space                    |
| **index.html**       | Static frontend for the web app                               |
| **README.md**        | Documentation file                                            |
---

## Model Overview

The project uses **YOLOv11m** from the Ultralytics suite, trained for precise insect detection on cassava leaves.

| Parameter                | Value              |
| ------------------------ | ------------------ |
| **Architecture**         | YOLOv11m           |
| **Framework**            | Ultralytics        |
| **Input Size**           | 1280 × 1280        |
| **Epochs**               | 100                |
| **Batch Size**           | 16                 |
| **Confidence Threshold** | 0.3                |
| **Export Format**        | ONNX (`best.onnx`) |

The final model is lightweight and optimised for fast inference while maintaining accuracy in dense pest clusters.

---

## Dataset

The dataset used is the **Cassava Whitefly Dataset** collected from the **National Crop Resources Research Institute (NaCRRI), Uganda**, publicly available on [Mendeley Data](https://doi.org/10.17632/5g38399z9p.3).

| Category               | Description                  |
| ---------------------- | ---------------------------- |
| **Low Abundance**      | <10 whiteflies per leaf      |
| **Moderate Abundance** | 10–100 whiteflies            |
| **Super Abundance**    | >100 whiteflies              |


During preprocessing, images were resized, normalized, and augmented through rotation, flipping, and brightness variation.

---

## Backend — Hugging Face Space

The backend service is hosted on **Hugging Face Spaces** ([nimalan/whitefly-cassava](https://huggingface.co/spaces/nimalan/whitefly-cassava)) using **Gradio** for the interface.
The `app.py` script loads the ONNX model, runs inference, and returns both the annotated image and the detection count.

### [app.py](https://huggingface.co/spaces/nimalan/whitefly-cassava/blob/main/app.py)
---

## Frontend — GitHub Pages

The web interface is hosted on GitHub Pages:
🔗 [https://nimalan-parameswaran.github.io/Whitefly-Detection-YOLOv11/](https://nimalan-parameswaran.github.io/Whitefly-Detection-YOLOv11/)

It communicates directly with the Hugging Face backend through the Gradio API.

### Example API Call (JavaScript)

```JavaScript
import { Client } from "@gradio/client";
	
	const response_0 = await fetch("https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png");
	const exampleImage = await response_0.blob();
						
	const client = await Client.connect("nimalan/whitefly-cassava");
	const result = await client.predict("/predict", { 
					image: exampleImage, 
	});

	console.log(result.data);
```

This allows any web app or frontend to integrate live model predictions.

---

## Installation

If you want to test the backend locally:

```bash
git clone https://github.com/nimalan-parameswaran/Whitefly-Detection-YOLOv11.git
cd Whitefly-Detection-YOLOv11
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL displayed in the console.

---

## Performance Metrics

| Metric                          | Value             |
| ------------------------------- | ----------------- |
| **Precision (mAP50)**           | 96.3%             |
| **Recall**                      | 94.7%             |
| **F1 Score**                    | 0.95              |
| **Inference Speed (1280×1280)** | 22 ms/frame (GPU) |

---

## License

This repository is licensed under the **MIT License**.

Dataset usage must follow the **CC BY 4.0** terms from Mendeley Data.

---

## Author

**NIMALAN P**
<h3 align="left">Connect With Me</h3>
<p align="left">
  <a href="mailto:nimalan936@gmail.com"><img src="https://skillicons.dev/icons?i=gmail" /></a>
  <a href="https://www.linkedin.com/in/nimalan-parameswaran" target="_blank"><img src="https://skillicons.dev/icons?i=linkedin" /></a>
</p>

---
