# 💧 Smart City Laboratory – Water Utility Meter Reading (Part 2)

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-24.0-blue?logo=docker)](https://www.docker.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-lightgrey)](https://opencv.org/)
[![MQTT](https://img.shields.io/badge/MQTT-broker-orange?logo=emqtt)](https://mqtt.org/)

---

## 📖 Introduction
This project is part of the **Smart City Laboratory** assignment: *Image Processing in Smart Cities – Online Measurement*.  

The goal is to develop a **microservices-based cloud application** that:
- Processes a water utility meter video stream  
- Extracts digital readings using **computer vision / AI**  
- Publishes results via **MQTT**  
- Provides optional time-series storage and analytics via **InfluxDB + Grafana**  

The system is fully containerized using **Docker** and orchestrated with **Docker Compose** for cloud-ready deployment.

---

## 🎯 Objectives
- Capture frames from RTSP stream: `rtsp://vizora.ddns.net:8554/watermeter`  
- Extract meter readings using **OpenCV** and AI (TFLite) models  
- Publish readings every 15 seconds to **MQTT** in JSON format  
- Deploy supporting containers for analytics (optional)  

---

## 📂 Project Structure

│── .gitignore
│── Dockerfile
│── docker-compose.yaml
│── mosquitto.conf
│── requirements.txt
│── water_meter_v4.py # Main Python script
│── 10.mp4 # Sample video
│── 15.jpg # Sample test image


---

## 🛠️ Setup & Installation

### 1️⃣ Requirements
- Docker  
- Docker Compose  
- (Optional) WSL2 + Docker Desktop on Windows  

### 2️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/shuhratkulboboev.git
cd shuhratkulboboev


### 3️⃣ Build Docker Image

Build the Docker image for the water meter service:

```bash
docker build -t water-meter-service .

4️⃣ Run with Docker Compose

Start all containers using Docker Compose:

docker-compose up


This will start:

Meter processing container

Mosquitto MQTT broker

Optional: InfluxDB + Grafana stack for analytics

⚙️ Usage
Process the Video Stream

Reads frames from RTSP stream: rtsp://vizora.ddns.net:8554/watermeter

Extracts meter digits (m³ resolution) using OpenCV / AI model

Publishes readings as JSON payload:

{ "meter": 123.456 }

MQTT Broker

Host: vizora.ddns.net

Port: 1883

Topic: VITMMB09/<your-identifier>

You can use a custom identifier instead of your Neptun code for privacy. Document it in your report.

🔄 Workflow

Capture frame from RTSP stream

Extract meter digits using OpenCV / AI model

Publish reading via MQTT

Repeat every 15 seconds

Verify readings using an MQTT client

🧪 Testing

Use 10.mp4 or 15.jpg for local testing

Verify MQTT messages with:

mosquitto_sub -h vizora.ddns.net -p 1883 -t VITMMB09/<identifier>
