# AI Assistant for Blackjack Game

A real-time application that uses the image recognition model (YOLO) to detect the state of the Blackjack table (player's and dealer's cards). Based on the detected cards, the system dynamically suggests the optimal mathematical move (Hit, Stand, Double, Split, Bust) in accordance with the basic game strategy.

---

## Application features

* **Real-time card detection:** Detection of face and card values based on the image from the webcam using YOLO
* **Intelligent table division:** Automatic division of the screen into dealer and player zones
* **Result stabilization:** System that eliminates the "blinking" effect and temporary AI errors by analyzing a set of cards from the last frames
* **Asynchronous video process:** Working with the camera in a separate thread guaranteeing high FPS and no lag

---

## Project structure

```
├── data
│   └── best.pt
├── main.py
├── notebooks
│   └── ai_blackjack.ipynb
├── requirements.txt
└── src
    ├── app.py
    ├── config.py
    ├── detector.py
    ├── game_logic.py
    ├── ui_renderer.py
    └── video_stream.py
```

---

## Startup instructions

### 1. Clone project and install required packages

```bash
pip install -r requirements.txt
```

### 2. Launching application

```bash
python main.py
```