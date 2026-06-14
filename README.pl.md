# Asystent AI do Gry w Blackjacka

Aplikacja działająca w czasie rzeczywistym, która wykorzystuje model rozpoznawania obrazu (YOLO) do detekcji stanu stołu w grze Blackjack (karty gracza oraz krupiera). Na podstawie wykrytych kart system dynamicznie sugeruje optymalny ruch matematyczny (Hit, Stand, Double, Split, Bust) zgodnie z podstawową strategią gry.

---

## Funkcje aplikacji

* **Detekcja kart w czasie rzeczywistym:** Wykrywanie figur i wartości kart na podstawie obrazu z kamery internetowej przy użyciu YOLO
* **Inteligentny podział stołu:** Automatyczny podział ekranu na strefę krupiera i gracza
* **Stabilizacja wyników:** System eliminujący efekt "mrugania" i chwilowych błędów AI poprzez analizę zestawu kart z ostatnich klatek
* **Asynchroniczny proces wideo:** Praca z kamerą w osobnym wątku gwarantująca wysoki wskaźnik FPS i brak opóźnień

---

## Struktura projektu

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

## Instrukcja uruchomienia

### 1. Sklonowanie projektu i zainstalowanie wymaganych pakietów

```bash
pip install -r requirements.txt
```

### 2. Uruchomienie aplikacji

```bash
python main.py
```
