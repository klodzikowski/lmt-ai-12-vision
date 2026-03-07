# VisionDesk Live (MacBook)

Konsumencka aplikacja webowa do:
- detekcji obiektow z kamery (`person`, `cup`, `dog`, `laptop`, itd.),
- rozpoznawania emocji twarzy (`happy`, `sad`, `neutral`, ...),
- rysowania bounding boxow i etykiet w czasie rzeczywistym.

W interfejsie jest panel `Models in use` z tooltipami (najedz kursorem), ktory opisuje dokladnie, jakie modele dzialaja pod spodem.

## Wymagania

- macOS + kamera (MacBook Pro 14 OK)
- nowoczesna przegladarka (Chrome lub Edge, Safari tez powinno dzialac)
- internet przy pierwszym uruchomieniu (pobranie modelu z CDN)

## Uruchomienie

1. Otworz terminal i przejdz do katalogu projektu:

```bash
cd "/Users/Kacplodz/Documents/New project/Codex/Camera Object Detector"
```

2. Uruchom prosty serwer HTTP:

```bash
python3 -m http.server 8010
```

3. Otworz aplikacje:

```text
http://127.0.0.1:8010/
```

4. Kliknij `Start camera` i zezwol na dostep do kamery.

## Uzycie

- Suwak `Confidence` ustawia prog detekcji.
- `Stop` zatrzymuje kamere.
- Panel boczny pokazuje `Live detections`, `Face emotions` i `Models in use` (modele + opisy w tooltipach).
