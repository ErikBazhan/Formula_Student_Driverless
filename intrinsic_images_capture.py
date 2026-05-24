import math
import time
from pathlib import Path

import cv2
import depthai as dai

"""
intrinsic_images_capture.py

Dieses Skript nimmt von jeder angeschlossenen Kamera nacheinander Bilder für
die Intrinsic-Kalibrierung auf.

Ablauf:
1. Alle angeschlossenen Kameras werden erkannt.
2. Für jede Kamera wird eine eigene Aufnahmephase gestartet.
3. Zu Beginn jeder Phase gibt es eine START_DELAY, damit Fokus und Belichtung
   eingestellt werden können.
4. Danach werden pro Kamera maximal MAX_IMAGES_PER_CAMERA Bilder gespeichert.
5. Die Bilder werden pro Kamera in separate Ordner unter captures/ abgelegt.
6. In der Vorschau wird ein Status-Text angezeigt, die gespeicherten Bilder
   selbst bleiben jedoch ohne Text.

Zweck:
Die aufgenommenen Bilder können später zur Berechnung der intrinsischen
Kameraparameter verwendet werden.
"""

SAVE_INTERVAL = 1.0
MAX_IMAGES_PER_CAMERA = 30
START_DELAY = 5.0
BASE_DIR = Path("captures_intrinsic")

device = dai.Device()

with dai.Pipeline(device) as pipeline:
    output_queues = {}
    cam_order = []

    sockets = device.getConnectedCameras()

    for cam_idx, socket in enumerate(sockets, start=1):
        cam_name = f"Cam{cam_idx}"
        cam_order.append(cam_name)

        cam_dir = BASE_DIR / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)

        camera_node = pipeline.create(dai.node.Camera).build(socket)

        output = camera_node.requestOutput(
            size=(640, 480),
            type=dai.ImgFrame.Type.BGR888p,
            resizeMode=dai.ImgResizeMode.CROP,
            fps=15,
        )

        output_queues[cam_name] = output.createOutputQueue(
            maxSize=1,
            blocking=False,
        )

    pipeline.start()

    try:
        for active_cam in cam_order:
            print(f"\n--- Starte Aufnahme für {active_cam} ---")
            print(f"{START_DELAY:.0f} Sekunden Zeit zum Einstellen...")

            capture_start = time.monotonic() + START_DELAY
            saved_images = 0
            last_saved = 0.0

            while saved_images < MAX_IMAGES_PER_CAMERA and pipeline.isRunning():
                now = time.monotonic()

                # Alle Queues kurz leeren, damit nichts hängen bleibt
                latest_frames = {}
                for cam_name, queue in output_queues.items():
                    packet = queue.tryGet()
                    if packet is not None:
                        latest_frames[cam_name] = packet.getCvFrame()

                # Nur das aktive Kamera-Bild anzeigen und speichern
                if active_cam in latest_frames:
                    frame = latest_frames[active_cam]
                    display_frame = frame.copy()

                    if now < capture_start:
                        remaining = math.ceil(capture_start - now)
                        cv2.putText(
                            display_frame,
                            f"Start in {remaining}s",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.putText(
                            display_frame,
                            f"{active_cam} | {saved_images}/"
                            f"{MAX_IMAGES_PER_CAMERA}",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        if now - last_saved >= SAVE_INTERVAL:
                            saved_images += 1
                            filename = (
                                BASE_DIR
                                / active_cam
                                / (
                                    f"{active_cam}_photo_"
                                    f"{saved_images:02d}.png"
                                )
                            )
                            cv2.imwrite(str(filename), frame)
                            last_saved = now
                            print(f"Gespeichert: {filename}")

                    cv2.imshow(active_cam, display_frame)

                if cv2.waitKey(1) == ord("q"):
                    raise KeyboardInterrupt

            print(f"{active_cam} fertig.")

        print("\nAlle Kameras wurden aufgenommen.")

    finally:
        cv2.destroyAllWindows()