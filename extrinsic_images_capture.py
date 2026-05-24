import math
import time
from itertools import combinations
from pathlib import Path

import cv2
import depthai as dai

"""
extrinsic_images_capture.py

Dieses Skript nimmt für alle Kamerapaare des angeschlossenen DepthAI-Systems
nacheinander synchronisierte Bildpaare auf.

Ablauf:
1. Alle vorhandenen Kamerapaare werden automatisch ermittelt.
2. Für jedes Kamerapaar wird eine eigene DepthAI-Pipeline gestartet.
3. Nach einer Startverzögerung von START_DELAY Sekunden können Fokus und
   Belichtung eingestellt werden.
4. Danach werden pro Kamerapaar maximal MAX_IMAGES_PER_CAMERA Bildpaare
   aufgenommen.
5. Die Bilder werden in separate Ordner gespeichert:
   captures_extrinsic/CamX_CamY/...
6. Der Text zur Orientierung wird nur in der Vorschau angezeigt, nicht in den
   gespeicherten Bildern.

Zweck:
Die aufgenommenen Bildpaare können später für die Extrinsic-Kalibrierung
eines Kamerapaares verwendet werden.
"""

SAVE_INTERVAL = 1.0
MAX_IMAGES_PER_CAMERA = 30
START_DELAY = 10.0
BASE_DIR = Path("captures_extrinsic")


def draw_text(frame, text, pos=(30, 50), color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )


# Erst einmal die vorhandenen Kameras ermitteln
with dai.Device() as probe_device:
    sockets = probe_device.getConnectedCameras()
    cam_infos = [
        (f"Cam{i}", socket) for i, socket in enumerate(sockets, start=1)
    ]

camera_pairs = list(combinations(cam_infos, 2))

try:
    for (cam_a, socket_a), (cam_b, socket_b) in camera_pairs:
        pair_name = f"{cam_a}_{cam_b}"
        pair_dir = BASE_DIR / pair_name
        dir_a = pair_dir / cam_a
        dir_b = pair_dir / cam_b

        dir_a.mkdir(parents=True, exist_ok=True)
        dir_b.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Starte Aufnahme für {pair_name} ---")
        print(f"{START_DELAY:.0f} Sekunden Zeit zum Einstellen...")

        # WICHTIG: pro Kamerapaar eine frische Device-Instanz
        with dai.Device() as device:
            with dai.Pipeline(device) as pipeline:
                output_queues = {}

                camera_node_a = pipeline.create(dai.node.Camera).build(
                    socket_a
                )
                output_a = camera_node_a.requestOutput(
                    size=(640, 480),
                    type=dai.ImgFrame.Type.BGR888p,
                    resizeMode=dai.ImgResizeMode.CROP,
                    fps=15,
                )
                output_queues[cam_a] = output_a.createOutputQueue(
                    maxSize=1,
                    blocking=False,
                )

                camera_node_b = pipeline.create(dai.node.Camera).build(
                    socket_b
                )
                output_b = camera_node_b.requestOutput(
                    size=(640, 480),
                    type=dai.ImgFrame.Type.BGR888p,
                    resizeMode=dai.ImgResizeMode.CROP,
                    fps=15,
                )
                output_queues[cam_b] = output_b.createOutputQueue(
                    maxSize=1,
                    blocking=False,
                )

                pipeline.start()

                capture_start = time.monotonic() + START_DELAY
                saved_images = 0
                last_saved = 0.0
                last_frames = {}

                while saved_images < MAX_IMAGES_PER_CAMERA and pipeline.isRunning():
                    now = time.monotonic()

                    for cam_name, queue in output_queues.items():
                        packet = queue.tryGet()
                        while packet is not None:
                            last_frames[cam_name] = packet.getCvFrame()
                            packet = queue.tryGet()

                    if cam_a not in last_frames or cam_b not in last_frames:
                        if cv2.waitKey(1) == ord("q"):
                            raise KeyboardInterrupt
                        continue

                    raw_frame_a = last_frames[cam_a].copy()
                    raw_frame_b = last_frames[cam_b].copy()

                    display_frame_a = raw_frame_a.copy()
                    display_frame_b = raw_frame_b.copy()

                    if now < capture_start:
                        remaining = math.ceil(capture_start - now)
                        draw_text(
                            display_frame_a,
                            f"{pair_name} | Start in {remaining}s",
                            color=(0, 255, 255),
                        )
                        draw_text(
                            display_frame_b,
                            f"{pair_name} | Start in {remaining}s",
                            color=(0, 255, 255),
                        )
                    else:
                        draw_text(
                            display_frame_a,
                            f"{pair_name} | {saved_images}/"
                            f"{MAX_IMAGES_PER_CAMERA}",
                            color=(0, 255, 0),
                        )
                        draw_text(
                            display_frame_b,
                            f"{pair_name} | {saved_images}/"
                            f"{MAX_IMAGES_PER_CAMERA}",
                            color=(0, 255, 0),
                        )

                        if now - last_saved >= SAVE_INTERVAL:
                            saved_images += 1

                            filename_a = (
                                dir_a / f"{cam_a}_photo_{saved_images:02d}.png"
                            )
                            filename_b = (
                                dir_b / f"{cam_b}_photo_{saved_images:02d}.png"
                            )

                            cv2.imwrite(str(filename_a), raw_frame_a)
                            cv2.imwrite(str(filename_b), raw_frame_b)

                            last_saved = now
                            print(
                                f"Gespeichert: {filename_a} und {filename_b}"
                            )

                    cv2.imshow(cam_a, display_frame_a)
                    cv2.imshow(cam_b, display_frame_b)

                    if cv2.waitKey(1) == ord("q"):
                        raise KeyboardInterrupt

        print(f"{pair_name} fertig.")

    print("\nAlle Kamerapaare wurden aufgenommen.")

finally:
    cv2.destroyAllWindows()