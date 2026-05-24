import cv2
import depthai as dai

"""
display_all_cameras.py

Dieses Skript zeigt von jeder angeschlossenen Kamera eine Vorschau. Hilfreich
um vorab alle Kameras zu checken.

Ablauf:
1. Alle angeschlossenen Kameras werden erkannt.
2. Für jede Kamera wird eine eigene Aufnahmephase gestartet.

Zweck:
Mit der Vorschau aller Kameraperspektiven können Probleme vorab erkannt werden.
"""


device = dai.Device()

with dai.Pipeline(device) as pipeline:
    outputQueues = {}
    sockets = device.getConnectedCameras()

    for socket in sockets:
        camera_node = pipeline.create(dai.node.Camera).build(socket)

        output = camera_node.requestOutput(
        size=(640, 480),
        type=dai.ImgFrame.Type.BGR888p,
        resizeMode=dai.ImgResizeMode.CROP,
        fps=15
        )

        outputQueues[str(socket)] = output.createOutputQueue()

    pipeline.start()

    while pipeline.isRunning():
        for name, queue in outputQueues.items():
            videoIn = queue.get()
            assert isinstance(videoIn, dai.ImgFrame)
            cv2.imshow(name, videoIn.getCvFrame())

        if cv2.waitKey(1) == ord("q"):
            break