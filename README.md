# People Flow Detection using Object Tracking & Heatmap Visualization

Counts how many people move **IN** and **OUT** of an area by tracking them across two horizontal lines in a video, and generates a **heatmap** showing where people spent the most time.

Built with YOLOv8 (detection), ByteTrack (tracking) and OpenCV (counting and visualization). Runs in Google Colab.

## Demo

| Output video | Heatmap | Heatmap overlay |
|---|---|---|
| `output_people_flow.mp4` | `heatmap.png` | `heatmap_overlay.png` |

<!-- Add screenshots or a GIF here, e.g. ![Heatmap](heatmap_overlay.png) -->

## Features

- Person detection with a pre-trained YOLOv8 model
- Persistent IDs across frames using ByteTrack
- Two color-coded horizontal lines: green for IN, red for OUT
- Live IN / OUT counters drawn on the output video
- Bounding box and track ID on every person
- Motion heatmap built from bounding-box centers, saved as a standalone image and as an overlay

## Input video

[people-walking.mp4](https://media.roboflow.com/supervision/video-examples/people-walking.mp4) (Roboflow Supervision examples). The notebook downloads it automatically.

## Getting started

1. Open `people_flow_detection.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook).
2. Set the runtime to GPU: Runtime → Change runtime type → T4 GPU.
3. Run all cells from top to bottom.
4. Download the generated video and heatmap images from the last cell.

To run locally instead:

```bash
pip install ultralytics
```

then run the notebook's code cells in Jupyter.

## Detection method

- **Model:** YOLOv8n (Ultralytics, pre-trained on COCO), filtered to class `0` (person), confidence threshold 0.3. Swap in `yolov8s.pt` or `yolov8m.pt` for better accuracy at the cost of speed.
- **Tracker:** ByteTrack via `model.track(..., tracker="bytetrack.yaml", persist=True)`, which assigns each person a persistent ID.
- **Position:** the center of each bounding box, `(cx, cy) = ((x1 + x2) / 2, (y1 + y2) / 2)`.
- **State:** a dictionary `prev_center = {track_id: (cx, cy)}` holds each person's position from the previous frame.

## Line coordinates

Both lines are full-width horizontal lines, so each is defined by a single y-value (pixels). Coordinates were chosen with [PolygonZone](https://polygonzone.roboflow.com/).

| Line | Color | Purpose | y-coordinate |
|---|---|---|---|
| Upper | Green | IN | `LINE_TOP_Y = 162` |
| Lower | Red | OUT | `LINE_BOTTOM_Y = 918` |

Frame size: `1920 x 1080` px

## IN / OUT logic

- **IN:** the person's previous center was above the upper line and the current center is on or below it (`prev_y < LINE_TOP_Y <= cy`). They came from the top and crossed the upper line going down.
- **OUT:** the person's previous center was below the lower line and the current center is on or above it (`prev_y > LINE_BOTTOM_Y >= cy`). They came from the bottom and crossed the lower line going up.
- Each track ID is stored in a set (`counted_in`, `counted_out`), so a person is counted at most once per direction even if the tracker jitters near a line.

## Heatmap

Every frame, each bounding-box center adds 1 to a 2D array the size of the frame. At the end the array is:

1. Gaussian-blurred to smooth the points into regions
2. Normalized to 0-255
3. Colored with the JET colormap (red/yellow = high presence, blue = low)

It is saved as `heatmap.png`, and blended 50/50 with the first video frame as `heatmap_overlay.png`.

## Results

| Metric | Value |
|---|---|
| IN count | `13` |
| OUT count | `8` |
| Frames processed | `341` |

## Project structure

```
.
├── people_flow_detection.ipynb   # Full pipeline (Colab notebook)
├── output_people_flow.mp4        # Annotated video with live counters
├── heatmap.png                   # Final heatmap
├── heatmap_overlay.png           # Heatmap overlaid on the first frame
└── README.md
```

## Limitations

- Counting uses bounding-box centers, so people partly hidden by others or by the frame edge can shift the center and affect crossings.
- ID switches during occlusion can cause a person to be counted twice or missed.
- Line positions are set manually and must be adjusted for a different camera angle or video.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Roboflow Supervision](https://github.com/roboflow/supervision) for the sample video
- [PolygonZone](https://polygonzone.roboflow.com/) for line coordinates

Author

Nadiya Nowshin
