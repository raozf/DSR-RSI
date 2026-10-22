"""Acquire images only; semantic scene estimation remains an explicit input."""
import time
from .config import bounded
from .executor import utc_now, write_json
from .model import PreparationError


def capture(config, output):
    spec = config.data.get("camera", {})
    kind = spec.get("kind")
    if kind not in {"opencv", "realsense"}:
        raise PreparationError("Set camera.kind to opencv or realsense and fill its device/serial in local/site.json")
    width = int(bounded(spec.get("width"), 320, 4096, "camera.width", True))
    height = int(bounded(spec.get("height"), 240, 2160, "camera.height", True))
    import cv2
    rgb, depth, meta = None, None, {"source": kind, "captured_at": utc_now(), "calibration_id": config.calibration["calibration_id"]}
    if kind == "opencv":
        device = spec.get("device")
        if not isinstance(device, (str, int)) or isinstance(device, bool) or device == "":
            raise PreparationError("Fill camera.device (e.g. /dev/v4l/by-id/...) before capture")
        cam = cv2.VideoCapture(device)
        try:
            if not cam.isOpened():
                raise PreparationError("Cannot open camera: "+str(device))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(5):
                ok, rgb = cam.read()
                if not ok:
                    raise PreparationError("Camera did not return an RGB frame")
        finally:
            cam.release()
        meta["device"] = str(device)
    else:
        serial = spec.get("serial")
        if not isinstance(serial, str) or not serial:
            raise PreparationError("Fill the RealSense camera serial explicitly")
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise PreparationError("Install pyrealsense2 in xtrainer_real/.venv for RealSense capture") from exc
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(serial)
        rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        started = False
        try:
            profile = pipeline.start(rs_config)
            started = True
            aligned = rs.align(rs.stream.color)
            for _ in range(5):
                frames = aligned.process(pipeline.wait_for_frames(3000))
            color, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth_frame:
                raise PreparationError("RealSense did not return aligned RGB/depth")
            import numpy as np
            rgb, depth = np.asanyarray(color.get_data()), np.asanyarray(depth_frame.get_data())
            k = color.profile.as_video_stream_profile().intrinsics
            meta.update(serial=serial, intrinsics={"fx": k.fx, "fy": k.fy, "cx": k.ppx, "cy": k.ppy,
                                                   "width": k.width, "height": k.height, "distortion": list(k.coeffs)},
                        depth_scale_m=profile.get_device().first_depth_sensor().get_depth_scale())
        finally:
            if started:
                pipeline.stop()
    if not cv2.imwrite(str(output/"color.png"), rgb):
        raise PreparationError("Failed to write color image")
    if depth is not None:
        import numpy as np
        np.save(output/"depth_uint16.npy", depth)
    meta["captured_at"] = utc_now()
    write_json(output/"camera.json", meta)
    return meta
