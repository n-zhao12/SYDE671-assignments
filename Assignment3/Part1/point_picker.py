import cv2
import numpy as np


def _display_panel(image: np.ndarray, max_width: int, max_height: int):
	height, width = image.shape[:2]
	scale = min(max_width / width, max_height / height, 1.0)
	display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
	display = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
	return display, scale


def select_correspondences(
	moving_image: np.ndarray,
	reference_image: np.ndarray,
	title: str = "Select corresponding points",
	min_points: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
	"""Click matching points in side-by-side images; return (moving, reference) xy pairs."""
	panel_width = 700
	panel_height = 800
	moving_display, moving_scale = _display_panel(moving_image, panel_width, panel_height)
	reference_display, reference_scale = _display_panel(reference_image, panel_width, panel_height)
	gap = 12
	canvas_height = max(moving_display.shape[0], reference_display.shape[0]) + 72
	moving_origin = (0, 42)
	reference_origin = (moving_display.shape[1] + gap, 42)
	canvas_width = moving_display.shape[1] + reference_display.shape[1] + gap
	moving_points: list[tuple[float, float]] = []
	reference_points: list[tuple[float, float]] = []

	def render() -> np.ndarray:
		canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
		canvas[moving_origin[1]:moving_origin[1] + moving_display.shape[0], :moving_display.shape[1]] = moving_display
		canvas[
			reference_origin[1]:reference_origin[1] + reference_display.shape[0],
		reference_origin[0]:reference_origin[0] + reference_display.shape[1],
		] = reference_display
		cv2.putText(canvas, "MOVING", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(canvas, "REFERENCE", (reference_origin[0] + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(
			canvas, "Click matching numbered points. U: undo pair, Q: finish, Esc: cancel",
			(8, canvas_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
		)
		for index, (x, y) in enumerate(moving_points):
			point = (round(x * moving_scale), round(y * moving_scale + moving_origin[1]))
			cv2.circle(canvas, point, 5, (0, 255, 255), -1)
			cv2.putText(canvas, str(index + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
		for index, (x, y) in enumerate(reference_points):
			point = (
				round(x * reference_scale + reference_origin[0]),
				round(y * reference_scale + reference_origin[1]),
			)
			cv2.circle(canvas, point, 5, (0, 255, 255), -1)
			cv2.putText(canvas, str(index + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
		return canvas

	def on_mouse(event, x, y, _flags, _parameter):
		if event != cv2.EVENT_LBUTTONDOWN or y < 42:
			return
		if x < moving_display.shape[1] and len(moving_points) <= len(reference_points):
			moving_points.append(((x - moving_origin[0]) / moving_scale,
								  (y - moving_origin[1]) / moving_scale))
		elif x >= reference_origin[0] and len(reference_points) <= len(moving_points):
			reference_points.append(((x - reference_origin[0]) / reference_scale,
								 (y - reference_origin[1]) / reference_scale))

	cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
	cv2.setMouseCallback(title, on_mouse)
	try:
		while True:
			cv2.imshow(title, render())
			key = cv2.waitKey(20) & 0xFF
			if key in (ord("q"), 13):
				if len(moving_points) >= min_points and len(moving_points) == len(reference_points):
					return np.asarray(moving_points), np.asarray(reference_points)
				print(f"Select at least {min_points} complete point pairs before finishing.")
			elif key == ord("u"):
				if len(moving_points) > len(reference_points):
					moving_points.pop()
				elif len(reference_points) > len(moving_points):
					reference_points.pop()
				elif moving_points:
					moving_points.pop()
					reference_points.pop()
			elif key == 27:
				raise KeyboardInterrupt("Point selection cancelled")
	finally:
		cv2.destroyWindow(title)


def select_points(
	image: np.ndarray,
	title: str,
	count: int,
) -> np.ndarray:
	"""Click a fixed number of (x, y) points in one image."""
	max_width = 1000
	max_height = 800
	display, scale = _display_panel(image, max_width, max_height)
	points: list[tuple[float, float]] = []

	def render() -> np.ndarray:
		canvas = display.copy()
		for index, (x, y) in enumerate(points):
			point = (round(x * scale), round(y * scale))
			cv2.circle(canvas, point, 5, (0, 255, 255), -1)
			cv2.putText(canvas, str(index + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
		return canvas

	def on_mouse(event, x, y, _flags, _parameter):
		if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
			points.append((x / scale, y / scale))

	cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
	cv2.setMouseCallback(title, on_mouse)
	try:
		while True:
			cv2.imshow(title, render())
			key = cv2.waitKey(20) & 0xFF
			if key in (ord("q"), 13):
				if len(points) == count:
					return np.asarray(points)
				print(f"Select {count - len(points)} more point(s) before finishing.")
			elif key == ord("u") and points:
				points.pop()
			elif key == 27:
				raise KeyboardInterrupt("Point selection cancelled")
	finally:
		cv2.destroyWindow(title)