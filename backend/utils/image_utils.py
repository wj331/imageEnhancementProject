import cv2
import base64
import io

def numpy_to_base64(img):
    """
    Convert a NumPy array (OpenCV image) to a base64-encoded string.
    """
    # Encode the image into a byte buffer (PNG format)
    _, buffer = cv2.imencode('.png', img)

    # Convert the byte buffer to a base64 string
    base64_str = base64.b64encode(buffer).decode('utf-8')

    return base64_str

def pil_to_base64(img):
    """
    Convert a PIL image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_byte_arr = buffered.getvalue()
    base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
    return base64_str
