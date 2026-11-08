import io
import pytest
from PIL import Image
from experiments.r1.fetch_mfnet import validate_png


def test_complete_png_required():
    buffer = io.BytesIO()
    Image.new('RGB', (16, 16), (5, 10, 20)).save(buffer, format='PNG')
    payload = buffer.getvalue()
    validate_png(payload)
    with pytest.raises((ValueError, OSError, SyntaxError)):
        validate_png(payload[:40])
    with pytest.raises(ValueError):
        validate_png(b'<html>login</html>')
