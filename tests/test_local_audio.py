"""The HTTP TTS adapter rejects incompatible wire data early."""
import httpx
import pytest

from local_voice.audio import validate_tts_response


def test_pcm24k_wire_response_is_valid():
    response = httpx.Response(
        200, headers={"X-Sample-Rate": "24000"}, content=b"\x00\x00\x01\x00"
    )
    assert validate_tts_response(response) == b"\x00\x00\x01\x00"


@pytest.mark.parametrize("headers,content", [
    ({"X-Sample-Rate": "22050"}, b"\x00\x00"),
    ({"X-Sample-Rate": "24000"}, b""),
    ({"X-Sample-Rate": "24000"}, b"\x00"),
])
def test_bad_pcm_is_rejected(headers, content):
    with pytest.raises(ValueError):
        validate_tts_response(httpx.Response(200, headers=headers, content=content))


def test_tts_http_error_is_raised():
    with pytest.raises(httpx.HTTPStatusError):
        validate_tts_response(httpx.Response(
            503, headers={"X-Sample-Rate": "24000"}, content=b"\x00\x00",
            request=httpx.Request("POST", "http://tts:8011/synthesize"),
        ))
