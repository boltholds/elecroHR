"""The HTTP TTS adapter rejects incompatible wire data early."""
import httpx
import pytest

from local_voice.audio import validate_tts_response


def test_pcm24k_wire_response_is_valid():
    response = httpx.Response(
        200, headers={"X-Sample-Rate": "24000"}, content=b"\x00\x00\x01\x00",
        request=httpx.Request("POST", "http://tts:8011/synthesize")
    )
    assert validate_tts_response(response) == b"\x00\x00\x01\x00"


@pytest.mark.parametrize("headers,content", [
    ({"X-Sample-Rate": "22050"}, b"\x00\x00"),
    ({"X-Sample-Rate": "24000"}, b""),
    ({"X-Sample-Rate": "24000"}, b"\x00"),
])
def test_bad_pcm_is_rejected(headers, content):
    with pytest.raises(ValueError):
        validate_tts_response(httpx.Response(
            200, headers=headers, content=content,
            request=httpx.Request("POST", "http://tts:8011/synthesize"),
        ))


def test_tts_http_error_is_raised():
    with pytest.raises(httpx.HTTPStatusError):
        validate_tts_response(httpx.Response(
            503, headers={"X-Sample-Rate": "24000"}, content=b"\x00\x00",
            request=httpx.Request("POST", "http://tts:8011/synthesize"),
        ))



def test_tts_client_collects_server_inference_header():
    from local_voice.audio import AudioModels

    def handler(request):
        return httpx.Response(
            200, request=request, content=b"\x01\x00",
            headers={
                "X-Sample-Rate": "24000", "X-TTS-Model-Ms": "35.25",
            }
        )

    audio = AudioModels.__new__(AudioModels)
    audio.tts_url = "http://tts:8011/synthesize"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        audio._http = client
        pcm, timings = audio.synthesize_timed("Здравствуйте.")
    assert pcm == b"\x01\x00"
    assert timings["tts_model_ms"] == 35.25


def test_tts_client_remains_compatible_with_server_without_timing_header():
    from local_voice.audio import AudioModels

    def handler(request):
        return httpx.Response(
            200, request=request, content=b"\x01\x00",
            headers={"X-Sample-Rate": "24000"}
        )

    audio = AudioModels.__new__(AudioModels)
    audio.tts_url = "http://tts:8011/synthesize"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        audio._http = client
        pcm, timings = audio.synthesize_timed("Hello")
    assert pcm == b"\x01\x00"
    assert timings == {}
