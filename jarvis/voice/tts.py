"""Text-to-speech via edge-tts — returns MP3 bytes (no files, no API key)."""
from __future__ import annotations

import edge_tts


async def synthesize(text: str, voice: str = "en-GB-RyanNeural") -> bytes:
    """Synthesize `text` to MP3 bytes using an edge-tts neural voice."""
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)
