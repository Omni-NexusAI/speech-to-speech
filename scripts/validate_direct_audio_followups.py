"""Explicit local WAV follow-up diagnostic, not a microphone/listening test.

Uses the already-running selected Gemma and resident TTS. Does not save settings,
load models, record microphones or write audio unless --audio-dir is supplied.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from collections import Counter
from pathlib import Path

import httpx
import websockets

from synthetic_conversation_realtime_client import load_pcm16_mono_16k, stream_prompt, write_wav


_SAFE_ERROR_TEXT_CODES = {
    "Direct audio model response timed out.": "model_timeout",
}


def _classify_realtime_error(event):
    """Return a stable diagnostic category without exposing provider text."""
    error = event.get("error")
    if not isinstance(error, dict):
        return "unspecified"
    return _SAFE_ERROR_TEXT_CODES.get(error.get("message"), "unspecified")


async def run(args):
    with httpx.Client(timeout=10, trust_env=False) as client:
        health = client.get(args.candidate.rstrip('/') + '/health').raise_for_status().json()
    if health.get('state') != 'loaded' or not health.get('singleResident'):
        raise RuntimeError('Diagnostic requires one already-resident candidate model')
    clips = [load_pcm16_mono_16k(path) for path in args.clips]
    if any(not clip or len(clip) > 30 * 32000 for clip in clips):
        raise RuntimeError('Every explicit test clip must contain at most 30 seconds')
    report = {'model': health['activeModel'], 'engine_epoch': health.get('engineEpoch'),
              'transport': 'synthetic WAV over real-time-paced WebSocket PCM',
              'microphone_verified': False, 'listening_verified': False, 'turns': []}
    async with websockets.connect(args.websocket, max_size=2**24) as ws:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
        if first.get('type') != 'session.created':
            raise RuntimeError('Realtime session was not created')

        async def configure(event, expected):
            await ws.send(json.dumps(event))
            while True:
                response = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                if response.get('type') == 'error':
                    raise RuntimeError('Realtime configuration rejected: ' + str(response.get('error', {}).get('code', 'unknown')))
                if response.get('type') == expected:
                    return

        await configure({'type': 'pipeline.config.update', 'config': {
            'tts_backend': 'qwen3tts-audiocpp', 'full_buffer_tts': False,
            'live_transcription': False, 'max_response_tokens': 128,
            'model_endpoint': {'provider': 'remote', 'base_url': args.gemma,
                               'model': args.model, 'api_key': os.environ.get(args.api_key_env, '')},
        }}, 'pipeline.config.updated')
        await configure({'type': 'session.update', 'session': {
            'type': 'realtime', 'instructions': 'Answer the spoken request briefly and directly.',
            'audio': {'output': {'voice': args.voice}},
        }}, 'session.updated')
        for index, clip in enumerate(clips):
            counts = Counter()
            pcm = bytearray()
            text = []
            metrics = []
            started = time.monotonic()
            send_task = asyncio.create_task(stream_prompt(ws, clip))
            try:
                while True:
                    event = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(.1, 180 - (time.monotonic() - started))))
                    kind = event.get('type', '')
                    counts[kind] += 1
                    if kind == 'error':
                        raise RuntimeError('Realtime diagnostic failed: ' + _classify_realtime_error(event))
                    if kind == 'response.output_audio.delta':
                        pcm.extend(base64.b64decode(event.get('delta', ''), validate=True))
                    if kind == 'response.output_audio_transcript.delta':
                        text.append(event.get('delta', ''))
                    if kind == 'pipeline.metric':
                        metrics.append({key: event[key] for key in ('stage', 'event', 'elapsed_ms', 'response_epoch', 'detail') if key in event})
                    if kind == 'response.done':
                        if event.get('response', {}).get('status') != 'completed' or not pcm:
                            raise RuntimeError('Turn did not complete with audio')
                        break
            finally:
                await send_task
            if len(pcm) % 2:
                raise RuntimeError('Incomplete PCM16 output')
            report['turns'].append({'index': index + 1, 'counts': dict(counts), 'text': ''.join(text),
                                    'bytes': len(pcm), 'elapsed_s': time.monotonic() - started, 'metrics': metrics})
            if args.audio_dir:
                args.audio_dir.mkdir(parents=True, exist_ok=True)
                write_wav(args.audio_dir / f'followup-{index + 1}.wav', bytes(pcm), rate=24000)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Explicitly start local diagnostic generation')
    parser.add_argument('--clips', type=Path, nargs='+', required=True)
    parser.add_argument('--websocket', default='ws://127.0.0.1:8765/v1/realtime')
    parser.add_argument('--candidate', default='http://127.0.0.1:8890')
    parser.add_argument('--gemma', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--voice', required=True)
    parser.add_argument('--api-key-env', default='HFRT_DIAGNOSTIC_GEMMA_KEY')
    parser.add_argument('--audio-dir', type=Path)
    arguments = parser.parse_args()
    if not arguments.execute or not 1 <= len(arguments.clips) <= 6:
        parser.error('Use --execute and one to six explicitly selected local diagnostic clips')
    asyncio.run(run(arguments))
