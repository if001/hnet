import argparse
import codecs
import json
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Ensure project root is importable when run as:
#   python scripts/hnet_openai_server.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate import generate as hnet_generate
from generate import load_from_pretrained
from hnet.training.data import DefaultRecordFormatter


def run_warmup(
    model,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> int:
    generated_tokens = 0
    for _token_id in hnet_generate(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ):
        generated_tokens += 1
    return generated_tokens


class HNetOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "HNetOpenAIServer/0.1"

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _build_chat_prompt(self, messages: list[dict[str, Any]]) -> str:
        formatter = DefaultRecordFormatter()
        rendered = formatter.format_record({"messages": messages})
        if rendered is None:
            return "assistant: "
        return f"{rendered}\nassistant: "

    def _default_system_message(self) -> dict[str, str]:
        control = "/think" if self.server.think_mode else "/no_think"
        system_content = f"{self.server.system_prompt}\n{control}".strip()
        return {"role": "system", "content": system_content}

    def _generate_text(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        chunks: list[str] = []

        for token_id in hnet_generate(
            self.server.model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            chunk = decoder.decode(bytes([token_id]), final=False)
            if chunk:
                chunks.append(chunk)

        tail = decoder.decode(b"", final=True)
        if tail:
            chunks.append(tail)

        return "".join(chunks)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return

        if self.path == "/v1/models":
            self._write_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "hnet-local",
                        }
                    ],
                },
            )
            return

        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": {"message": f"Unknown path: {self.path}", "type": "not_found"}},
        )

    def do_POST(self) -> None:
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": {"message": f"Unknown path: {self.path}", "type": "not_found"}},
            )
            return

        try:
            payload = self._read_json_body()
        except json.JSONDecodeError as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": f"Invalid JSON: {exc}", "type": "invalid_request_error"}},
            )
            return

        max_tokens = int(payload.get("max_tokens", self.server.default_max_tokens))
        temperature = float(payload.get("temperature", self.server.default_temperature))
        top_p = float(payload.get("top_p", self.server.default_top_p))

        if self.path == "/v1/completions":
            prompt = str(payload.get("prompt", ""))
            if self.server.raw_prompt:
                inference_prompt = prompt
            else:
                inference_prompt = self._build_chat_prompt(
                    [
                        self._default_system_message(),
                        {"role": "user", "content": prompt},
                    ]
                )
        else:
            messages = payload.get("messages", [])
            if not isinstance(messages, list):
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": "messages must be a list", "type": "invalid_request_error"}},
                )
                return
            if self.server.raw_prompt:
                inference_prompt = "\n".join(str(message.get("content", "")) for message in messages)
            else:
                inference_prompt = self._build_chat_prompt(messages)

        if not inference_prompt:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "prompt is empty", "type": "invalid_request_error"}},
            )
            return

        generated = self._generate_text(
            prompt=inference_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if self.path == "/v1/completions":
            response = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": self.server.model_name,
                "choices": [
                    {
                        "text": generated,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        else:
            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": self.server.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        self._write_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve H-Net model with OpenAI-compatible endpoints")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="hnet-local")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Inference dtype for model load (default: auto).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt used for chat-style formatting.",
    )
    parser.add_argument(
        "--think-mode",
        action="store_true",
        help="Use '/think' control tag instead of '/no_think'.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Disable chat-style formatting and use raw prompt text.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Disable one-time warmup generation at server startup.",
    )
    parser.add_argument(
        "--warmup-prompt",
        type=str,
        default="warmup",
        help="Prompt text used for one-time warmup generation.",
    )
    parser.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=1,
        help="Number of tokens to generate during startup warmup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("loading_model=true")
    model = load_from_pretrained(
        args.model_path,
        args.config_path,
        requested_dtype=args.dtype,
    )
    print("loading_model=false")

    if not args.skip_warmup:
        print("warmup_started=true")
        generated_tokens = run_warmup(
            model=model,
            prompt=args.warmup_prompt,
            max_tokens=args.warmup_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"warmup_finished=true generated_tokens={generated_tokens}")

    server = ThreadingHTTPServer((args.host, args.port), HNetOpenAIHandler)
    server.model = model
    server.model_name = args.model_name
    server.default_max_tokens = args.max_tokens
    server.default_temperature = args.temperature
    server.default_top_p = args.top_p
    server.system_prompt = args.system_prompt
    server.think_mode = args.think_mode
    server.raw_prompt = args.raw_prompt

    print(
        "server_started=true "
        f"host={args.host} port={args.port} model_name={args.model_name} "
        f"raw_prompt={args.raw_prompt} think_mode={args.think_mode}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("server_stopped=true")


if __name__ == "__main__":
    main()
