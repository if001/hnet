import argparse
import codecs
import dataclasses
import json
import queue
import sys
import time
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

# Ensure project root is importable when run as:
#   python scripts/hnet_openai_server.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate import generate as hnet_generate
from generate import load_from_pretrained
from hnet.training.data import DefaultRecordFormatter
from hnet.utils.tokenizers import ByteTokenizer


@dataclasses.dataclass
class GenerationTask:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    done_event: threading.Event
    result_text: str = ""
    error: Exception | None = None


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


def sample_next_token(logits, temperature: float, top_p: float) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    scaled_logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scaled_logits = scaled_logits.clone()
        scaled_logits[indices_to_remove] = -float("inf")

    probs = torch.softmax(scaled_logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_batch(model, tasks: list[GenerationTask]) -> None:
    if not tasks:
        return

    tokenizer = ByteTokenizer()
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_idx

    sequences = [
        list(tokenizer.encode([task.prompt], add_bos=True)[0]["input_ids"])
        for task in tasks
    ]
    batch_size = len(sequences)
    max_new_tokens = max(task.max_tokens for task in tasks)

    decoders = [
        codecs.getincrementaldecoder("utf-8")(errors="replace")
        for _ in range(batch_size)
    ]
    chunks_per_task: list[list[str]] = [[] for _ in range(batch_size)]
    generated_count = [0 for _ in range(batch_size)]
    finished = [False for _ in range(batch_size)]

    for _ in range(max_new_tokens):
        if all(finished):
            break

        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=eos_id,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        for row, seq in enumerate(sequences):
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
            input_ids[row, : seq_tensor.numel()] = seq_tensor
            mask[row, : seq_tensor.numel()] = True

        with torch.inference_mode():
            output = model.forward(input_ids, mask=mask)

        row_indices = torch.arange(batch_size, device=device)
        last_positions = torch.tensor(lengths, dtype=torch.long, device=device) - 1
        logits = output.logits[row_indices, last_positions, :]

        for index, task in enumerate(tasks):
            if finished[index]:
                continue

            next_token_id = sample_next_token(
                logits[index], temperature=task.temperature, top_p=task.top_p
            )

            if next_token_id == eos_id:
                finished[index] = True
                continue

            sequences[index].append(next_token_id)
            generated_count[index] += 1

            token_chunk = decoders[index].decode(bytes([next_token_id]), final=False)
            if token_chunk:
                chunks_per_task[index].append(token_chunk)

            if generated_count[index] >= task.max_tokens:
                finished[index] = True

    for index, task in enumerate(tasks):
        tail = decoders[index].decode(b"", final=True)
        if tail:
            chunks_per_task[index].append(tail)
        task.result_text = "".join(chunks_per_task[index])


def generation_worker(
    *,
    worker_id: int,
    model,
    request_queue: queue.Queue,
    batch_size: int,
    batch_wait_ms: int,
) -> None:
    del worker_id  # reserved for future worker-specific logging
    while True:
        first_task = request_queue.get()
        if first_task is None:
            request_queue.task_done()
            return

        tasks: list[GenerationTask] = [first_task]
        wait_seconds = max(batch_wait_ms, 0) / 1000.0
        deadline = time.time() + wait_seconds

        while len(tasks) < batch_size:
            timeout = deadline - time.time()
            if timeout <= 0:
                break
            try:
                next_task = request_queue.get(timeout=timeout)
            except queue.Empty:
                break

            if next_task is None:
                request_queue.task_done()
                request_queue.put(None)
                break

            tasks.append(next_task)

        try:
            generate_batch(model, tasks)
        except Exception as exc:  # noqa: BLE001
            for task in tasks:
                task.error = exc
        finally:
            for task in tasks:
                task.done_event.set()
                request_queue.task_done()


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

    def _generate_text(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> str:
        task = GenerationTask(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            done_event=threading.Event(),
        )
        self.server.request_queue.put(task)
        task.done_event.wait()
        if task.error is not None:
            raise task.error
        return task.result_text

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
                {
                    "error": {
                        "message": f"Unknown path: {self.path}",
                        "type": "not_found",
                    }
                },
            )
            return

        try:
            payload = self._read_json_body()
        except json.JSONDecodeError as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "message": f"Invalid JSON: {exc}",
                        "type": "invalid_request_error",
                    }
                },
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
                    {
                        "error": {
                            "message": "messages must be a list",
                            "type": "invalid_request_error",
                        }
                    },
                )
                return
            if self.server.raw_prompt:
                inference_prompt = "\n".join(
                    str(message.get("content", "")) for message in messages
                )
            else:
                inference_prompt = self._build_chat_prompt(messages)

        if not inference_prompt:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "message": "prompt is empty",
                        "type": "invalid_request_error",
                    }
                },
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
    parser = argparse.ArgumentParser(
        description="Serve H-Net model with OpenAI-compatible endpoints"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="hnet-local")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-active-generations",
        "--max-concurrent",
        dest="max_active_generations",
        type=int,
        default=1,
        help="Maximum number of model instances / generation workers (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum requests merged into one generation batch per worker (default: 8).",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=int,
        default=5,
        help="Batching window in milliseconds to wait for additional requests (default: 5).",
    )
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
    if args.max_active_generations < 1:
        raise ValueError("--max-active-generations/--max-concurrent must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_wait_ms < 0:
        raise ValueError("--batch-wait-ms must be >= 0")

    print(f"loading_model=true instances={args.max_active_generations}")
    models = []
    for index in range(args.max_active_generations):
        print(f"loading_model_instance={index + 1}/{args.max_active_generations}")
        model = load_from_pretrained(
            args.model_path,
            args.config_path,
            requested_dtype=args.dtype,
        )
        models.append(model)
    print("loading_model=false")

    if not args.skip_warmup:
        for index, model in enumerate(models, start=1):
            print(f"warmup_started=true instance={index}/{len(models)}")
            generated_tokens = run_warmup(
                model=model,
                prompt=args.warmup_prompt,
                max_tokens=args.warmup_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(
                f"warmup_finished=true instance={index}/{len(models)} generated_tokens={generated_tokens}"
            )

    server = ThreadingHTTPServer((args.host, args.port), HNetOpenAIHandler)
    server.request_queue = queue.Queue()
    server.workers = []
    for worker_id, model in enumerate(models):
        worker = threading.Thread(
            target=generation_worker,
            kwargs={
                "worker_id": worker_id,
                "model": model,
                "request_queue": server.request_queue,
                "batch_size": args.batch_size,
                "batch_wait_ms": args.batch_wait_ms,
            },
            daemon=True,
        )
        worker.start()
        server.workers.append(worker)
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
        f"raw_prompt={args.raw_prompt} think_mode={args.think_mode} "
        f"max_active_generations={args.max_active_generations} "
        f"batch_size={args.batch_size} batch_wait_ms={args.batch_wait_ms}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for _ in server.workers:
            server.request_queue.put(None)
        for worker in server.workers:
            worker.join()
        server.server_close()
        print("server_stopped=true")


if __name__ == "__main__":
    main()
