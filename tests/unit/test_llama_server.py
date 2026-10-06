"""Launching llama-server: the command line, and a real child process that is started and stopped."""

import socket
import sys
import textwrap

import httpx
import psutil
import pytest

from jarvis import doctor
from jarvis.agent.llama_server import LlamaServer, build_command, server_address
from jarvis.agent.llm_client import LLMClient
from jarvis.utils.config import JarvisConfig, LLMConfig

# Stands in for llama-server: answers /health and /props on --port, streams "hel" + "lo" from
# /v1/chat/completions, ignores other flags. --crash-marker/--crash-mode make the first chat
# request kill the server (before any output, or after "hel"); --late-marker makes every launch
# after the first load slowly.
FAKE_SERVER = textwrap.dedent("""
    import argparse, json, os, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--alias")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--die", action="store_true")
    parser.add_argument("--crash-marker")
    parser.add_argument("--crash-mode", choices=["before", "mid"], default="before")
    parser.add_argument("--late-marker")
    parser.add_argument("--late-delay", type=float, default=30.0)
    args, _ = parser.parse_known_args()
    if args.die:
        print("error: invalid argument: --nonsense", flush=True)
        raise SystemExit(2)
    if args.late_marker:
        if os.path.exists(args.late_marker):
            time.sleep(args.late_delay)
        else:
            open(args.late_marker, "w").close()
    time.sleep(args.delay)

    def event(content=None, finish=None):
        delta = {} if content is None else {"content": content}
        chunk = {"choices": [{"delta": delta, "finish_reason": finish}]}
        return ("data: " + json.dumps(chunk) + "\\n\\n").encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            crash = args.crash_marker and not os.path.exists(args.crash_marker)
            if crash:
                open(args.crash_marker, "w").close()
                if args.crash_mode == "before":
                    os._exit(1)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(event("hel"))
            self.wfile.flush()
            if crash:
                os._exit(1)
            self.wfile.write(event("lo"))
            self.wfile.write(event(finish="stop"))
            self.wfile.write(b"data: [DONE]\\n\\n")

        def do_GET(self):
            payload = {"status": "ok"}
            if self.path == "/props":
                payload = {"model_alias": args.alias, "modalities": {"vision": False},
                           "default_generation_settings": {"n_ctx": 4096}}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    HTTPServer((args.host, args.port), Handler).serve_forever()
""")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def setup(tmp_path):
    """A fake server script, a fake model file, and a config that launches them."""
    script = tmp_path / "fake_llama_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    port = free_port()

    def make(**server) -> LLMConfig:
        server = {"command": sys.executable, "args": [str(script)], "model": str(model),
                  "startup_timeout": 20, **server}
        return LLMConfig(provider="llamacpp", model="qwen3.5-4b", context_window=4096,
                         base_url=f"http://127.0.0.1:{port}", server=server)

    return make, port, tmp_path


class TestBuildCommand:
    def test_managed_flags_come_after_user_flags_so_they_win(self, setup):
        make, port, _ = setup
        command = build_command(make(args=["-ngl", "99", "-fa", "on"]))
        assert command[1:5] == ["-ngl", "99", "-fa", "on"]
        tail = command[5:]
        assert tail[tail.index("--port") + 1] == str(port)
        assert tail[tail.index("--host") + 1] == "127.0.0.1"
        assert tail[tail.index("-c") + 1] == "4096"
        assert tail[tail.index("--alias") + 1] == "qwen3.5-4b"
        assert tail[tail.index("-np") + 1] == "1"
        assert "--mmproj" not in command

    def test_projector_is_added_only_when_configured_and_present(self, setup):
        make, _, tmp_path = setup
        projector = tmp_path / "mmproj.gguf"
        projector.write_bytes(b"x")
        command = build_command(make(mmproj=str(projector)))
        assert command[command.index("--mmproj") + 1] == str(projector.resolve())
        with pytest.raises(RuntimeError, match="projector not found"):
            build_command(make(mmproj=str(tmp_path / "missing.gguf")))

    def test_missing_model_names_the_file(self, setup):
        make, _, tmp_path = setup
        with pytest.raises(RuntimeError, match="model file not found"):
            build_command(make(model=str(tmp_path / "missing.gguf")))

    def test_missing_binary_says_how_to_fix_it(self, setup):
        make, _, _ = setup
        with pytest.raises(RuntimeError, match="llm.server.command"):
            build_command(make(command="definitely-not-llama-server"))

    def test_no_model_means_nothing_to_launch(self):
        with pytest.raises(RuntimeError, match="llm.server.model"):
            build_command(LLMConfig(provider="llamacpp"))

    def test_the_port_comes_from_base_url(self):
        assert server_address("http://127.0.0.1:9000") == ("127.0.0.1", 9000)
        assert server_address("http://localhost") == ("localhost", 8080)


class TestLifecycle:
    async def test_start_and_stop_a_real_process(self, setup):
        make, port, tmp_path = setup
        server = LlamaServer(make(), tmp_path / "logs" / "llama-server.log")
        await server.start()
        try:
            assert server.alive()
        finally:
            await server.stop()
        assert not server.alive()
        assert (tmp_path / "logs" / "llama-server.log").exists()

    async def test_the_client_launches_waits_and_stops_the_server(self, setup):
        make, port, tmp_path = setup
        client = LLMClient(make(args=[str(tmp_path / "fake_llama_server.py"), "--delay", "1"]),
                           log_dir=tmp_path)
        await client.startup()
        try:
            assert port_open(port)
            assert client._server is not None and client._server.alive()
            assert client.supports_images is False
        finally:
            await client.shutdown()
        assert not port_open(port), "the server Jarvis started was left running"

    async def test_a_server_that_is_already_running_is_used_and_left_alone(self, setup):
        make, port, tmp_path = setup
        first = LLMClient(make(), log_dir=tmp_path)
        await first.startup()
        try:
            second = LLMClient(make(), log_dir=tmp_path)
            await second.startup()
            assert second._server is None, "launched a second server on the same port"
            await second.shutdown()
            assert port_open(port), "shutting down a client stopped a server it did not start"
        finally:
            await first.shutdown()

    async def test_a_server_that_exits_during_startup_is_reported_with_its_output(self, setup):
        make, port, tmp_path = setup
        client = LLMClient(make(args=[str(tmp_path / "fake_llama_server.py"), "--die"]),
                           log_dir=tmp_path)
        with pytest.raises(RuntimeError, match="exited during startup") as caught:
            await client.startup()
        assert "invalid argument: --nonsense" in str(caught.value)
        assert client._server is None and client._client is None

    async def test_a_server_that_never_becomes_ready_is_stopped_and_reported(self, setup):
        make, port, tmp_path = setup
        client = LLMClient(make(args=[str(tmp_path / "fake_llama_server.py"), "--delay", "30"],
                                startup_timeout=1.5), log_dir=tmp_path)
        with pytest.raises(RuntimeError, match="still loading"):
            await client.startup()
        assert not port_open(port)

    async def test_unbuildable_command_fails_startup_cleanly(self, setup):
        make, _, _ = setup
        client = LLMClient(make(command="definitely-not-llama-server"))
        with pytest.raises(RuntimeError, match="llama-server not found"):
            await client.startup()
        assert client._client is None


def crash(client: LLMClient) -> None:
    """Kill the server a client launched, as an out-of-memory kill or a driver reset would."""
    process = psutil.Process(client._server._process.pid)
    family = [*process.children(recursive=True), process]  # the venv launcher has a child
    for member in family:
        member.kill()
    psutil.wait_procs(family, timeout=10)


class TestSelfHealing:
    messages = [{"role": "user", "content": "hi"}]

    async def started(self, setup, *flags: str) -> tuple[LLMClient, int]:
        make, port, tmp_path = setup
        script = str(tmp_path / "fake_llama_server.py")
        client = LLMClient(make(args=[script, *flags]), log_dir=tmp_path)
        await client.startup()
        return client, port

    async def test_a_server_that_died_between_turns_is_restarted(self, setup):
        client, port = await self.started(setup)
        try:
            first_pid = client._server._process.pid
            crash(client)
            assert not port_open(port)
            reply = await client.chat(self.messages)
            assert reply["message"]["content"] == "hello"
            assert client._server.alive() and client._server._process.pid != first_pid
        finally:
            await client.shutdown()
        assert not port_open(port), "the restarted server was left running"

    async def test_a_request_that_loses_the_server_before_any_reply_is_sent_again(self, setup):
        _, _, tmp_path = setup
        marker = tmp_path / "crashed"
        client, port = await self.started(setup, "--crash-marker", str(marker))
        try:
            first_pid = client._server._process.pid
            reply = await client.chat(self.messages)
            assert marker.exists(), "the server never crashed, so nothing was tested"
            assert reply["message"]["content"] == "hello"
            assert client._server._process.pid != first_pid
        finally:
            await client.shutdown()

    async def test_a_reply_cut_off_part_way_is_an_error_not_a_replay(self, setup):
        _, _, tmp_path = setup
        marker = tmp_path / "crashed"
        client, port = await self.started(
            setup, "--crash-marker", str(marker), "--crash-mode", "mid")
        try:
            spoken = []
            with pytest.raises(httpx.TransportError):
                async for event in client.chat_stream_events(self.messages):
                    if event["type"] == "content":
                        spoken.append(event["text"])
            assert spoken == ["hel"], "the part already passed on must not be sent twice"
            reply = await client.chat(self.messages)  # the next request restarts the server
            assert reply["message"]["content"] == "hello"
        finally:
            await client.shutdown()

    async def test_a_server_that_cannot_be_restarted_says_why(self, setup):
        _, _, tmp_path = setup
        client, _ = await self.started(setup)
        try:
            crash(client)
            (tmp_path / "model.gguf").unlink()
            with pytest.raises(RuntimeError, match="model file not found"):
                await client.chat(self.messages)
        finally:
            await client.shutdown()

    async def test_a_restart_that_never_finishes_loading_is_stopped(self, setup):
        make, port, tmp_path = setup
        script = str(tmp_path / "fake_llama_server.py")
        client = LLMClient(make(args=[script, "--late-marker", str(tmp_path / "launched")],
                                startup_timeout=1.5), log_dir=tmp_path)
        await client.startup()
        try:
            crash(client)
            with pytest.raises(RuntimeError, match="still loading"):
                await client.chat(self.messages)
            assert not client._server.alive() and not port_open(port)
        finally:
            await client.shutdown()

    async def test_a_server_somebody_else_started_is_not_taken_over(self, setup):
        make, port, tmp_path = setup
        owner = LLMClient(make(), log_dir=tmp_path)
        await owner.startup()
        guest = LLMClient(make(), log_dir=tmp_path)
        await guest.startup()
        try:
            assert guest._server is None
            crash(owner)
            with pytest.raises(httpx.TransportError):
                await guest.chat(self.messages)
            assert not port_open(port), "the guest started a server it does not own"
        finally:
            await guest.shutdown()
            await owner.shutdown()


class TestDoctor:
    def config(self, llm: LLMConfig) -> JarvisConfig:
        return JarvisConfig(llm=llm)

    async def refuse(self, setup):
        def handler(request):
            raise httpx.ConnectError("refused")

        return httpx.MockTransport(handler)

    async def test_a_launchable_server_that_is_not_running_is_fine(self, setup):
        make, _, _ = setup
        [check] = await doctor.check_llamacpp(self.config(make()), await self.refuse(setup))
        assert check.status == doctor.OK and "Jarvis will start" in check.detail

    async def test_an_unlaunchable_server_is_a_failure_with_the_reason(self, setup):
        make, _, tmp_path = setup
        config = self.config(make(model=str(tmp_path / "missing.gguf")))
        [check] = await doctor.check_llamacpp(config, await self.refuse(setup))
        assert check.status == doctor.FAIL and "model file not found" in check.detail
