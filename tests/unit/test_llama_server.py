"""Launching llama-server: the command line, and a real child process that is started and stopped."""

import socket
import sys
import textwrap

import httpx
import pytest

from jarvis import doctor
from jarvis.agent.llama_server import LlamaServer, build_command, server_address
from jarvis.agent.llm_client import LLMClient
from jarvis.utils.config import JarvisConfig, LLMConfig

# Stands in for llama-server: answers /health and /props on --port, ignores other flags.
FAKE_SERVER = textwrap.dedent("""
    import argparse, json, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--alias")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--die", action="store_true")
    args, _ = parser.parse_known_args()
    if args.die:
        print("error: invalid argument: --nonsense", flush=True)
        raise SystemExit(2)
    time.sleep(args.delay)

    class Handler(BaseHTTPRequestHandler):
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
