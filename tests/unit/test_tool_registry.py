"""Unit tests for jarvis.agent.tool_registry."""

import pytest

from jarvis.agent.tool_registry import ToolDefinition, ToolRegistry, validate_arguments


def _make_tool(**overrides) -> ToolDefinition:
    async def handler(**kwargs):
        return kwargs

    spec = {
        "name": "demo",
        "description": "demo tool",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "label": {"type": "string"},
            },
            "required": ["count"],
        },
        "handler": handler,
    }
    spec.update(overrides)
    return ToolDefinition(**spec)


class TestValidateArguments:
    def test_int_coerced_from_string(self):
        args, err = validate_arguments(_make_tool(), {"count": "5"})
        assert err is None
        assert args["count"] == 5

    def test_int_rejects_garbage(self):
        _, err = validate_arguments(_make_tool(), {"count": "five"})
        assert err is not None and "integer" in err

    def test_number_coerced(self):
        args, err = validate_arguments(_make_tool(), {"count": 1, "ratio": "2.5"})
        assert err is None
        assert args["ratio"] == 2.5

    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("1", True), ("yes", True),
        ("false", False), ("0", False), ("no", False),
    ])
    def test_bool_coercion(self, raw, expected):
        args, err = validate_arguments(_make_tool(), {"count": 1, "flag": raw})
        assert err is None
        assert args["flag"] is expected

    def test_bool_rejects_garbage(self):
        _, err = validate_arguments(_make_tool(), {"count": 1, "flag": "maybe"})
        assert err is not None and "boolean" in err

    def test_string_coerced_from_number(self):
        args, err = validate_arguments(_make_tool(), {"count": 1, "label": 42})
        assert err is None
        assert args["label"] == "42"

    def test_missing_required(self):
        _, err = validate_arguments(_make_tool(), {})
        assert err is not None and "Missing required" in err

    def test_none_counts_as_missing(self):
        _, err = validate_arguments(_make_tool(), {"count": None})
        assert err is not None

    def test_unknown_params_pass_through(self):
        args, err = validate_arguments(_make_tool(), {"count": 1, "extra": "kept"})
        assert err is None
        assert args["extra"] == "kept"


class TestRegistryExecution:
    async def test_execute_returns_handler_result(self):
        registry = ToolRegistry()

        @registry.tool(name="greet", description="greet",
                       parameters={"type": "object", "properties": {}})
        async def greet():
            return "hello"

        assert await registry.execute("greet") == "hello"

    async def test_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            await registry.execute("nope")

    async def test_sync_function_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(TypeError, match="async"):
            @registry.tool(name="bad", description="x",
                           parameters={"type": "object", "properties": {}})
            def bad():
                return None

    async def test_confirmation_denied_cancels(self):
        async def deny(name, args):
            return False

        registry = ToolRegistry(confirm_callback=deny)
        called = False

        @registry.tool(name="danger", description="x", requires_confirmation=True,
                       parameters={"type": "object", "properties": {}})
        async def danger():
            nonlocal called
            called = True
            return "done"

        result = await registry.execute("danger")
        assert result == "Action cancelled by the user."
        assert called is False

    async def test_confirmation_approved_runs(self):
        async def approve(name, args):
            return True

        registry = ToolRegistry(confirm_callback=approve)

        @registry.tool(name="danger", description="x", requires_confirmation=True,
                       parameters={"type": "object", "properties": {}})
        async def danger():
            return "done"

        assert await registry.execute("danger") == "done"

    async def test_no_callback_means_no_confirmation(self):
        # Tools marked dangerous run without prompting when no callback wired.
        registry = ToolRegistry()

        @registry.tool(name="danger", description="x", requires_confirmation=True,
                       parameters={"type": "object", "properties": {}})
        async def danger():
            return "done"

        assert await registry.execute("danger") == "done"

    async def test_invalid_args_return_error_string(self):
        registry = ToolRegistry()

        @registry.tool(
            name="vol", description="x",
            parameters={
                "type": "object",
                "properties": {"level": {"type": "integer"}},
                "required": ["level"],
            },
        )
        async def vol(level: int):
            return level

        result = await registry.execute("vol", level="loud")
        assert isinstance(result, str) and "Invalid arguments" in result

    def test_ollama_schema_shape(self):
        registry = ToolRegistry()

        @registry.tool(
            name="greet", description="Say hi",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        async def greet(name: str = ""):
            return name

        (schema,) = registry.get_ollama_tools()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert schema["function"]["description"] == "Say hi"
        assert "name" in schema["function"]["parameters"]["properties"]
