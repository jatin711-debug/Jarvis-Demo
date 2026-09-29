"""Unit tests for tool-call parsing in jarvis.agent.tool_calls."""

from jarvis.agent.tool_calls import normalize_native_calls, parse_func_call, parse_tool_call


class TestParseFuncCall:
    def test_no_args(self):
        result = parse_func_call("get_datetime()")
        assert result == {"tool": "get_datetime", "args": {}}

    def test_named_string_arg(self):
        result = parse_func_call('open_application(app_name="notepad")')
        assert result["tool"] == "open_application"
        assert result["args"] == {"app_name": "notepad"}

    def test_single_quoted_arg(self):
        result = parse_func_call("open_url(url='https://example.com')")
        assert result["args"] == {"url": "https://example.com"}

    def test_int_and_float_coercion(self):
        result = parse_func_call("foo(level=50, ratio=2.5)")
        assert result["args"] == {"level": 50, "ratio": 2.5}

    def test_bool_and_none_coercion(self):
        result = parse_func_call("foo(a=true, b=False, c=null)")
        assert result["args"] == {"a": True, "b": False, "c": None}

    def test_positional_arg_collected(self):
        result = parse_func_call('open_application("notepad")')
        assert result["args"] == {}
        assert result["_positional"] == ["notepad"]

    def test_mixed_positional_and_named(self):
        result = parse_func_call('foo("bar", count=3)')
        assert result["args"] == {"count": 3}
        assert result["_positional"] == ["bar"]

    def test_escaped_quotes_in_string(self):
        result = parse_func_call('type_text(text="say \\"hi\\"")')
        assert result["args"] == {"text": 'say "hi"'}

    def test_not_a_call_returns_none(self):
        assert parse_func_call("just some text") is None


class TestParseToolCall:
    def test_lfm_bracket_format(self):
        text = "<|tool_call_start|>[get_datetime()]<|tool_call_end|>"
        result = parse_tool_call(text)
        assert result == {"tool": "get_datetime", "args": {}}

    def test_lfm_with_args(self):
        text = '<|tool_call_start|>[open_application(app_name="calc")]<|tool_call_end|>'
        result = parse_tool_call(text)
        assert result["tool"] == "open_application"
        assert result["args"] == {"app_name": "calc"}

    def test_lfm_without_brackets(self):
        text = '<|tool_call_start|>run_command(command="dir")<|tool_call_end|>'
        result = parse_tool_call(text)
        assert result["tool"] == "run_command"
        assert result["args"] == {"command": "dir"}

    def test_json_code_block_tool_key(self):
        text = '```json\n{"tool": "get_datetime", "args": {}}\n```'
        result = parse_tool_call(text)
        assert result["tool"] == "get_datetime"

    def test_json_name_arguments_shape(self):
        text = '{"name": "open_url", "arguments": {"url": "https://example.com"}}'
        result = parse_tool_call(text)
        assert result["tool"] == "open_url"
        assert result["args"] == {"url": "https://example.com"}

    def test_json_native_function_shape(self):
        text = '{"function": {"name": "set_volume", "arguments": "{\\"level\\": 30}"}}'
        result = parse_tool_call(text)
        assert result["tool"] == "set_volume"
        assert result["args"] == {"level": 30}

    def test_tool_name_with_trailing_parens_cleaned(self):
        text = '```json\n{"tool": "get_datetime()", "args": {}}\n```'
        result = parse_tool_call(text)
        assert result["tool"] == "get_datetime"

    def test_plain_text_returns_none(self):
        assert parse_tool_call("Hello, how can I help you today?") is None

    def test_surrounding_prose_ignored(self):
        text = (
            "Let me check that for you.\n"
            "```json\n{\"tool\": \"get_datetime\", \"args\": {}}\n```"
        )
        result = parse_tool_call(text)
        assert result["tool"] == "get_datetime"


class TestNormalizeNativeCalls:
    def test_dict_arguments(self):
        calls = [{"function": {"name": "get_datetime", "arguments": {}}}]
        assert normalize_native_calls(calls) == [{"tool": "get_datetime", "args": {}}]

    def test_string_arguments_parsed(self):
        calls = [{"function": {"name": "set_volume", "arguments": '{"level": 10}'}}]
        assert normalize_native_calls(calls) == [{"tool": "set_volume", "args": {"level": 10}}]

    def test_invalid_string_arguments_become_empty(self):
        calls = [{"function": {"name": "f", "arguments": "not json"}}]
        assert normalize_native_calls(calls) == [{"tool": "f", "args": {}}]

    def test_missing_name_skipped(self):
        calls = [{"function": {"arguments": {}}}]
        assert normalize_native_calls(calls) == []
