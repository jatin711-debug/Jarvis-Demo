"""The capability router picks exactly one tool surface per request."""

import pytest

from jarvis.agent.routing import (
    SYSTEM_CAPABILITIES,
    Surface,
    is_follow_up,
    needs_vision,
    needs_web,
    route_request,
    select_system_tools,
)

ALL_SYSTEM_TOOLS = set().union(*SYSTEM_CAPABILITIES.values())


def route(text, *, browser=True, desktop=True, active=False, tools=None, previous=None):
    return route_request(
        text,
        system_tools=ALL_SYSTEM_TOOLS if tools is None else tools,
        browser_available=browser,
        desktop_available=desktop,
        desktop_active=active,
        previous=previous,
    )


class TestSurfaces:
    @pytest.mark.parametrize("text", ["hello", "How are you doing?", "Hey Jarvis, thanks"])
    def test_greetings_take_the_fast_path(self, text):
        assert route(text).surface is Surface.SMALL_TALK

    @pytest.mark.parametrize("text", [
        "search youtube for cats",
        "google the weather in Paris",
        "browse to github.com",
        "look up the news online",
        "open the reddit site",
    ])
    def test_web_requests_use_the_browser(self, text):
        assert route(text).surface is Surface.BROWSER

    @pytest.mark.parametrize("text", ["open notepad", "click the save button", "type hello in the window"])
    def test_app_requests_use_the_desktop(self, text):
        assert route(text).surface is Surface.DESKTOP

    def test_web_wins_over_desktop_words(self):
        assert route("search the web and click a link").surface is Surface.BROWSER

    def test_explanations_offer_no_tools(self):
        result = route("Explain gravity")
        assert result.surface is Surface.CHAT
        assert not result.system_tools

    def test_system_requests_get_only_matching_tools(self):
        result = route("What time is it?", desktop=False, browser=False)
        assert result.surface is Surface.SYSTEM
        assert result.system_tools == {"get_datetime"}
        assert result.groups == ("clock",)

    def test_missing_registries_fall_back_to_system(self):
        result = route("list the files in my downloads", browser=False, desktop=False)
        assert result.surface is Surface.SYSTEM
        assert "list_directory" in result.system_tools


class TestDesktopFollowUps:
    def test_follow_up_stays_on_desktop_only_while_a_task_is_active(self):
        assert route("do that again", active=True).surface is Surface.DESKTOP
        assert route("do that again", active=False).surface is not Surface.DESKTOP

    def test_unrelated_request_leaves_the_desktop(self):
        assert route("what is the weather", active=True).surface is not Surface.DESKTOP

    def test_desktop_needs_a_desktop_backend(self):
        assert route("open notepad", desktop=False).surface is not Surface.DESKTOP


class TestFollowUps:
    """"Now read it back" names no capability; without the last request's tools the model cannot act."""

    def files(self):
        return route("Create a file named todo.txt in the folder docs containing: buy milk",
                     desktop=False, browser=False)

    @pytest.mark.parametrize("text", [
        "Now read it back to me.", "and delete that", "do it again", "what is in there",
        "okay, the second one", "read it back", "Then move them somewhere else",
    ])
    def test_short_references_to_the_last_request_are_follow_ups(self, text):
        assert is_follow_up(text)

    @pytest.mark.parametrize("text", [
        "Explain gravity", "What is 17 times 23?", "Tell me a joke", "Who wrote Hamlet",
        "Explain how photosynthesis works in plants and why the leaves turn red in autumn it is odd",
    ])
    def test_new_questions_are_not_follow_ups(self, text):
        assert not is_follow_up(text)

    def test_a_follow_up_keeps_the_tools_of_the_last_request(self):
        first = self.files()
        assert first.surface is Surface.SYSTEM
        second = route("Now read it back to me.", desktop=False, browser=False, previous=first)
        assert second.surface is Surface.SYSTEM
        assert "read_file" in second.system_tools and "write_file" in second.system_tools
        assert second.groups == first.groups

    def test_without_a_previous_request_it_is_just_chat(self):
        assert route("Now read it back to me.", desktop=False, browser=False).surface is Surface.CHAT

    def test_a_write_verb_after_files_were_in_play_is_unambiguous(self):
        listing = route("list the files in my downloads", desktop=False, browser=False)
        assert "delete_file" not in listing.system_tools
        deleting = route("delete it", desktop=False, browser=False, previous=listing)
        assert deleting.surface is Surface.SYSTEM
        assert {"delete_file", "list_directory"} <= deleting.system_tools

    def test_it_carries_on_through_a_chain_of_follow_ups(self):
        first = route("list the files in my downloads", desktop=False, browser=False)
        second = route("now delete it", desktop=False, browser=False, previous=first)
        third = route("and move that too", desktop=False, browser=False, previous=second)
        assert {"list_directory", "delete_file", "move_file"} <= third.system_tools

    def test_a_chat_answer_in_between_ends_the_chain(self):
        files = self.files()
        chat = route("Explain gravity", desktop=False, browser=False, previous=files)
        assert chat.surface is Surface.CHAT
        assert route("tell me more about it", desktop=False, browser=False,
                     previous=chat).surface is Surface.CHAT

    def test_a_request_with_its_own_capability_is_not_widened(self):
        result = route("what time is it", desktop=False, browser=False, previous=self.files())
        assert result.groups == ("clock",) and result.system_tools == {"get_datetime"}

    def test_a_long_new_question_does_not_inherit_tools(self):
        question = "Explain how photosynthesis works in plants and why the leaves turn red in autumn it is odd"
        assert route(question, desktop=False, browser=False,
                     previous=self.files()).surface is Surface.CHAT

    def test_a_browser_follow_up_stays_in_the_browser(self):
        searched = route("search youtube for cats", desktop=False)
        assert searched.surface is Surface.BROWSER
        assert route("play the first one", desktop=False, previous=searched).surface is Surface.BROWSER
        assert route("play the first one", desktop=False, browser=False,
                     previous=searched).surface is not Surface.BROWSER

    def test_small_talk_stays_small_talk(self):
        assert route("thanks", previous=self.files()).surface is Surface.SMALL_TALK


class TestWebDetection:
    @pytest.mark.parametrize("text", [
        "what time is it", "open notepad", "list the files in my downloads",
        "set volume to 30", "take a screenshot", "search the files in this project",
    ])
    def test_local_requests_are_not_web(self, text):
        assert not needs_web(text)

    def test_files_on_github_are_web(self):
        assert needs_web("find the files on github")


class TestVision:
    @pytest.mark.parametrize("text", [
        "what's on my screen", "take a screenshot", "click the OK button",
        "where is the taskbar", "can you see this error message",
    ])
    def test_visual_requests(self, text):
        assert needs_vision(text)

    @pytest.mark.parametrize("text", [
        "what time is it", "open notepad", "search youtube for music", "set volume to 50",
    ])
    def test_non_visual_requests(self, text):
        assert not needs_vision(text)

    @pytest.mark.parametrize("text", [
        "Find the file report-2026.txt somewhere inside the folder docs",
        "Read the file C:/Windows/win.ini and show me what it says.",
        "look at the files in my downloads", "where is my project folder", "locate the repo",
    ])
    def test_file_requests_never_take_a_screenshot(self, text):
        assert not needs_vision(text)

    @pytest.mark.parametrize("text", [
        "find the save button", "look at this", "show me my screen", "where is the start menu icon",
        "click the blue one",
    ])
    def test_looking_phrases_still_mean_the_screen_when_no_file_is_named(self, text):
        assert needs_vision(text)

    def test_desktop_route_skips_screenshots(self):
        assert not route("click the save button").vision

    def test_screen_question_is_chat_with_vision(self):
        result = route("what's on my screen", desktop=False, browser=False)
        assert result.surface is Surface.CHAT
        assert result.vision


class TestSystemSelection:
    def test_write_verbs_alone_do_not_offer_file_writes(self):
        tools, groups = select_system_tools("delete it", ALL_SYSTEM_TOOLS)
        assert "files_write" not in groups
        assert "delete_file" not in tools

    def test_write_verbs_with_file_words_offer_file_writes(self):
        tools, groups = select_system_tools("create a new file", ALL_SYSTEM_TOOLS)
        assert {"files_read", "files_write"} <= set(groups)
        assert "write_file" in tools

    def test_only_available_tools_are_returned(self):
        tools, _ = select_system_tools("read the file", {"read_file"})
        assert tools == {"read_file"}

    def test_describe_is_a_compact_log_line(self):
        text = route("What time is it?", desktop=False, browser=False).describe()
        assert text == "surface=system groups=clock system_tools=1"
