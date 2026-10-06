"""What is said aloud: the markup a model writes for the eye is not read to the ear."""

import pytest

from jarvis.voice.speech_text import speakable


class TestSpeakable:
    @pytest.mark.parametrize("text, said", [
        ("Hello there.", "Hello there."),
        ("It is 3.5 degrees, so bring a coat!", "It is 3.5 degrees, so bring a coat!"),
        ("**Note:** the file is saved.", "Note: the file is saved."),
        ("This is __really__ important.", "This is really important."),
        ("It is *quite* cold.", "It is quite cold."),
        ("Run `ls -la` first.", "Run ls -la first."),
        ("## Results", "Results"),
        ("> a quoted line", "a quoted line"),
    ])
    def test_emphasis_headings_and_code_marks_are_dropped(self, text, said):
        assert speakable(text) == said

    @pytest.mark.parametrize("text, said", [
        ("- first item", "first item"),
        ("* second item", "second item"),
        ("  + indented item", "indented item"),
        ("• dotted item", "dotted item"),
        ("- **Name:** Jarvis", "Name: Jarvis"),
    ])
    def test_bullets_are_dropped_but_the_item_is_kept(self, text, said):
        assert speakable(text) == said

    def test_numbers_that_are_content_are_kept(self):
        assert speakable("2 * 3 is 6, and 5 - 2 is 3.") == "2 * 3 is 6, and 5 - 2 is 3."
        assert speakable("1. Open the file.") == "1. Open the file."
        assert speakable("-5 degrees") == "-5 degrees"

    def test_a_link_is_its_words_and_a_bare_address_is_its_site(self):
        assert speakable("See [the docs](https://example.com/a/b) for more.") == "See the docs for more."
        assert speakable("Open https://www.example.com/path?q=1 now.") == "Open example.com now."
        assert speakable("It is at https://example.com/a.") == "It is at example.com."
        assert speakable("![a cat](https://x.org/cat.png)") == "a cat"

    def test_a_divider_is_silent(self):
        assert speakable("---") == ""
        assert speakable("* * *") == ""
        assert speakable("***") == ""

    def test_half_a_marker_is_dropped(self):
        # the sentence splitter can cut between the two halves of **bold phrase**
        assert speakable("**Heads up.") == "Heads up."
        assert speakable("the end** of it.") == "the end of it."

    def test_lines_become_one_line(self):
        assert speakable("one\n\ntwo   three") == "one two three"

    def test_empty_stays_empty(self):
        assert speakable("") == ""
        assert speakable("  \n ") == ""
