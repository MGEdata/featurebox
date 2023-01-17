# -*- coding: utf-8 -*-

# @Time    : 2021/8/6 16:07
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import re

try:
    from translate import Translator
except ImportError:
    raise ImportError("'translate' should be install. Try ``pip install translate``")


def translate_en_to_zh(words: str):
    translator = Translator(to_lang="zh")
    return translator.translate(words)


def get_msgid(text_lines):
    text = "".join(text_lines)
    text_ms = re.split(text, "msgid")
    text_cuple = [re.split(i, "msgstr") for i in text_ms]
    return text_cuple
    # for i in text_cuple:

    # j.split("\n")


f = open("featurebox.selection.po")
text_ = f.readlines()
get_msgid(text_lines=text_)
