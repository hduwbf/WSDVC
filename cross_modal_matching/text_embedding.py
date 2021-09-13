import re
from typing import List


def preprocess_bert_paragraph(paragraph: List[str]) -> List[List[str]]:
    new_paragraph = []
    for i, sentence in enumerate(paragraph):
        new_sentence = []
        if i == 0:
            new_sentence.append("[CLS]")
        preproc_sentence = preprocess_bert_sentence(sentence)
        for word in preproc_sentence:
            new_sentence.append(word.strip())
        new_paragraph.append(new_sentence)
    return new_paragraph


def preprocess_bert_sentence(sentence_str: str) -> List[str]:
    if sentence_str[-1] == ".":
        sentence_str = sentence_str[:-1]
    sentence_str = sentence_str.replace(". ", " [SEP] ")
    sentence_str += " [SEP] "
    sentence_str = re.sub(r"\s+", " ", sentence_str).strip()
    words = sentence_str.split(" ")
    return words