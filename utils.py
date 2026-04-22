import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

# QWERTY keyboard adjacency map for typo simulation
KEYBOARD_ADJACENT = {
    'a': 'sqwz', 'b': 'vngh', 'c': 'xdfv', 'd': 'sfcerc', 'e': 'rwds',
    'f': 'dgcvr', 'g': 'fhtbv', 'h': 'gjybn', 'i': 'ujko', 'j': 'hkunm',
    'k': 'jloi', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'etdf', 's': 'adwxz', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
    'z': 'asx',
}


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


def get_synonym(word):
    """Get a synonym for the given word using WordNet."""
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    for syn in synsets:
        for lemma in syn.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if candidate.lower() != word.lower():
                return candidate
    return None


def introduce_typo(word):
    """Introduce a typo by replacing a random character with an adjacent key."""
    if len(word) < 3:
        return word
    idx = random.randint(1, len(word) - 1)
    char = word[idx].lower()
    if char in KEYBOARD_ADJACENT:
        adjacent = KEYBOARD_ADJACENT[char]
        replacement = random.choice(adjacent)
        if word[idx].isupper():
            replacement = replacement.upper()
        word = word[:idx] + replacement + word[idx + 1:]
    return word


def custom_transform(example):
    """
    Combined transformation applying three perturbations to simulate
    realistic out-of-distribution text:

    1. Synonym replacement (~35% chance per eligible word): replaces a word
       with a WordNet synonym, preserving meaning but shifting vocabulary.

    2. Keyboard typo injection (~35% chance per eligible word, if synonym
       not applied): swaps one character for an adjacent QWERTY key,
       simulating realistic typing mistakes.

    3. Guaranteed typo floor: after the per-word pass, if fewer than 5
       words were changed in a sentence with >10 tokens, we force-inject
       typos on randomly chosen words. This ensures every sentence is
       noticeably perturbed, which is key to causing a meaningful accuracy
       drop on the model without changing the ground-truth sentiment label.

    Why this is "reasonable": real user-submitted reviews frequently contain
    synonyms (people use different vocabulary for the same sentiment) and
    typos (fast typing on mobile/keyboard). A model deployed in production
    should handle both. The label is preserved because synonym replacement
    maintains semantics and typos do not change the sentiment polarity of a
    review.
    """
    text = example["text"]
    words = word_tokenize(text)

    changes_made = 0
    new_words = []

    for word in words:
        if len(word) <= 2 or not word.isalpha():
            new_words.append(word)
            continue

        r = random.random()

        # 35% chance: synonym replacement
        if r < 0.35:
            synonym = get_synonym(word)
            if synonym is not None:
                new_words.append(synonym)
                changes_made += 1
                continue

        # 35% chance (r in [0.35, 0.70]): typo injection
        if r < 0.70:
            new_words.append(introduce_typo(word))
            changes_made += 1
            continue

        new_words.append(word)

    # Guarantee floor: force at least 5 typos on longer sentences
    if changes_made < 5 and len(words) > 10:
        for _ in range(7):
            idx = random.randint(0, len(new_words) - 1)
            if len(new_words[idx]) > 3 and new_words[idx].isalpha():
                new_words[idx] = introduce_typo(new_words[idx])

    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(new_words)
    return example
