""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from phonemizer import phonemize
from cvutils import Phonemiser
from .symbols import _punctuation

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners2(text):
    '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True,
                         with_stress=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def latvian_cleaner(text):
    '''Pipeline for Latvian text'''
    text = text.lower()

    text = text.replace('\n', '')
    text = text.replace('–', '—')
    text = text.replace('-', '—')
    text = text.replace('...', '…')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('/', ' ')

    return text


def latvian_cleaner_1(text):
    """Pipeline for Latvian text 1"""
    text = text.replace('–', '-')
    text = text.replace('‘', "'")
    text = text.replace('’', "'")
    text = text.replace('—', '-')

    return text


def latvian_cleaner_advanced(text):
    '''Pipeline for Latvian text with phonemization'''

    text = latvian_cleaner(text)

    p = Phonemiser('lv')
    phonemised_words = []

    for word in text.split(' '):
        word = word.replace('x', 'iks')
        word = word.replace('y', 'igrek')
        word = word.replace('q', 'kū')
        word = word.replace('www', 'vvv')

        if word in _punctuation:
            phonemised_words.append(word)
            continue

        phonemes = p.phonemise(word)

        phonemes = (
            phonemes
            .replace('̪', '')
            .replace('̯̯', '')
            .replace('͡', '')
        )

        phonemised_words.append(phonemes)

    return ' '.join(phonemised_words)
