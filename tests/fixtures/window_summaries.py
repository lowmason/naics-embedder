'''
Stubs for the window-fitting summaries tests (roadmap Stage 6b): a token counter that counts
words.
'''

from typing import List, Sequence

def words(texts: Sequence[str]) -> List[int]:
    '''A stub token counter: one token per whitespace-separated word.'''

    return [len(text.split()) for text in texts]
