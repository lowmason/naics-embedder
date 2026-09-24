'''
Sealed evaluation panels (roadmap Stage 2 onward).

The outcome panel scores text-to-code decoding on held-out Census index entries: ``index_roles``
gives every entry exactly one role, ``leakage`` keeps held-out queries out of training text,
``decoding`` scores an encoder over the six-digit candidates, and ``outcome`` exposes the splits
behind ``selection_log``, which records every read and the one logged opening of the test split.
'''
