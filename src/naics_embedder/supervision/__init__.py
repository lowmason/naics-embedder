'''
The Stage-3 supervision bundle and what training reads from it.

Import from the submodules: ``schema`` (the contract vocabulary and the manifest), ``artifacts``
(the validated bundle), ``code_targets`` and ``queries`` (the inputs of Req 11's terms),
``activity`` (a cross-reference's activity phrase) and ``checkpoints`` (the checkpoint contract).
The package imports nothing itself, so importing a submodule loads only what that submodule imports
(``activity`` and ``schema`` load nothing else).
'''
