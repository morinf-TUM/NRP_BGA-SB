"""Task engines for NRP_BGA-SB.

Each engine represents a distinct task paradigm (go/no-go, two-choice, stop-signal,
change-of-mind). Engines share a common interface: accept configuration + policy,
run trials on a logical clock, and return TrialLog objects.
"""
