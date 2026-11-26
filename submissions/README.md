# My three final submissions

These are the exact packages I uploaded and chose as my three final selections. The private leaderboard scores the best of the three, and the platform doesn't show which one produced 94th.

| File | Public LB | Platform runtime | What it is |
|---|---|---|---|
| `sub48.zip` | 0.66230 | 56:12 | Ridge stack of W384F, mktsoupF, mixF, wpg and the organisers' baseline; 4-bit weights on W384F |
| `sub52.zip` | 0.66223 | 49:12 | sub48 with a distilled W384DF in W384F's slot |
| `sub43.zip` | 0.66180 | 42:21 | An earlier stack: two 192-wide REG192F seeds, mktsoupF, mixF, wpg and the organisers' baseline |

Each zip holds:
- `solution.py`: the streaming inference engine. It's Numba plus small C kernels that are compiled for the host CPU at load time and self-tested, falling back to Numba if anything fails.
- `weights.npz`: my members' weights and the stack weights. The organisers' baseline member is not included (see below).

It follows the competition's submission interface: one row in, two predictions out, state reset between sequences. Running it needs Python with NumPy and Numba, plus gcc for the fast path. The competition data isn't included.

**The organisers' baseline is removed.** All three stacks include the organisers' starter GRU as one member. It isn't mine to republish, so its weights (the `g0_*` arrays) are stripped from these copies. As published, the packages won't run end to end: the engine expects that member. Anyone who has the competition's starter pack can port its baseline back in. Every other member was trained by me. Claude (Anthropic's model, through Claude Code) wrote the code; [technical.md](../technical.md) has the details.
