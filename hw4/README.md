# lab4-2026

This assignment is going to be a little different. We are trying a new assignment where you are required to vibe code.

## What is vibe coding

> In computer programming, vibe coding is a software development practice assisted by artificial intelligence (AI) such as by chatbots (programs that simulate conversation) or AI agents such as Codex or Claude Code. [...]
> The term was coined by computer scientist Andrej Karpathy, a co-founder of OpenAI and former AI leader at Tesla, in February 2025.

*[Vibe coding, Wikipedia. Accessed April 2026](https://en.wikipedia.org/wiki/Vibe_coding)*

## Your assignment

Your assignment is to complete [Stanford CS149 Assignment 3](https://github.com/stanford-cs149/asst3), commit `43bf472`, with vibe coding.

The grading will be as follows:

100 points

- 75 points: dervied from `checker.py` scores, which validates your code's performance versus the CS149 baseline
  - You need to get their reference baselines (such as `render_ref_x86`) working *on the titan machine* (yes, we know it doesn't 'just work' with this Ubuntu version. You can get it working without `sudo` permission, this is part of the challenge.)
  - Score calculated in two parts
    - 50 points: Sum all possible 82 points from render `checker.py`, `checker.py find_repeats` (5pts), `checker.py scan` (5pts), then subtract 32
    - 25 points: Beat the reference by 10x for rank100k, snowsingle, biglittle, rand1M, and micro2M (5 points each)
- 25 points: writeup, suggested to write by hand (submit in `SUBMISSION.md`)
  - 12.5 points: Explain the vibe coding setups that you tried. What worked, what didn't? If you had to start a similar project in the future, how would you do it?
  - 12.5 points: Make a list of optimizations that the AI performed. Take one of these optimizations related to how a GPU works, and explain it in depth. How does your code relate to the physical hardware on the GPU? How could you apply this optimization to a similar problem in the future related to your research or a problem you are interested in?

When finished, save your submission to `lab4_2026_submission.zip` in your home directory. Your submission should be self-contained.

---

Assignment written Spring 2026 by Sam Foxman.