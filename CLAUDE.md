# Axon

Tabular reinforcement learning framework in Java.

## Build & Test

```bash
./gradlew build       # build all modules
./gradlew test        # run all tests
```

Java 21 is required.

## Module Structure

```
modules/
  core/        # RL framework — interfaces, algorithms, policies, value functions
  gridworld/   # GridWorld domain with visualization
  examples/    # Runnable examples combining planning and learning
  util/        # Swing GUI utilities
```

## Package

All framework code lives under `net.davidrobles.axon`.

## Git

Never add Claude as a co-author or contributor in commit messages.

After every change, always check whether anything in `README.md` or `CLAUDE.md` needs to be updated to reflect the change.

## Key Conventions

- Module dependencies: `examples` → `gridworld` → `core`, `util`
- Algorithms are generic over `<S, A>` (state, action) — no domain-specific code in `core`
- Value functions are injected into agents and policies; agents do not own the training loop — `RLLoop` does
- Planning algorithms use `MDP<S, A>`; model-free algorithms use `Environment<S, A>`
- Code is formatted with Spotless (Google Java Format, AOSP style) — run `./gradlew spotlessApply` before committing
