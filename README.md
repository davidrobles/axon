# Axon

A tabular reinforcement learning framework in Java. Implements classic model-free and model-based algorithms with a clean, generic API.

## Modules

| Module | Description |
|--------|-------------|
| `core` | Core framework — interfaces, algorithms, policies, value functions |
| `envs` | Environment implementations (e.g. GridWorld) with visualization |
| `examples` | Runnable examples combining planning and learning |
| `util` | Swing GUI utilities |

## Algorithms

**Model-Free Control**
- Monte Carlo Control (on-policy, first-visit)
- Q-Learning (off-policy)
- SARSA (on-policy)
- Expected SARSA (on-policy, lower variance)
- SARSA(λ) with eligibility traces

**Prediction**
- Monte Carlo Prediction (first-visit)
- TD(0)
- TD(λ) with eligibility traces

**Planning (model-based)**
- Value Iteration
- Policy Iteration

## Core Concepts

### Environment vs MDP

- `Environment<S, A>` — interactive interface for model-free algorithms; exposes `step()`, `reset()`, `getActions()`
- `MDP<S, A>` — full model for planning; exposes transition probabilities and reward function

### Agent

```java
interface Agent<S, A> {
    A selectAction(S state, List<A> actions);
    void update(S state, A action, StepResult<S> result, List<A> nextActions);
}
```

Agents only define the update rule. The `RLLoop` drives the episode loop.

### RLLoop

```java
RLLoop.run(environment, agent, policy, numEpisodes);
```

Handles reset → select → step → update for each episode.

### Policies

| Policy | Behavior |
|--------|----------|
| `GreedyPolicy` | Always selects the highest Q-value action |
| `EpsilonGreedy` | ε-random exploration; supports linear ε decay |
| `RandomPolicy` | Uniform random selection |
| `TabularPolicy` | Deterministic state→action map (used by planners) |

Policies implement lifecycle hooks (`reset()`, `onStep()`, `onEpisodeEnd()`) for scheduling.

### Value Functions

| Class | Type | Notes |
|-------|------|-------|
| `TabularQFunction` | Q(s,a) | HashMap-backed; defaults to 0.0 |
| `TabularVFunction` | V(s) | HashMap-backed; α=1.0 by default |

Both implement `TrainableQFunction` / `TrainableVFunction` with an `update(state, tdTarget)` method.

### Observer Pattern

Agents implement `ObservableQAgent` or `ObservableVAgent`, notifying registered observers after each value function update. Used to hook in real-time visualization.

## Quick Start

**Q-Learning on GridWorld:**

```java
var rng = new Random(42);
var mdp = new GridWorldMDP(10, 10, rng);
var env = new GridWorldEnv(mdp, rng);

var qFunc = new TabularQFunction<GWState, GWAction>(0.1);   // α = 0.1
var policy = new EpsilonGreedy<>(qFunc, 0.1, rng);
var agent = new QLearning<>(qFunc, policy, 0.99);           // γ = 0.99

RLLoop.run(env, agent, policy, 1000);
```

**Value Iteration on GridWorld:**

```java
var mdp = new GridWorldMDP(10, 10, new Random(42));
var planner = new ValueIteration<>(mdp, 0.01, 0.99);        // θ=0.01, γ=0.99
Policy<GWState, GWAction> policy = planner.solve();
```

## Requirements

- Java 21
- Gradle 8.13

## Build

```bash
./gradlew build
./gradlew test
```
