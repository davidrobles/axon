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
- Double Q-Learning (van Hasselt, 2010)
- SARSA (on-policy)
- Expected SARSA (on-policy, lower variance)
- n-step SARSA
- SARSA(λ) with eligibility traces

**Model-Based**
- Dyna-Q (Q-Learning + learned model + planning)

**Experience Replay**
- Q-Learning with experience replay (fixed-capacity circular buffer, random mini-batch sampling)

**Exploration**
- UCB (Upper Confidence Bound / UCB1)
- Softmax (Boltzmann exploration)

**Planning**
- Value Iteration
- Policy Iteration

**Prediction**
- Monte Carlo Prediction (first-visit)
- n-step TD
- TD(0)
- TD(λ) with eligibility traces


## Core Concepts

### Environment vs MDP

- `Environment<S, A>` — interactive interface for model-free algorithms; exposes `step()`, `reset()`, `getActions()`
- `MDP<S, A>` — full model for planning; exposes transition probabilities and reward function

### Agent

```java
interface Agent<S, A> {
    A selectAction(S state, List<A> actions);
    void update(Experience<S, A> experience);
}
```

Agents only define the update rule. The `InteractionLoop` drives the interaction loop.

### Predictor

```java
interface Predictor<S> {
    void observe(S state, StepResult<S> result);
}
```

Predictors solve the prediction problem only. They estimate `V(s)` under an external policy and do
not select actions themselves.

### InteractionLoop

```java
InteractionLoop.run(environment, agent, policy, numEpisodes);
```

Handles reset → select → step → update for each episode.

Scheduling and other training-loop concerns are supplied separately via `LoopListener`s:

```java
InteractionLoop.run(environment, agent, policy, numEpisodes, listener);
```

Prediction algorithms use the predictor overload:

```java
InteractionLoop.run(environment, policy, predictor, numEpisodes);
```

### Policies

| Policy | Behavior |
|--------|----------|
| `GreedyPolicy` | Always selects the highest Q-value action |
| `EpsilonGreedy` | ε-random exploration; supports linear ε decay |
| `UCBPolicy` | UCB1 bonus: Q(s,a) + c·√(ln N(s) / N(s,a)) |
| `SoftmaxPolicy` | Boltzmann sampling with temperature τ |
| `RandomPolicy` | Uniform random selection |
| `TabularPolicy` | Deterministic state→action map (used by planners) |

Policies only choose actions. Loop lifecycle callbacks live on `LoopListener`.
Use separate policy instances for training and evaluation when you want different behavior, e.g.
`EpsilonGreedy` for training and `GreedyPolicy` for evaluation.

### Value Functions

| Class | Type | Notes |
|-------|------|-------|
| `TabularQFunction` | Q(s,a) | HashMap-backed; defaults to 0.0 |
| `TabularVFunction` | V(s) | HashMap-backed; α=1.0 by default |

Both implement `TrainableQFunction` / `TrainableVFunction` with an `update(state, tdTarget)` method.

### Observer Pattern

Control algorithms and predictors expose observable value functions, notifying registered
observers after each value update. This is used to hook in real-time visualization.

## Quick Start

**Q-Learning on GridWorld:**

```java
var rng = new Random(42);
var mdp = new GridWorldMDP(10, 10, rng);
var env = new GridWorldEnv(mdp, rng);

var qFunc = new TabularQFunction<GWState, GWAction>(0.1);   // α = 0.1
var policy = new EpsilonGreedy<>(qFunc, 0.1, rng);
var agent = new QLearning<>(qFunc, policy, 0.99);           // γ = 0.99

InteractionLoop.run(env, agent, policy, 1000, policy);
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
