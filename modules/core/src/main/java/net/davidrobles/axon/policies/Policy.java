package net.davidrobles.axon.policies;

import java.util.List;

/**
 * A Reinforcement Learning policy: maps a state and its available actions to a chosen action.
 *
 * <p>Any value function the policy relies on (e.g. Q-function for greedy selection) is bound at
 * construction time, not passed on every call. Because tabular value functions are mutable
 * references, the policy always sees the latest estimates as the algorithm updates them.
 *
 * <p>Policies can optionally expose a train/eval mode toggle via {@link #setTrainingMode(boolean)}.
 * Training-loop lifecycle callbacks live on {@link net.davidrobles.axon.LoopListener} instead of on
 * the policy itself.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface Policy<S, A> {
    /**
     * Selects an action for the given state.
     *
     * @param state the current state
     * @param actions the list of available actions (non-empty)
     * @return the selected action
     */
    A selectAction(S state, List<A> actions);

    /**
     * Switches the policy between training mode (exploration enabled) and evaluation mode
     * (deterministic / greedy). This method is not called automatically by {@link
     * net.davidrobles.axon.RLLoop}; callers must invoke it explicitly before evaluation runs.
     * Default implementation is a no-op.
     *
     * @param training {@code true} to enable exploration, {@code false} for greedy evaluation
     */
    default void setTrainingMode(boolean training) {}
}
