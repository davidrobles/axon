package net.davidrobles.axon.policies;

import java.util.List;

/**
 * A Reinforcement Learning policy: maps a state and its available actions to a chosen action.
 *
 * <p>Any value function the policy relies on (e.g. Q-function for greedy selection) is bound at
 * construction time, not passed on every call. Because tabular value functions are mutable
 * references, the policy always sees the latest estimates as the algorithm updates them.
 *
 * <p>Training-loop lifecycle callbacks live on {@link net.davidrobles.axon.LoopListener} instead of
 * on the policy itself.
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
}
