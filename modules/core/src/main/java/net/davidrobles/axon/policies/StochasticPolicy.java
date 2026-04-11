package net.davidrobles.axon.policies;

import java.util.List;

/**
 * A stochastic policy that can report the probability of selecting a given action.
 *
 * <p>Required by algorithms such as Expected SARSA that compute the expected value over the
 * policy's action distribution: Σ π(a|s) · Q(s,a).
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface StochasticPolicy<S, A> extends Policy<S, A> {
    /**
     * Returns the probability of selecting {@code action} in {@code state}.
     *
     * @param state the current state
     * @param action the action whose probability is queried
     * @param actions the full list of available actions (defines the support of the distribution)
     * @return π(action | state), a value in [0, 1]
     */
    double probability(S state, A action, List<A> actions);
}
