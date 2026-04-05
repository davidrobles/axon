package net.davidrobles.axon.planning;

import net.davidrobles.axon.policies.Policy;

/**
 * A planning algorithm that computes an optimal policy from a known model of the environment.
 *
 * <p>Unlike online {@link net.davidrobles.axon.Agent} implementations, a planner has full access to
 * the MDP dynamics and runs to convergence before returning a result.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface Planner<S, A> {
    /** Runs the planning algorithm to convergence and returns the resulting policy. */
    Policy<S, A> solve();
}
