package net.davidrobles.axon;

import net.davidrobles.axon.policies.Policy;

/**
 * Interface for prediction algorithms — algorithms that estimate the value of states under a fixed
 * policy without learning to act.
 *
 * <p>A predictor observes state transitions and updates its internal value estimate V(s). It has no
 * opinion about what action to take; action selection is entirely external. Predictors are
 * typically used as components inside larger systems (e.g. the critic in an actor-critic
 * architecture) or driven by {@link InteractionLoop#run(Environment, Policy, Predictor, int)}.
 *
 * <p>For algorithms that also select actions (control), see {@link Agent}.
 *
 * @param <S> the type of the states
 */
public interface Predictor<S> {
    /**
     * Updates the value estimate from a single observed transition.
     *
     * @param state the state before the transition
     * @param result the step result (next state, reward, done flag)
     */
    void observe(S state, StepResult<S> result);
}
