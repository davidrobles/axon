package net.davidrobles.axon;

import java.util.List;

/**
 * Base interface for <em>prediction</em> algorithms — agents that estimate the value of a fixed
 * policy without improving it. Action selection is delegated entirely to an external policy; the
 * algorithm only updates its value estimates from observed transitions.
 *
 * <p>The {@link #observe} method is the primary hook; {@link Agent#update} is provided as a default
 * that ignores the action and next-action arguments, which are irrelevant for prediction.
 *
 * <p>For <em>control</em> (learning to act by improving a policy), see {@link Agent}.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface Evaluator<S, A> extends Agent<S, A> {
    /**
     * Observes a transition and updates the value estimate for {@code state}.
     *
     * @param state the state before the transition
     * @param result the result of the transition (reward, next state, done flag)
     */
    void observe(S state, StepResult<S> result);

    @Override
    default void update(S state, A action, StepResult<S> result, List<A> nextActions) {
        observe(state, result);
    }
}
