package net.davidrobles.axon;

import java.util.List;

/**
 * Core interface for control algorithms that both act and learn from experience.
 *
 * <p>An agent encapsulates action selection and the update rule only; it does not own an
 * environment or episode loop. Use {@link RLLoop} to drive training.
 *
 * <p>For prediction-only algorithms that estimate state values under an external policy, see {@link
 * Predictor}.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface Agent<S, A> {
    /**
     * Selects an action to take in the given state.
     *
     * @param state the current state
     * @param actions the list of available actions (non-empty)
     * @return the selected action
     */
    A selectAction(S state, List<A> actions);

    /**
     * Updates the agent's internal state (e.g. value function) from one transition.
     *
     * @param experience the full transition experienced during this step
     */
    void update(Experience<S, A> experience);
}
