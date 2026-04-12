package net.davidrobles.axon;

import java.util.List;

/**
 * A single transition experienced by an agent during environment interaction.
 *
 * <p>Captures the full SARS' tuple — the state the agent was in, the action it took, the reward
 * received, the resulting next state, whether the episode ended, and the actions available from the
 * next state — everything an agent needs to perform one update step.
 *
 * @param state the state in which the action was taken
 * @param action the action that was taken
 * @param reward the reward signal received for this transition
 * @param nextState the state reached after the action was applied
 * @param done {@code true} if {@code nextState} is a terminal state (episode over)
 * @param nextActions available actions in {@code nextState}; empty when {@code done} is true
 * @param <S> the state type
 * @param <A> the action type
 */
public record Experience<S, A>(
        S state, A action, double reward, S nextState, boolean done, List<A> nextActions) {}
