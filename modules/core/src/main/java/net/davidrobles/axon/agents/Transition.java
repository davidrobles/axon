package net.davidrobles.axon.agents;

import java.util.List;
import net.davidrobles.axon.StepResult;

/**
 * A single environment transition, as experienced by an agent.
 *
 * <p>Bundles the four values passed to {@link net.davidrobles.axon.Agent#update} into a reusable
 * record, primarily for storage in a {@link ReplayBuffer}.
 *
 * @param state the state in which the action was taken
 * @param action the action that was taken
 * @param result the step result (next state, reward, done flag)
 * @param nextActions available actions in the next state; empty if the episode is done
 * @param <S> the state type
 * @param <A> the action type
 */
public record Transition<S, A>(S state, A action, StepResult<S> result, List<A> nextActions) {}
