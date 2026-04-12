package net.davidrobles.axon.prediction;

import java.util.List;
import java.util.Objects;
import net.davidrobles.axon.Agent;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.AbstractVFunctionObservable;

/**
 * Abstract base class for tabular V-function prediction algorithms.
 *
 * <p>Holds the behavior policy, implements {@link #selectAction} by delegating to it, and bridges
 * {@link Agent#update} to the algorithm-specific {@link #observe} method. Subclasses only need to
 * implement {@link #observe}.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public abstract class AbstractVAgent<S, A> extends AbstractVFunctionObservable<S>
        implements Agent<S, A> {
    protected final Policy<S, A> policy;

    protected AbstractVAgent(Policy<S, A> policy) {
        this.policy = Objects.requireNonNull(policy, "policy must not be null");
    }

    @Override
    public A selectAction(S state, List<A> actions) {
        return policy.selectAction(state, actions);
    }

    @Override
    public void update(Experience<S, A> exp) {
        observe(exp.state(), new StepResult<>(exp.nextState(), exp.reward(), exp.done()));
    }

    /**
     * Updates the value estimate from a single observed transition.
     *
     * @param state the state before the transition
     * @param result the step result (next state, reward, done flag)
     */
    public abstract void observe(S state, StepResult<S> result);
}
