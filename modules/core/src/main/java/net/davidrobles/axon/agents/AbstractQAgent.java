package net.davidrobles.axon.agents;

import java.util.List;
import java.util.Objects;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.AbstractQFunctionObservable;

/**
 * Abstract base class for tabular Q-function agents.
 *
 * <p>Holds the behavior policy and implements {@link #selectAction} by delegating to it. Subclasses
 * only need to implement {@link #update}.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public abstract class AbstractQAgent<S, A> extends AbstractQFunctionObservable<S, A> {
    protected final Policy<S, A> policy;

    protected AbstractQAgent(Policy<S, A> policy) {
        this.policy = Objects.requireNonNull(policy, "policy must not be null");
    }

    @Override
    public A selectAction(S state, List<A> actions) {
        return policy.selectAction(state, actions);
    }
}
