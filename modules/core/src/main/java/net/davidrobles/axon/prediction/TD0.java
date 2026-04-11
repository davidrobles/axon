package net.davidrobles.axon.prediction;

import java.util.List;
import java.util.Objects;
import net.davidrobles.axon.Evaluator;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.AbstractVFunctionObservable;
import net.davidrobles.axon.valuefunctions.TrainableVFunction;

/**
 * TD(0) for on-policy state value prediction.
 *
 * <p>Estimates the value function V^π for a fixed policy π using one-step temporal-difference
 * updates. Action selection is fully delegated to the provided policy.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class TD0<S, A> extends AbstractVFunctionObservable<S> implements Evaluator<S, A> {
    private final Policy<S, A> policy;
    private final double gamma;
    private final TrainableVFunction<S> table;

    /**
     * @param table the V-function to evaluate and update (shared with the caller); owns the
     *     learning rate
     * @param policy the behavior policy used for action selection
     * @param gamma discount factor
     */
    public TD0(TrainableVFunction<S> table, Policy<S, A> policy, double gamma) {
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.policy = Objects.requireNonNull(policy, "policy must not be null");
        this.gamma = gamma;
    }

    @Override
    public A selectAction(S state, List<A> actions) {
        return policy.selectAction(state, actions);
    }

    @Override
    public void observe(S state, StepResult<S> result) {
        double nextV = result.done() ? 0.0 : table.getValue(result.nextState());
        table.update(state, result.reward() + gamma * nextV);
        notifyVFunctionObservers(table);
    }
}
