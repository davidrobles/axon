package net.davidrobles.axon.agents;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import net.davidrobles.axon.ObservableQAgent;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.StochasticPolicy;
import net.davidrobles.axon.valuefunctions.QFunctionObserver;
import net.davidrobles.axon.valuefunctions.TrainableQFunction;

/**
 * On-policy tabular Expected SARSA.
 *
 * <p>Like SARSA, but replaces the sampled next action value with the <em>expected</em> value over
 * the policy's action distribution:
 *
 * <pre>Q(s,a) ← Q(s,a) + α * [r + γ * Σ_{a'} π(a'|s') Q(s',a') − Q(s,a)]</pre>
 *
 * <p>This removes the variance from sampling the next action, making it lower-variance than SARSA
 * while remaining on-policy. It requires a {@link StochasticPolicy} to compute π(a'|s').
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class ExpectedSARSA<S, A> implements ObservableQAgent<S, A> {
    private final StochasticPolicy<S, A> policy;
    private final double gamma;
    private final TrainableQFunction<S, A> table;
    private final Set<QFunctionObserver<S, A>> qFunctionObservers = new LinkedHashSet<>();

    /**
     * @param table the Q-function to update (shared with the behavior policy); owns the learning
     *     rate
     * @param policy the stochastic behavior policy used for action selection and probability queries
     * @param gamma discount factor
     */
    public ExpectedSARSA(TrainableQFunction<S, A> table, StochasticPolicy<S, A> policy, double gamma) {
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
    public void update(S state, A action, StepResult<S> result, List<A> nextActions) {
        double expectedNextQ = 0.0;

        if (!result.done() && !nextActions.isEmpty()) {
            for (A nextAction : nextActions) {
                double prob = Math.exp(policy.logProbability(result.nextState(), nextAction, nextActions));
                expectedNextQ += prob * table.getValue(result.nextState(), nextAction);
            }
        }

        table.update(state, action, result.reward() + gamma * expectedNextQ);
        notifyQFunctionUpdate();
    }

    @Override
    public void addQFunctionObserver(QFunctionObserver<S, A> observer) {
        qFunctionObservers.add(observer);
    }

    private void notifyQFunctionUpdate() {
        for (QFunctionObserver<S, A> observer : qFunctionObservers)
            observer.qFunctionUpdated(table);
    }
}
