package net.davidrobles.axon.agents;

import java.util.Objects;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.policies.StochasticPolicy;
import net.davidrobles.axon.values.TrainableQFunction;

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
public class ExpectedSARSA<S, A> extends AbstractQAgent<S, A> {
    private final StochasticPolicy<S, A> stochasticPolicy;
    private final double gamma;
    private final TrainableQFunction<S, A> table;

    /**
     * @param table the Q-function to update (shared with the behavior policy); owns the learning
     *     rate
     * @param policy the stochastic behavior policy used for action selection and probability
     *     queries
     * @param gamma discount factor
     */
    public ExpectedSARSA(
            TrainableQFunction<S, A> table, StochasticPolicy<S, A> policy, double gamma) {
        super(policy);
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.stochasticPolicy = Objects.requireNonNull(policy, "policy must not be null");
        this.gamma = gamma;
    }

    @Override
    public void update(Experience<S, A> exp) {
        double expectedNextQ = 0.0;

        if (!exp.done() && !exp.nextActions().isEmpty()) {
            for (A nextAction : exp.nextActions()) {
                double prob =
                        stochasticPolicy.probability(
                                exp.nextState(), nextAction, exp.nextActions());
                expectedNextQ += prob * table.getValue(exp.nextState(), nextAction);
            }
        }

        table.update(exp.state(), exp.action(), exp.reward() + gamma * expectedNextQ);
        notifyQFunctionObservers(table);
    }
}
