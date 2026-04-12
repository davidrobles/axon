package net.davidrobles.axon.agents;

import java.util.Objects;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.values.TrainableQFunction;

/**
 * Off-policy tabular Q-Learning (Watkins, 1989).
 *
 * <p>The update target is the greedy (max) action value over the next state, making this an
 * off-policy algorithm: the behavior policy used for exploration can differ from the implicit
 * greedy target policy.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class QLearning<S, A> extends AbstractQAgent<S, A> {
    private final double gamma;
    private final TrainableQFunction<S, A> table;

    /**
     * @param table the Q-function to update (shared with the behavior policy); owns the learning
     *     rate
     * @param policy the behavior policy used for action selection
     * @param gamma discount factor
     */
    public QLearning(TrainableQFunction<S, A> table, Policy<S, A> policy, double gamma) {
        super(policy);
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.gamma = gamma;
    }

    @Override
    public void update(Experience<S, A> exp) {
        double maxNextQ = 0.0;

        if (!exp.done() && !exp.nextActions().isEmpty()) {
            maxNextQ = Double.NEGATIVE_INFINITY;
            for (A nextAction : exp.nextActions()) {
                double v = table.getValue(exp.nextState(), nextAction);
                if (v > maxNextQ) maxNextQ = v;
            }
        }

        table.update(exp.state(), exp.action(), exp.reward() + gamma * maxNextQ);
        notifyQFunctionObservers(table);
    }
}
