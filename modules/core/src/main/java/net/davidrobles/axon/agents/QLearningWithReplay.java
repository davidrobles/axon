package net.davidrobles.axon.agents;

import java.util.Objects;
import java.util.Random;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.replay.ReplayBuffer;
import net.davidrobles.axon.valuefunctions.TrainableQFunction;

/**
 * Off-policy Q-Learning with experience replay (Lin, 1992).
 *
 * <p>Each real transition is stored in a {@link ReplayBuffer}. Once the buffer holds at least
 * {@code batchSize} transitions, a random mini-batch is sampled and a Q-Learning update is applied
 * to every transition in the batch. This breaks the temporal correlation between consecutive
 * updates and improves sample efficiency.
 *
 * <p>No updates are performed until the buffer has accumulated at least {@code batchSize}
 * transitions.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class QLearningWithReplay<S, A> extends AbstractQAgent<S, A> {
    private final double gamma;
    private final int batchSize;
    private final Random rng;
    private final TrainableQFunction<S, A> table;
    private final ReplayBuffer<S, A> buffer;

    /**
     * @param table the Q-function to update (shared with the behavior policy); owns the learning
     *     rate
     * @param policy the behavior policy used for action selection
     * @param gamma discount factor
     * @param buffer the replay buffer used to store and sample transitions
     * @param batchSize number of transitions to sample per update; must be >= 1
     * @param rng random number generator used for sampling the mini-batch
     */
    public QLearningWithReplay(
            TrainableQFunction<S, A> table,
            Policy<S, A> policy,
            double gamma,
            ReplayBuffer<S, A> buffer,
            int batchSize,
            Random rng) {
        super(policy);
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        if (batchSize < 1)
            throw new IllegalArgumentException("batchSize must be >= 1, got: " + batchSize);
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.buffer = Objects.requireNonNull(buffer, "buffer must not be null");
        this.rng = Objects.requireNonNull(rng, "rng must not be null");
        this.gamma = gamma;
        this.batchSize = batchSize;
    }

    @Override
    public void update(Experience<S, A> exp) {
        buffer.add(exp);

        if (buffer.size() < batchSize) return;

        for (Experience<S, A> t : buffer.sample(batchSize, rng)) {
            qLearningUpdate(t);
        }
    }

    private void qLearningUpdate(Experience<S, A> exp) {
        double maxNextQ = 0.0;
        if (!exp.done() && !exp.nextActions().isEmpty()) {
            maxNextQ = Double.NEGATIVE_INFINITY;
            for (A a : exp.nextActions()) {
                double v = table.getValue(exp.nextState(), a);
                if (v > maxNextQ) maxNextQ = v;
            }
        }
        table.update(exp.state(), exp.action(), exp.reward() + gamma * maxNextQ);
        notifyQFunctionObservers(table);
    }
}
