package net.davidrobles.axon.agents;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import net.davidrobles.axon.ReplayBuffer;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.Transition;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.QFunctionObservable;
import net.davidrobles.axon.valuefunctions.QFunctionObserver;
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
public class QLearningWithReplay<S, A> implements QFunctionObservable<S, A> {
    private final Policy<S, A> policy;
    private final double gamma;
    private final int batchSize;
    private final Random rng;
    private final TrainableQFunction<S, A> table;
    private final ReplayBuffer<S, A> buffer;
    private final Set<QFunctionObserver<S, A>> qFunctionObservers = new LinkedHashSet<>();

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
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        if (batchSize < 1)
            throw new IllegalArgumentException("batchSize must be >= 1, got: " + batchSize);
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.policy = Objects.requireNonNull(policy, "policy must not be null");
        this.buffer = Objects.requireNonNull(buffer, "buffer must not be null");
        this.rng = Objects.requireNonNull(rng, "rng must not be null");
        this.gamma = gamma;
        this.batchSize = batchSize;
    }

    @Override
    public A selectAction(S state, List<A> actions) {
        return policy.selectAction(state, actions);
    }

    @Override
    public void update(S state, A action, StepResult<S> result, List<A> nextActions) {
        buffer.add(new Transition<>(state, action, result, nextActions));

        if (buffer.size() < batchSize) return;

        for (Transition<S, A> t : buffer.sample(batchSize, rng)) {
            qLearningUpdate(t.state(), t.action(), t.result(), t.nextActions());
        }
    }

    private void qLearningUpdate(S state, A action, StepResult<S> result, List<A> nextActions) {
        double maxNextQ = 0.0;
        if (!result.done() && !nextActions.isEmpty()) {
            maxNextQ = Double.NEGATIVE_INFINITY;
            for (A a : nextActions) {
                double v = table.getValue(result.nextState(), a);
                if (v > maxNextQ) maxNextQ = v;
            }
        }
        table.update(state, action, result.reward() + gamma * maxNextQ);
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
