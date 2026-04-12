package net.davidrobles.axon.agents;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.TrainableQFunction;

/**
 * On-policy tabular n-step SARSA.
 *
 * <p>Generalizes SARSA and Monte Carlo control by bootstrapping from the Q-value estimate n steps
 * ahead:
 *
 * <pre>G_{t:t+n} = r_{t+1} + γr_{t+2} + ... + γ^{n-1}r_{t+n} + γ^n Q(s_{t+n}, a_{t+n})</pre>
 *
 * <p>Setting n=1 recovers SARSA; as n→∞ the return approaches Monte Carlo control.
 *
 * <p>Transitions are buffered internally. Updates to Q(s_t, a_t) are applied n steps after the pair
 * is first encountered, or at episode end for pairs within the last n steps.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class NStepSARSA<S, A> extends AbstractQAgent<S, A> {
    private record Entry<S, A>(S state, A action, double reward) {}

    private final double gamma;
    private final int n;
    private final TrainableQFunction<S, A> table;
    private final Deque<Entry<S, A>> buffer = new ArrayDeque<>();

    /**
     * @param table the Q-function to update (shared with the behavior policy); owns the learning
     *     rate
     * @param policy the behavior policy used for action selection and on-policy bootstrapping
     * @param n number of steps to look ahead before bootstrapping; must be >= 1
     * @param gamma discount factor
     */
    public NStepSARSA(TrainableQFunction<S, A> table, Policy<S, A> policy, int n, double gamma) {
        super(policy);
        if (n < 1) throw new IllegalArgumentException("n must be >= 1, got: " + n);
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.n = n;
        this.gamma = gamma;
    }

    @Override
    public void update(Experience<S, A> exp) {
        buffer.addLast(new Entry<>(exp.state(), exp.action(), exp.reward()));

        if (buffer.size() == n) {
            updateOldest(exp.nextState(), exp.nextActions(), exp.done());
        }

        if (exp.done()) {
            flush();
        }
    }

    /** Computes the n-step return for the oldest buffered pair and updates the Q-function. */
    private void updateOldest(S nextState, List<A> nextActions, boolean done) {
        double G = computeReturn(nextState, nextActions, done);
        Entry<S, A> oldest = buffer.removeFirst();
        table.update(oldest.state(), oldest.action(), G);
        notifyQFunctionObservers(table);
    }

    /**
     * Computes G = Σ_{i=0}^{k-1} γ^i * r_i + γ^k * Q(nextState, nextAction), where k =
     * buffer.size(). The bootstrap term is omitted when {@code done} is true or there are no
     * available next actions. The next action is selected on-policy from {@code nextActions}.
     */
    private double computeReturn(S nextState, List<A> nextActions, boolean done) {
        double G = 0.0;
        double discount = 1.0;
        for (Entry<S, A> entry : buffer) {
            G += discount * entry.reward();
            discount *= gamma;
        }
        if (!done && !nextActions.isEmpty()) {
            A nextAction = policy.selectAction(nextState, nextActions);
            G += discount * table.getValue(nextState, nextAction);
        }
        return G;
    }

    /**
     * Flushes remaining buffered pairs at episode end. Each pair uses only the rewards still in the
     * buffer with no bootstrap, since the episode has terminated.
     */
    private void flush() {
        while (!buffer.isEmpty()) {
            Entry<S, A> oldest = buffer.peekFirst();
            double G = 0.0;
            double discount = 1.0;
            for (Entry<S, A> entry : buffer) {
                G += discount * entry.reward();
                discount *= gamma;
            }
            buffer.removeFirst();
            table.update(oldest.state(), oldest.action(), G);
            notifyQFunctionObservers(table);
        }
    }
}
