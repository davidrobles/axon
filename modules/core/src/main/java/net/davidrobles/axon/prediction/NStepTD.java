package net.davidrobles.axon.prediction;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import net.davidrobles.axon.Evaluator;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.AbstractVFunctionObservable;
import net.davidrobles.axon.valuefunctions.TrainableVFunction;

/**
 * n-step TD prediction for on-policy state value estimation.
 *
 * <p>Generalizes TD(0) and Monte Carlo prediction by bootstrapping from the value estimate n steps
 * ahead:
 *
 * <pre>G_{t:t+n} = r_{t+1} + γr_{t+2} + ... + γ^{n-1}r_{t+n} + γ^n V(s_{t+n})</pre>
 *
 * <p>Setting n=1 recovers TD(0); as n→∞ the return approaches Monte Carlo.
 *
 * <p>Transitions are buffered internally. Updates to V(s_t) are applied n steps after s_t is first
 * visited, or at episode end for states within the last n steps.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class NStepTD<S, A> extends AbstractVFunctionObservable<S, A> implements Evaluator<S, A> {
    private record Entry<S>(S state, double reward) {}

    private final Policy<S, A> policy;
    private final double gamma;
    private final int n;
    private final TrainableVFunction<S> table;
    private final Deque<Entry<S>> buffer = new ArrayDeque<>();

    /**
     * @param table the V-function to evaluate and update; owns the learning rate
     * @param policy the behavior policy used for action selection
     * @param n number of steps to look ahead before bootstrapping; must be >= 1
     * @param gamma discount factor
     */
    public NStepTD(TrainableVFunction<S> table, Policy<S, A> policy, int n, double gamma) {
        if (n < 1) throw new IllegalArgumentException("n must be >= 1, got: " + n);
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.policy = Objects.requireNonNull(policy, "policy must not be null");
        this.n = n;
        this.gamma = gamma;
    }

    @Override
    public A selectAction(S state, List<A> actions) {
        return policy.selectAction(state, actions);
    }

    @Override
    public void observe(S state, StepResult<S> result) {
        buffer.addLast(new Entry<>(state, result.reward()));

        if (buffer.size() == n) {
            updateOldest(result.nextState(), result.done());
        }

        if (result.done()) {
            flush();
        }
    }

    /** Computes the n-step return for the oldest buffered state and updates the value function. */
    private void updateOldest(S nextState, boolean done) {
        double G = computeReturn(nextState, done);
        S stateToUpdate = buffer.removeFirst().state();
        table.update(stateToUpdate, G);
        notifyVFunctionObservers(table);
    }

    /**
     * Computes G = Σ_{i=0}^{k-1} γ^i * r_i + γ^k * V(nextState), where k = buffer.size(). The
     * bootstrap term is omitted when {@code done} is true.
     */
    private double computeReturn(S nextState, boolean done) {
        double G = 0.0;
        double discount = 1.0;
        for (Entry<S> entry : buffer) {
            G += discount * entry.reward();
            discount *= gamma;
        }
        if (!done) {
            G += discount * table.getValue(nextState);
        }
        return G;
    }

    /**
     * Flushes remaining buffered states at episode end. Each state uses only the rewards still in
     * the buffer with no bootstrap, since the episode has terminated.
     */
    private void flush() {
        while (!buffer.isEmpty()) {
            S stateToUpdate = buffer.peekFirst().state();
            double G = 0.0;
            double discount = 1.0;
            for (Entry<S> entry : buffer) {
                G += discount * entry.reward();
                discount *= gamma;
            }
            buffer.removeFirst();
            table.update(stateToUpdate, G);
            notifyVFunctionObservers(table);
        }
    }
}
