package net.davidrobles.axon.prediction;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import net.davidrobles.axon.Evaluator;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.AbstractVFunctionObservable;
import net.davidrobles.axon.valuefunctions.TrainableVFunction;

/**
 * First-visit Monte Carlo prediction for on-policy state value estimation.
 *
 * <p>Buffers transitions throughout an episode and, at episode end, computes the actual discounted
 * return for each state and updates V(s) for the first visit to each state per episode.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class MCPrediction<S, A> extends AbstractVFunctionObservable<S, A>
        implements Evaluator<S, A> {
    private final Policy<S, A> policy;
    private final double gamma;
    private final TrainableVFunction<S> table;
    private final List<S> states = new ArrayList<>();
    private final List<Double> rewards = new ArrayList<>();

    /**
     * @param table the V-function to evaluate and update; owns the learning rate
     * @param policy the behavior policy used for action selection
     * @param gamma discount factor
     */
    public MCPrediction(TrainableVFunction<S> table, Policy<S, A> policy, double gamma) {
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
        states.add(state);
        rewards.add(result.reward());

        if (result.done()) {
            flush();
        }
    }

    private void flush() {
        int n = states.size();
        double G = 0.0;
        Set<S> visited = new HashSet<>();

        for (int t = n - 1; t >= 0; t--) {
            G = rewards.get(t) + gamma * G;
            S s = states.get(t);

            if (visited.add(s)) {
                table.update(s, G);
                notifyVFunctionObservers(table);
            }
        }

        states.clear();
        rewards.clear();
    }
}
