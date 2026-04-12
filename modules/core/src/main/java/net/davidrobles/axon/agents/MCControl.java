package net.davidrobles.axon.agents;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.StateActionPair;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.values.TrainableQFunction;

/**
 * First-visit on-policy Monte Carlo control.
 *
 * <p>Buffers transitions throughout an episode and, at episode end, computes the actual discounted
 * return for each state-action pair and updates Q(s,a) for the first visit to each (s,a) pair per
 * episode.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class MCControl<S, A> extends AbstractQAgent<S, A> {
    private final double gamma;
    private final TrainableQFunction<S, A> table;
    private final List<S> states = new ArrayList<>();
    private final List<A> actions = new ArrayList<>();
    private final List<Double> rewards = new ArrayList<>();

    /**
     * @param table the Q-function to update (shared with the behavior policy); owns the learning
     *     rate
     * @param policy the behavior policy used for action selection
     * @param gamma discount factor
     */
    public MCControl(TrainableQFunction<S, A> table, Policy<S, A> policy, double gamma) {
        super(policy);
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.gamma = gamma;
    }

    @Override
    public void update(Experience<S, A> exp) {
        states.add(exp.state());
        actions.add(exp.action());
        rewards.add(exp.reward());

        if (exp.done()) {
            flush();
        }
    }

    private void flush() {
        int n = states.size();
        double G = 0.0;
        Set<StateActionPair<S, A>> visited = new HashSet<>();

        for (int t = n - 1; t >= 0; t--) {
            G = rewards.get(t) + gamma * G;
            StateActionPair<S, A> pair = new StateActionPair<>(states.get(t), actions.get(t));

            if (visited.add(pair)) {
                table.update(states.get(t), actions.get(t), G);
                notifyQFunctionObservers(table);
            }
        }

        states.clear();
        actions.clear();
        rewards.clear();
    }
}
