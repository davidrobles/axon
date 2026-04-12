package net.davidrobles.axon.agents;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.StateActionPair;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.values.TrainableQFunction;

/**
 * Tabular Dyna-Q (Sutton, 1990).
 *
 * <p>Combines model-free Q-Learning with model-based planning. After each real interaction with the
 * environment, Dyna-Q:
 *
 * <ol>
 *   <li>Applies a standard Q-Learning update from the real experience.
 *   <li>Updates its internal transition model: (s, a) → (r, s', nextActions).
 *   <li>Performs {@code planningSteps} additional Q-Learning updates by sampling previously
 *       observed (s, a) pairs uniformly at random and replaying their stored transitions.
 * </ol>
 *
 * <p>The model assumes deterministic transitions — the latest observed outcome for each (s, a) pair
 * is stored and reused during planning.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class DynaQ<S, A> extends AbstractQAgent<S, A> {
    private final double gamma;
    private final int planningSteps;
    private final Random rng;
    private final TrainableQFunction<S, A> table;
    private final Map<StateActionPair<S, A>, Experience<S, A>> model = new HashMap<>();
    private final List<StateActionPair<S, A>> observedPairs = new ArrayList<>();

    /**
     * @param table the Q-function to update (shared with the behavior policy); owns the learning
     *     rate
     * @param policy the behavior policy used for action selection
     * @param gamma discount factor
     * @param planningSteps number of simulated Q-Learning updates to perform per real step
     * @param rng random number generator used for sampling model transitions during planning
     */
    public DynaQ(
            TrainableQFunction<S, A> table,
            Policy<S, A> policy,
            double gamma,
            int planningSteps,
            Random rng) {
        super(policy);
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        if (planningSteps < 0)
            throw new IllegalArgumentException("planningSteps must be >= 0, got: " + planningSteps);
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.rng = Objects.requireNonNull(rng, "rng must not be null");
        this.gamma = gamma;
        this.planningSteps = planningSteps;
    }

    @Override
    public void update(Experience<S, A> exp) {
        // 1. Q-Learning update from real experience
        qLearningUpdate(exp);

        // 2. Update the transition model
        StateActionPair<S, A> pair = new StateActionPair<>(exp.state(), exp.action());
        if (!model.containsKey(pair)) {
            observedPairs.add(pair);
        }
        model.put(pair, exp);

        // 3. Planning: simulate planningSteps Q-Learning updates from the model
        for (int i = 0; i < planningSteps; i++) {
            StateActionPair<S, A> simPair = observedPairs.get(rng.nextInt(observedPairs.size()));
            qLearningUpdate(model.get(simPair));
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
