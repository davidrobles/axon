package net.davidrobles.axon.agents;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import net.davidrobles.axon.QPair;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.QFunctionObservable;
import net.davidrobles.axon.valuefunctions.QFunctionObserver;
import net.davidrobles.axon.valuefunctions.TrainableQFunction;

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
public class DynaQ<S, A> implements QFunctionObservable<S, A> {
    private record ModelEntry<S, A>(StepResult<S> result, List<A> nextActions) {}

    private final Policy<S, A> policy;
    private final double gamma;
    private final int planningSteps;
    private final Random rng;
    private final TrainableQFunction<S, A> table;
    private final Map<QPair<S, A>, ModelEntry<S, A>> model = new HashMap<>();
    private final List<QPair<S, A>> observedPairs = new ArrayList<>();
    private final Set<QFunctionObserver<S, A>> qFunctionObservers = new LinkedHashSet<>();

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
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        if (planningSteps < 0)
            throw new IllegalArgumentException("planningSteps must be >= 0, got: " + planningSteps);
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.policy = Objects.requireNonNull(policy, "policy must not be null");
        this.rng = Objects.requireNonNull(rng, "rng must not be null");
        this.gamma = gamma;
        this.planningSteps = planningSteps;
    }

    @Override
    public A selectAction(S state, List<A> actions) {
        return policy.selectAction(state, actions);
    }

    @Override
    public void update(S state, A action, StepResult<S> result, List<A> nextActions) {
        // 1. Q-Learning update from real experience
        qLearningUpdate(state, action, result, nextActions);

        // 2. Update the transition model
        QPair<S, A> pair = new QPair<>(state, action);
        if (!model.containsKey(pair)) {
            observedPairs.add(pair);
        }
        model.put(pair, new ModelEntry<>(result, nextActions));

        // 3. Planning: simulate planningSteps Q-Learning updates from the model
        for (int i = 0; i < planningSteps; i++) {
            QPair<S, A> simPair = observedPairs.get(rng.nextInt(observedPairs.size()));
            ModelEntry<S, A> entry = model.get(simPair);
            qLearningUpdate(simPair.state(), simPair.action(), entry.result(), entry.nextActions());
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
