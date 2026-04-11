package net.davidrobles.axon.agents;

import java.util.List;
import java.util.Objects;
import java.util.Random;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.AbstractQFunctionObservable;
import net.davidrobles.axon.valuefunctions.QFunction;
import net.davidrobles.axon.valuefunctions.TrainableQFunction;

/**
 * Double Q-Learning (van Hasselt, 2010).
 *
 * <p>Standard Q-Learning suffers from <em>maximization bias</em>: using the same table to both
 * select and evaluate the greedy action overestimates Q-values in stochastic environments. Double
 * Q-Learning decouples the two steps by maintaining two independent tables, QA and QB.
 *
 * <p>On each update a fair coin determines which table is updated:
 *
 * <ul>
 *   <li>Heads — update QA: select the greedy action {@code a* = argmax_a QA(s', a)}, then evaluate
 *       with QB: {@code target = r + γ·QB(s', a*)}.
 *   <li>Tails — update QB: select with QB, evaluate with QA.
 * </ul>
 *
 * <p>The behavior policy should be constructed with a composite Q-function that averages or sums
 * both tables (see factory helpers in the GridWorldDemo examples). Observers receive the average of
 * QA and QB after every update.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class DoubleQLearning<S, A> extends AbstractQFunctionObservable<S, A> {
    private final TrainableQFunction<S, A> qA;
    private final TrainableQFunction<S, A> qB;
    private final Policy<S, A> policy;
    private final double gamma;
    private final Random rng;
    private final QFunction<S, A> composite;

    /**
     * @param qA first Q-table
     * @param qB second Q-table
     * @param policy the behavior policy (should select actions based on both tables)
     * @param gamma discount factor in [0, 1]
     * @param rng random number generator used for the per-step coin flip
     */
    public DoubleQLearning(
            TrainableQFunction<S, A> qA,
            TrainableQFunction<S, A> qB,
            Policy<S, A> policy,
            double gamma,
            Random rng) {
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        this.qA = Objects.requireNonNull(qA, "qA must not be null");
        this.qB = Objects.requireNonNull(qB, "qB must not be null");
        this.policy = Objects.requireNonNull(policy, "policy must not be null");
        this.rng = Objects.requireNonNull(rng, "rng must not be null");
        this.gamma = gamma;
        // Pre-built composite for observer callbacks — avoids a new lambda per update
        this.composite = (s, a) -> (qA.getValue(s, a) + qB.getValue(s, a)) / 2.0;
    }

    @Override
    public A selectAction(S state, List<A> actions) {
        return policy.selectAction(state, actions);
    }

    @Override
    public void update(S state, A action, StepResult<S> result, List<A> nextActions) {
        if (rng.nextBoolean()) {
            // Heads: update qA, select with qA, evaluate with qB
            double target = computeTarget(result, nextActions, qA, qB);
            qA.update(state, action, result.reward() + gamma * target);
        } else {
            // Tails: update qB, select with qB, evaluate with qA
            double target = computeTarget(result, nextActions, qB, qA);
            qB.update(state, action, result.reward() + gamma * target);
        }
        notifyQFunctionObservers(composite);
    }

    /**
     * Computes the bootstrapped target for the next state by selecting the greedy action from
     * {@code selector} and evaluating it with {@code evaluator}.
     */
    private double computeTarget(
            StepResult<S> result,
            List<A> nextActions,
            QFunction<S, A> selector,
            QFunction<S, A> evaluator) {
        if (result.done() || nextActions.isEmpty()) return 0.0;
        A bestAction = null;
        double bestQ = Double.NEGATIVE_INFINITY;
        for (A a : nextActions) {
            double q = selector.getValue(result.nextState(), a);
            if (q > bestQ) {
                bestQ = q;
                bestAction = a;
            }
        }
        return evaluator.getValue(result.nextState(), bestAction);
    }
}
