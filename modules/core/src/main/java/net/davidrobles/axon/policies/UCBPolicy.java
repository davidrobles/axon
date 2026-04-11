package net.davidrobles.axon.policies;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import net.davidrobles.axon.QPair;
import net.davidrobles.axon.valuefunctions.QFunction;

/**
 * Upper Confidence Bound policy (UCB1).
 *
 * <p>Balances exploration and exploitation by adding an exploration bonus to each action's Q-value:
 *
 * <pre>UCB(s, a) = Q(s, a) + c · √(ln N(s) / N(s, a))</pre>
 *
 * <p>where N(s) is the total number of times state {@code s} has been visited and N(s, a) is the
 * number of times action {@code a} was selected in state {@code s}. The constant {@code c} controls
 * the strength of the exploration bonus.
 *
 * <p>Any action that has never been taken in a state is always selected before applying the UCB
 * formula, giving it an implicit score of +∞. Visit counts accumulate across episodes.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class UCBPolicy<S, A> implements Policy<S, A> {
    private final QFunction<S, A> qFunc;
    private final double c;
    private final Map<S, Integer> stateCounts = new HashMap<>();
    private final Map<QPair<S, A>, Integer> actionCounts = new HashMap<>();

    /**
     * @param qFunc the Q-function used for action value estimates
     * @param c exploration constant; higher values encourage more exploration (must be >= 0)
     */
    public UCBPolicy(QFunction<S, A> qFunc, double c) {
        if (c < 0) throw new IllegalArgumentException("c must be >= 0, got: " + c);
        this.qFunc = Objects.requireNonNull(qFunc, "qFunc must not be null");
        this.c = c;
    }

    @Override
    public A selectAction(S state, List<A> actions) {
        // Prioritize any unvisited action (implicit UCB score of +∞)
        for (A action : actions) {
            if (actionCounts.getOrDefault(new QPair<>(state, action), 0) == 0) {
                updateCounts(state, action);
                return action;
            }
        }

        // All actions visited — apply UCB formula
        int n = stateCounts.get(state);
        double logN = Math.log(n);
        A bestAction = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (A action : actions) {
            int na = actionCounts.get(new QPair<>(state, action));
            double score = qFunc.getValue(state, action) + c * Math.sqrt(logN / na);
            if (score > bestScore) {
                bestScore = score;
                bestAction = action;
            }
        }

        updateCounts(state, bestAction);
        return bestAction;
    }

    private void updateCounts(S state, A action) {
        stateCounts.merge(state, 1, Integer::sum);
        actionCounts.merge(new QPair<>(state, action), 1, Integer::sum);
    }
}
