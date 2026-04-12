package net.davidrobles.axon.prediction;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.values.TrainableVFunction;

/**
 * TD(λ) for on-policy state value prediction with eligibility traces.
 *
 * <p>Extends TD(0) with accumulating eligibility traces that spread credit across recently visited
 * states. Setting λ=0 recovers TD(0); λ=1 approximates Monte Carlo updates.
 *
 * @param <S> the type of the states
 */
public class TDLambda<S> extends AbstractPredictor<S> {
    private final double gamma;
    private final double lambda;
    private final TrainableVFunction<S> table;
    private final Map<S, Double> traces = new HashMap<>();

    /**
     * @param table the V-function to evaluate and update (shared with the caller); owns the
     *     learning rate
     * @param gamma discount factor
     * @param lambda eligibility-trace decay rate (0 = TD(0), 1 = Monte Carlo)
     */
    public TDLambda(TrainableVFunction<S> table, double gamma, double lambda) {
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        if (lambda < 0 || lambda > 1)
            throw new IllegalArgumentException("lambda must be in [0, 1]");
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.gamma = gamma;
        this.lambda = lambda;
    }

    @Override
    public void observe(S state, StepResult<S> result) {
        double nextV = result.done() ? 0.0 : table.getValue(result.nextState());
        double tdError = result.reward() + gamma * nextV - table.getValue(state);

        // Accumulating trace: e(s) += 1
        traces.merge(state, 1.0, Double::sum);

        for (Map.Entry<S, Double> entry : traces.entrySet()) {
            S s = entry.getKey();
            table.update(s, table.getValue(s) + tdError * entry.getValue());
            entry.setValue(gamma * lambda * entry.getValue());
        }

        if (result.done()) traces.clear();
        notifyVFunctionObservers(table);
    }
}
