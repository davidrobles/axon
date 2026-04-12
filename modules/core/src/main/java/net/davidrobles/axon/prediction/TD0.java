package net.davidrobles.axon.prediction;

import java.util.Objects;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.values.TrainableVFunction;

/**
 * TD(0) for on-policy state value prediction.
 *
 * <p>Estimates the value function V^π for a fixed policy π using one-step temporal-difference
 * updates. Action selection is handled externally by the caller.
 *
 * @param <S> the type of the states
 */
public class TD0<S> extends AbstractPredictor<S> {
    private final double gamma;
    private final TrainableVFunction<S> table;

    /**
     * @param table the V-function to evaluate and update (shared with the caller); owns the
     *     learning rate
     * @param gamma discount factor
     */
    public TD0(TrainableVFunction<S> table, double gamma) {
        if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
        this.table = Objects.requireNonNull(table, "table must not be null");
        this.gamma = gamma;
    }

    @Override
    public void observe(S state, StepResult<S> result) {
        double nextV = result.done() ? 0.0 : table.getValue(result.nextState());
        table.update(state, result.reward() + gamma * nextV);
        notifyVFunctionObservers(table);
    }
}
