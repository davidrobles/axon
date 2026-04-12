package net.davidrobles.axon.policies;

import java.util.List;
import java.util.Objects;
import java.util.Random;
import net.davidrobles.axon.values.QFunction;

/**
 * Softmax (Boltzmann) exploration policy.
 *
 * <p>Selects actions proportionally to an exponential of their Q-values, scaled by a temperature
 * parameter {@code τ}:
 *
 * <pre>π(a | s) = exp(Q(s, a) / τ) / Σ_b exp(Q(s, b) / τ)</pre>
 *
 * <p>A high temperature approaches uniform random exploration; a temperature near zero approaches
 * greedy selection. The log-sum-exp trick is used for numerical stability when computing
 * probabilities.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public class SoftmaxPolicy<S, A> implements StochasticPolicy<S, A> {
    private final QFunction<S, A> qFunc;
    private final double temperature;
    private final Random rng;

    /**
     * @param qFunc the Q-function used for action value estimates
     * @param temperature controls the spread of the distribution; must be > 0
     * @param rng random number generator used for sampling
     */
    public SoftmaxPolicy(QFunction<S, A> qFunc, double temperature, Random rng) {
        if (temperature <= 0)
            throw new IllegalArgumentException("temperature must be > 0, got: " + temperature);
        this.qFunc = Objects.requireNonNull(qFunc, "qFunc must not be null");
        this.temperature = temperature;
        this.rng = Objects.requireNonNull(rng, "rng must not be null");
    }

    @Override
    public A selectAction(S state, List<A> actions) {
        double[] logProbs = computeLogProbs(state, actions);
        // Convert log-probs back to probs and sample
        double[] probs = new double[logProbs.length];
        for (int i = 0; i < logProbs.length; i++) {
            probs[i] = Math.exp(logProbs[i]);
        }
        double r = rng.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < actions.size(); i++) {
            cumulative += probs[i];
            if (r < cumulative) return actions.get(i);
        }
        return actions.get(actions.size() - 1);
    }

    @Override
    public double probability(S state, A action, List<A> actions) {
        double[] logProbs = computeLogProbs(state, actions);
        int idx = actions.indexOf(action);
        return idx >= 0 ? Math.exp(logProbs[idx]) : 0.0;
    }

    /**
     * Computes log-probabilities for all actions using the log-sum-exp trick for numerical
     * stability.
     */
    private double[] computeLogProbs(S state, List<A> actions) {
        int n = actions.size();
        double[] scaled = new double[n];
        double maxScaled = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            scaled[i] = qFunc.getValue(state, actions.get(i)) / temperature;
            if (scaled[i] > maxScaled) maxScaled = scaled[i];
        }
        // log-sum-exp: log Σ exp(x_i) = max + log Σ exp(x_i - max)
        double sumExp = 0.0;
        for (double s : scaled) {
            sumExp += Math.exp(s - maxScaled);
        }
        double logSumExp = maxScaled + Math.log(sumExp);
        double[] logProbs = new double[n];
        for (int i = 0; i < n; i++) {
            logProbs[i] = scaled[i] - logSumExp;
        }
        return logProbs;
    }
}
