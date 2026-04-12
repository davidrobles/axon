package net.davidrobles.axon.prediction;

import net.davidrobles.axon.Predictor;
import net.davidrobles.axon.valuefunctions.AbstractVFunctionObservable;

/**
 * Abstract base class for tabular V-function prediction algorithms.
 *
 * <p>Prediction algorithms estimate state values under an externally supplied policy; they do not
 * select actions themselves. Subclasses only need to implement {@link Predictor#observe}.
 *
 * @param <S> the type of the states
 */
public abstract class AbstractPredictor<S> extends AbstractVFunctionObservable<S>
        implements Predictor<S> {}
