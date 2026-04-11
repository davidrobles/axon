package net.davidrobles.axon.valuefunctions;

import net.davidrobles.axon.Agent;

/**
 * An {@link Agent} that exposes its Q-function updates to registered observers.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface QFunctionObservable<S, A> extends Agent<S, A> {
    void addQFunctionObserver(QFunctionObserver<S, A> observer);
}
