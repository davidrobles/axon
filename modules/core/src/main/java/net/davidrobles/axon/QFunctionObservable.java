package net.davidrobles.axon;

import net.davidrobles.axon.valuefunctions.QFunctionObserver;

/**
 * An {@link Agent} that exposes its Q-function updates to registered observers.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface QFunctionObservable<S, A> extends Agent<S, A> {
    void addQFunctionObserver(QFunctionObserver<S, A> observer);
}
