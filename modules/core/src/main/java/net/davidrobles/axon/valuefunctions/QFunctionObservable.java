package net.davidrobles.axon.valuefunctions;

/**
 * Implemented by any object that exposes its Q-function updates to registered observers.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface QFunctionObservable<S, A> {
    void addQFunctionObserver(QFunctionObserver<S, A> observer);
}
