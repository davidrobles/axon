package net.davidrobles.axon.valuefunctions;

/**
 * Observer notified whenever a {@link QFunctionObservable} updates its Q-function estimate.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface QFunctionObserver<S, A> {
    /**
     * Called after each Q-function update.
     *
     * @param qFunction the current Q-function estimate
     */
    void qFunctionUpdated(QFunction<S, A> qFunction);
}
