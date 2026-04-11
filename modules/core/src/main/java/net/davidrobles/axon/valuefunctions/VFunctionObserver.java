package net.davidrobles.axon.valuefunctions;

/**
 * Observer notified whenever a {@link VFunctionObservable} updates its V-function estimate.
 *
 * @param <S> the type of the states
 */
public interface VFunctionObserver<S> {
    /**
     * Called after each V-function update.
     *
     * @param vFunction the current V-function estimate
     */
    void valueFunctionUpdated(VFunction<S> vFunction);
}
