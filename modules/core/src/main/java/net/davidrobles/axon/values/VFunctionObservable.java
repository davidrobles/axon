package net.davidrobles.axon.values;

/**
 * Implemented by any object that exposes its V-function updates to registered observers.
 *
 * @param <S> the type of the states
 */
public interface VFunctionObservable<S> {
    void addVFunctionObserver(VFunctionObserver<S> observer);
}
