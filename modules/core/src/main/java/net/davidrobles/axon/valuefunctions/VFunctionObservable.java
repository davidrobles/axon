package net.davidrobles.axon.valuefunctions;

import net.davidrobles.axon.Agent;

/**
 * An {@link Agent} that exposes its V-function updates to registered observers.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public interface VFunctionObservable<S, A> extends Agent<S, A> {
    void addVFunctionObserver(VFunctionObserver<S> observer);
}
