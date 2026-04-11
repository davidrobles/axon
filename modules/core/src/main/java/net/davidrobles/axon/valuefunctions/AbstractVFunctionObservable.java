package net.davidrobles.axon.valuefunctions;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Abstract base class for objects that expose V-function updates to registered observers.
 *
 * <p>Handles the observer set and notification boilerplate so subclasses only need to call {@link
 * #notifyVFunctionObservers(VFunction)} after each update.
 *
 * @param <S> the type of the states
 */
public abstract class AbstractVFunctionObservable<S> implements VFunctionObservable<S> {
    private final Set<VFunctionObserver<S>> vFunctionObservers = new LinkedHashSet<>();

    @Override
    public void addVFunctionObserver(VFunctionObserver<S> observer) {
        vFunctionObservers.add(observer);
    }

    /**
     * Notifies all registered observers with the current V-function estimate.
     *
     * @param vFunction the current V-function
     */
    protected void notifyVFunctionObservers(VFunction<S> vFunction) {
        for (VFunctionObserver<S> observer : vFunctionObservers)
            observer.valueFunctionUpdated(vFunction);
    }
}
