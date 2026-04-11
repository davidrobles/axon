package net.davidrobles.axon.valuefunctions;

import java.util.LinkedHashSet;
import java.util.Set;
import net.davidrobles.axon.Agent;

/**
 * Abstract base class for agents that expose Q-function updates to registered observers.
 *
 * <p>Handles the observer set and notification boilerplate so subclasses only need to call {@link
 * #notifyQFunctionObservers(QFunction)} after each update.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public abstract class AbstractQFunctionObservable<S, A>
        implements Agent<S, A>, QFunctionObservable<S, A> {
    private final Set<QFunctionObserver<S, A>> qFunctionObservers = new LinkedHashSet<>();

    @Override
    public void addQFunctionObserver(QFunctionObserver<S, A> observer) {
        qFunctionObservers.add(observer);
    }

    /**
     * Notifies all registered observers with the current Q-function estimate.
     *
     * @param qFunction the current Q-function
     */
    protected void notifyQFunctionObservers(QFunction<S, A> qFunction) {
        for (QFunctionObserver<S, A> observer : qFunctionObservers)
            observer.qFunctionUpdated(qFunction);
    }
}
