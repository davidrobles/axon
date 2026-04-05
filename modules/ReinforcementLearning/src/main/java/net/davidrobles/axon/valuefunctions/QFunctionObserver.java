package net.davidrobles.axon.valuefunctions;

public interface QFunctionObserver<S, A> {
    void qFunctionUpdated(QFunction<S, A> qFunction);
}
