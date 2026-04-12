package net.davidrobles.axon;

/**
 * A state-action pair, used as a key in tabular Q-function lookups and eligibility trace maps.
 *
 * @param <S> the type of the states
 * @param <A> the type of the actions
 */
public record StateActionPair<S, A>(S state, A action) {}
