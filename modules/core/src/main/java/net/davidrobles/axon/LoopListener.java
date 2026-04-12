package net.davidrobles.axon;

/**
 * Listener for training-loop lifecycle events emitted by {@link InteractionLoop}.
 *
 * <p>Use this for schedules or other stateful behavior that should react to episode and step
 * boundaries without overloading the {@code Policy} abstraction.
 */
public interface LoopListener {
    /**
     * Called at the start of each episode before the environment is reset.
     *
     * @param episode the episode about to start (0-indexed)
     */
    default void onEpisodeStart(int episode) {}

    /**
     * Called after each environment step with the running total of steps taken.
     *
     * @param totalSteps running total of environment steps taken so far
     */
    default void onStep(int totalSteps) {}

    /**
     * Called at the end of each episode.
     *
     * @param episode the episode just completed (0-indexed)
     */
    default void onEpisodeEnd(int episode) {}
}
