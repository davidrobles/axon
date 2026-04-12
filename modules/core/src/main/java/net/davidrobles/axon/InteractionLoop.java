package net.davidrobles.axon;

import java.util.List;
import net.davidrobles.axon.policies.Policy;

/**
 * Drives the standard RL episode loop: reset, select action, step, update — repeat.
 *
 * <p>Keeps the loop logic out of individual {@link Agent} implementations so each algorithm only
 * needs to define its update rule.
 *
 * <p>Loop listener lifecycle calls per episode:
 *
 * <ol>
 *   <li>{@link LoopListener#onEpisodeStart(int)} — start of episode
 *   <li>{@link LoopListener#onStep(int)} — after every step
 *   <li>{@link LoopListener#onEpisodeEnd(int)} — end of episode
 * </ol>
 */
public class InteractionLoop {
    private InteractionLoop() {}

    /**
     * Runs {@code numEpisodes} full episodes of interaction between {@code env} and {@code agent}.
     *
     * @param env the environment
     * @param agent the agent to train
     * @param policy the policy driving action selection
     * @param numEpisodes number of episodes to run; must be non-negative
     * @param <S> state type
     * @param <A> action type
     * @throws IllegalArgumentException if {@code numEpisodes} is negative
     */
    public static <S, A> void run(
            Environment<S, A> env, Agent<S, A> agent, Policy<S, A> policy, int numEpisodes) {
        run(env, agent, policy, numEpisodes, new LoopListener[0]);
    }

    /**
     * Runs {@code numEpisodes} full episodes of interaction between {@code env} and {@code agent}.
     *
     * @param env the environment
     * @param agent the agent to train
     * @param policy the policy driving action selection
     * @param numEpisodes number of episodes to run; must be non-negative
     * @param listeners loop listeners receiving training lifecycle callbacks
     * @param <S> state type
     * @param <A> action type
     * @throws IllegalArgumentException if {@code numEpisodes} is negative
     */
    public static <S, A> void run(
            Environment<S, A> env,
            Agent<S, A> agent,
            Policy<S, A> policy,
            int numEpisodes,
            LoopListener... listeners) {
        if (numEpisodes < 0)
            throw new IllegalArgumentException(
                    "numEpisodes must be non-negative, got: " + numEpisodes);
        int totalSteps = 0;

        for (int ep = 0; ep < numEpisodes; ep++) {
            notifyEpisodeStart(ep, listeners);
            S state = env.reset();
            List<A> actions = env.getActions(state);

            while (!actions.isEmpty()) {
                A action = agent.selectAction(state, actions);
                StepResult<S> result = env.step(action);
                List<A> nextActions =
                        result.done() ? List.of() : env.getActions(result.nextState());
                agent.update(
                        new Experience<>(
                                state,
                                action,
                                result.reward(),
                                result.nextState(),
                                result.done(),
                                nextActions));
                notifyStep(++totalSteps, listeners);

                if (result.done()) break;
                state = result.nextState();
                actions = nextActions;
            }

            notifyEpisodeEnd(ep, listeners);
        }
    }

    /**
     * Runs {@code numEpisodes} full episodes using an externally supplied policy and a predictor
     * that only updates state values.
     *
     * @param env the environment
     * @param policy the policy driving action selection
     * @param predictor the predictor to update from observed transitions
     * @param numEpisodes number of episodes to run; must be non-negative
     * @param <S> state type
     * @param <A> action type
     * @throws IllegalArgumentException if {@code numEpisodes} is negative
     */
    public static <S, A> void run(
            Environment<S, A> env, Policy<S, A> policy, Predictor<S> predictor, int numEpisodes) {
        run(env, policy, predictor, numEpisodes, new LoopListener[0]);
    }

    /**
     * Runs {@code numEpisodes} full episodes using an externally supplied policy and a predictor
     * that only updates state values.
     *
     * @param env the environment
     * @param policy the policy driving action selection
     * @param predictor the predictor to update from observed transitions
     * @param numEpisodes number of episodes to run; must be non-negative
     * @param listeners loop listeners receiving training lifecycle callbacks
     * @param <S> state type
     * @param <A> action type
     * @throws IllegalArgumentException if {@code numEpisodes} is negative
     */
    public static <S, A> void run(
            Environment<S, A> env,
            Policy<S, A> policy,
            Predictor<S> predictor,
            int numEpisodes,
            LoopListener... listeners) {
        if (numEpisodes < 0)
            throw new IllegalArgumentException(
                    "numEpisodes must be non-negative, got: " + numEpisodes);
        int totalSteps = 0;

        for (int ep = 0; ep < numEpisodes; ep++) {
            notifyEpisodeStart(ep, listeners);
            S state = env.reset();
            List<A> actions = env.getActions(state);

            while (!actions.isEmpty()) {
                A action = policy.selectAction(state, actions);
                StepResult<S> result = env.step(action);
                predictor.observe(state, result);
                notifyStep(++totalSteps, listeners);

                if (result.done()) break;
                state = result.nextState();
                actions = env.getActions(state);
            }

            notifyEpisodeEnd(ep, listeners);
        }
    }

    private static void notifyEpisodeStart(int episode, LoopListener... listeners) {
        for (LoopListener listener : listeners) {
            listener.onEpisodeStart(episode);
        }
    }

    private static void notifyStep(int totalSteps, LoopListener... listeners) {
        for (LoopListener listener : listeners) {
            listener.onStep(totalSteps);
        }
    }

    private static void notifyEpisodeEnd(int episode, LoopListener... listeners) {
        for (LoopListener listener : listeners) {
            listener.onEpisodeEnd(episode);
        }
    }
}
