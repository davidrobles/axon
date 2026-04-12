package net.davidrobles.axon;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;
import net.davidrobles.axon.policies.Policy;
import org.junit.Before;
import org.junit.Test;

public class InteractionLoopTest {

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /** Counts every Agent lifecycle call. */
    private static class CountingAgent implements Agent<Integer, String> {
        final List<Integer> statesSeen = new ArrayList<>();
        final List<String> actionsSeen = new ArrayList<>();
        int updateCount = 0;

        @Override
        public String selectAction(Integer state, List<String> actions) {
            statesSeen.add(state);
            return actions.get(0);
        }

        @Override
        public void update(Experience<Integer, String> exp) {
            actionsSeen.add(exp.action());
            updateCount++;
        }
    }

    /** Counts every loop lifecycle call. */
    private static class CountingListener implements LoopListener {
        final List<Integer> onEpisodeStartArgs = new ArrayList<>();
        final List<Integer> onStepArgs = new ArrayList<>();
        final List<Integer> onEpisodeEndArgs = new ArrayList<>();

        @Override
        public void onEpisodeStart(int episode) {
            onEpisodeStartArgs.add(episode);
        }

        @Override
        public void onStep(int totalSteps) {
            onStepArgs.add(totalSteps);
        }

        @Override
        public void onEpisodeEnd(int episode) {
            onEpisodeEndArgs.add(episode);
        }
    }

    private TestEnvironment env;
    private CountingAgent agent;
    private Policy<Integer, String> policy;
    private CountingListener listener;

    @Before
    public void setUp() {
        env = new TestEnvironment();
        agent = new CountingAgent();
        policy = (state, actions) -> actions.get(0);
        listener = new CountingListener();
    }

    private static class CountingPredictor implements Predictor<Integer> {
        final List<Integer> statesSeen = new ArrayList<>();
        final List<StepResult<Integer>> resultsSeen = new ArrayList<>();
        int observeCount = 0;

        @Override
        public void observe(Integer state, StepResult<Integer> result) {
            statesSeen.add(state);
            resultsSeen.add(result);
            observeCount++;
        }
    }

    // -------------------------------------------------------------------------
    // Episode count
    // -------------------------------------------------------------------------

    @Test
    public void zeroEpisodesRunsNoSteps() {
        InteractionLoop.run(env, agent, policy, 0);
        assertEquals(0, agent.updateCount);
        assertTrue(listener.onEpisodeStartArgs.isEmpty());
    }

    @Test
    public void singleEpisodeTwoSteps() {
        // TestEnvironment: step 0→1 (not done), step 1→2 (done)
        InteractionLoop.run(env, agent, policy, 1);
        assertEquals(2, agent.updateCount);
    }

    @Test
    public void threeEpisodesRunsSixUpdates() {
        InteractionLoop.run(env, agent, policy, 3);
        assertEquals(6, agent.updateCount);
    }

    // -------------------------------------------------------------------------
    // Policy lifecycle hook correctness
    // -------------------------------------------------------------------------

    @Test
    public void resetCalledOncePerEpisode() {
        InteractionLoop.run(env, agent, policy, 3, listener);
        assertEquals(List.of(0, 1, 2), listener.onEpisodeStartArgs);
    }

    @Test
    public void onEpisodeEndCalledOncePerEpisodeWithCorrectIndex() {
        InteractionLoop.run(env, agent, policy, 3, listener);
        assertEquals(List.of(0, 1, 2), listener.onEpisodeEndArgs);
    }

    @Test
    public void onStepCalledWithMonotonicallyIncreasingTotalSteps() {
        InteractionLoop.run(env, agent, policy, 2, listener);
        // 2 episodes × 2 steps each = 4 total steps, numbered 1..4
        assertEquals(List.of(1, 2, 3, 4), listener.onStepArgs);
    }

    @Test
    public void totalStepCountContinuesAcrossEpisodes() {
        InteractionLoop.run(env, agent, policy, 3, listener);
        // 6 steps total, numbered 1..6
        assertEquals(List.of(1, 2, 3, 4, 5, 6), listener.onStepArgs);
    }

    // -------------------------------------------------------------------------
    // Agent receives correct (state, action, result, nextActions)
    // -------------------------------------------------------------------------

    @Test
    public void agentSeesCorrectStateSequence() {
        InteractionLoop.run(env, agent, policy, 1);
        // selectAction called for state 0 and state 1
        assertEquals(List.of(0, 1), agent.statesSeen);
    }

    @Test
    public void agentUpdateReceivesTerminalNextActionsAsEmpty() {
        // Capture nextActions from last update call
        List<List<String>> nextActionsList = new ArrayList<>();
        Agent<Integer, String> capturingAgent =
                new Agent<>() {
                    @Override
                    public String selectAction(Integer s, List<String> a) {
                        return a.get(0);
                    }

                    @Override
                    public void update(Experience<Integer, String> exp) {
                        nextActionsList.add(exp.nextActions());
                    }
                };

        InteractionLoop.run(env, capturingAgent, policy, 1);
        // Second update (terminal step) should have empty nextActions
        assertTrue(nextActionsList.get(1).isEmpty());
    }

    @Test
    public void agentUpdateReceivesNonTerminalNextActionsNonEmpty() {
        List<List<String>> nextActionsList = new ArrayList<>();
        Agent<Integer, String> capturingAgent =
                new Agent<>() {
                    @Override
                    public String selectAction(Integer s, List<String> a) {
                        return a.get(0);
                    }

                    @Override
                    public void update(Experience<Integer, String> exp) {
                        nextActionsList.add(exp.nextActions());
                    }
                };

        InteractionLoop.run(env, capturingAgent, policy, 1);
        // First update (non-terminal step) should have non-empty nextActions
        assertFalse(nextActionsList.get(0).isEmpty());
    }

    // -------------------------------------------------------------------------
    // numEpisodes validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void negativeNumEpisodesThrows() {
        InteractionLoop.run(env, agent, policy, -1);
    }

    // -------------------------------------------------------------------------
    // Predictor overload
    // -------------------------------------------------------------------------

    @Test
    public void predictorLoopRunsOneObservePerStep() {
        CountingPredictor predictor = new CountingPredictor();
        InteractionLoop.run(env, policy, predictor, 3);
        assertEquals(6, predictor.observeCount);
    }

    @Test
    public void predictorLoopUsesPolicyForActionSelection() {
        CountingPredictor predictor = new CountingPredictor();
        InteractionLoop.run(env, policy, predictor, 1);
        assertEquals(List.of(0, 1), predictor.statesSeen);
    }

    @Test
    public void predictorLoopReceivesTerminalStepResult() {
        CountingPredictor predictor = new CountingPredictor();
        InteractionLoop.run(env, policy, predictor, 1);
        assertTrue(predictor.resultsSeen.get(1).done());
    }

    @Test
    public void predictorLoopInvokesPolicyLifecycleHooks() {
        CountingPredictor predictor = new CountingPredictor();
        InteractionLoop.run(env, policy, predictor, 2, listener);
        assertEquals(List.of(0, 1), listener.onEpisodeStartArgs);
        assertEquals(List.of(1, 2, 3, 4), listener.onStepArgs);
        assertEquals(List.of(0, 1), listener.onEpisodeEndArgs);
    }

    @Test(expected = IllegalArgumentException.class)
    public void predictorLoopRejectsNegativeNumEpisodes() {
        InteractionLoop.run(env, policy, new CountingPredictor(), -1);
    }

    // -------------------------------------------------------------------------
    // Integration: QLearning converges on TestEnvironment
    // -------------------------------------------------------------------------

    @Test
    public void qLearningLearnsPositiveQValueAfterManyEpisodes() {
        net.davidrobles.axon.values.TabularQFunction<Integer, String> q =
                new net.davidrobles.axon.values.TabularQFunction<>(0.1);
        net.davidrobles.axon.policies.EpsilonGreedy<Integer, String> epsilonGreedy =
                new net.davidrobles.axon.policies.EpsilonGreedy<>(q, 0.1, new java.util.Random(0));
        net.davidrobles.axon.agents.QLearning<Integer, String> ql =
                new net.davidrobles.axon.agents.QLearning<>(q, epsilonGreedy, 0.9);

        InteractionLoop.run(env, ql, epsilonGreedy, 500);

        // After 500 episodes the Q-values at state 0 and 1 should be positive
        assertTrue("Q(0,go) should be positive", q.getValue(0, TestEnvironment.GO) > 0.5);
        assertTrue("Q(1,go) should be positive", q.getValue(1, TestEnvironment.GO) > 0.8);
    }
}
