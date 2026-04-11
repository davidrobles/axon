package net.davidrobles.axon.policies;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Random;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import org.junit.Before;
import org.junit.Test;

public class EpsilonGreedyTest {

    private static final double EPS = 1e-9;

    private TabularQFunction<String, String> q;

    @Before
    public void setUp() {
        q = new TabularQFunction<>(0.5);
        q.setValue("s0", "a0", 1.0);
        q.setValue("s0", "a1", 5.0); // a1 is greedy best
    }

    // -------------------------------------------------------------------------
    // Deterministic behaviour (epsilon=0 or epsilon=1 with single action)
    // -------------------------------------------------------------------------

    @Test
    public void epsilonZeroAlwaysGreedy() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 0.0, new Random(0));
        // epsilon=0 → always greedy → a1 (higher Q)
        for (int i = 0; i < 20; i++) {
            assertEquals("a1", policy.selectAction("s0", List.of("a0", "a1")));
        }
    }

    @Test
    public void epsilonOneWithSingleActionAlwaysReturnsThatAction() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 1.0, new Random(0));
        for (int i = 0; i < 20; i++) {
            assertEquals("a0", policy.selectAction("s0", List.of("a0")));
        }
    }

    @Test
    public void epsilonZeroInTrainingModeIsStillGreedy() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 0.0, new Random(0));
        policy.setTrainingMode(true);
        assertEquals("a1", policy.selectAction("s0", List.of("a0", "a1")));
    }

    // -------------------------------------------------------------------------
    // Training mode toggle
    // -------------------------------------------------------------------------

    @Test
    public void setTrainingModeFalseDisablesExploration() {
        // epsilon=1 normally means always random; with training=false it should be greedy
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 1.0, new Random(0));
        policy.setTrainingMode(false);
        for (int i = 0; i < 20; i++) {
            assertEquals("a1", policy.selectAction("s0", List.of("a0", "a1")));
        }
    }

    // -------------------------------------------------------------------------
    // getEpsilon / setEpsilon
    // -------------------------------------------------------------------------

    @Test
    public void getEpsilonReturnsInitialValue() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 0.3, new Random(0));
        assertEquals(0.3, policy.getEpsilon(), EPS);
    }

    @Test
    public void setEpsilonUpdatesCurrentValue() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 0.5, new Random(0));
        policy.setEpsilon(0.0);
        assertEquals(0.0, policy.getEpsilon(), EPS);
        // Now greedy
        assertEquals("a1", policy.selectAction("s0", List.of("a0", "a1")));
    }

    // -------------------------------------------------------------------------
    // Linear epsilon decay
    // -------------------------------------------------------------------------

    @Test
    public void fixedEpsilonConstructorDoesNotDecay() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 0.5, new Random(0));
        for (int ep = 0; ep < 10; ep++) policy.onEpisodeEnd(ep);
        assertEquals(0.5, policy.getEpsilon(), EPS);
    }

    @Test
    public void decayReachesEndEpsilonAfterDecayEpisodes() {
        // decayEpisodes=10, start=1.0, end=0.0
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 1.0, 0.0, 10, new Random(0));

        // After episode 9 (ep=9): t = min(1, 10/10) = 1 → epsilon = 1 + 1*(0-1) = 0
        policy.onEpisodeEnd(9);
        assertEquals(0.0, policy.getEpsilon(), EPS);
    }

    @Test
    public void decayIsLinearAtMidpoint() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 1.0, 0.0, 10, new Random(0));

        // After episode 4 (ep=4): t = min(1, 5/10) = 0.5 → epsilon = 1 + 0.5*(0-1) = 0.5
        policy.onEpisodeEnd(4);
        assertEquals(0.5, policy.getEpsilon(), EPS);
    }

    @Test
    public void decayDoesNotGoBelowEndEpsilon() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 1.0, 0.1, 5, new Random(0));
        // Run many more episodes than decayEpisodes
        for (int ep = 0; ep < 100; ep++) policy.onEpisodeEnd(ep);
        assertEquals(0.1, policy.getEpsilon(), EPS);
    }

    @Test
    public void singleDecayEpisodeDoesNotDecay() {
        // decayEpisodes=1 → onEpisodeEnd is a no-op (guarded by decayEpisodes <= 1)
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 0.9, 0.1, 1, new Random(0));
        policy.onEpisodeEnd(0);
        assertEquals(0.9, policy.getEpsilon(), EPS);
    }

    // -------------------------------------------------------------------------
    // probability: π(greedy) = ε/n + (1-ε)/numTied, π(non-greedy) = ε/n
    // -------------------------------------------------------------------------

    @Test
    public void probabilityGreedyActionHigherThanNonGreedy() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 0.1, new Random(0));
        // a1 is greedy (Q=5.0 > Q=1.0)
        double pGreedy = policy.probability("s0", "a1", List.of("a0", "a1"));
        double pNonGreedy = policy.probability("s0", "a0", List.of("a0", "a1"));
        assertTrue(pGreedy > pNonGreedy);
    }

    @Test
    public void probabilitiesSumToOne() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 0.2, new Random(0));
        List<String> actions = List.of("a0", "a1");
        double sum = 0.0;
        for (String a : actions) sum += policy.probability("s0", a, actions);
        assertEquals(1.0, sum, EPS);
    }

    @Test
    public void probabilityEpsilonZeroGreedyGetsAllMass() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 0.0, new Random(0));
        // ε=0 → greedy (a1) gets prob=1.0
        assertEquals(1.0, policy.probability("s0", "a1", List.of("a0", "a1")), EPS);
    }

    @Test
    public void probabilityEpsilonOneIsUniform() {
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(q, 1.0, new Random(0));
        // ε=1 → uniform: each action gets 1/n = 0.5
        assertEquals(0.5, policy.probability("s0", "a0", List.of("a0", "a1")), EPS);
        assertEquals(0.5, policy.probability("s0", "a1", List.of("a0", "a1")), EPS);
    }

    @Test
    public void probabilityTiedActionsShareGreedyMass() {
        // All Q-values equal → all actions tied → uniform distribution regardless of ε
        TabularQFunction<String, String> uniform = new TabularQFunction<>(0.5);
        EpsilonGreedy<String, String> policy = new EpsilonGreedy<>(uniform, 0.0, new Random(0));
        List<String> actions = List.of("a0", "a1", "a2");
        assertEquals(1.0 / 3.0, policy.probability("s0", "a0", actions), EPS);
    }
}
