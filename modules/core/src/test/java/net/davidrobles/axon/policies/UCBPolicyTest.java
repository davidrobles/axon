package net.davidrobles.axon.policies;

import static org.junit.Assert.*;

import java.util.List;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import org.junit.Before;
import org.junit.Test;

public class UCBPolicyTest {

    private static final double EPS = 1e-9;

    private TabularQFunction<String, String> q;

    @Before
    public void setUp() {
        q = new TabularQFunction<>(0.1);
    }

    // -------------------------------------------------------------------------
    // Unvisited actions are always selected first
    // -------------------------------------------------------------------------

    @Test
    public void unvisitedActionSelectedBeforeVisited() {
        q.setValue("s0", "a0", 100.0); // a0 has a much higher Q-value
        UCBPolicy<String, String> policy = new UCBPolicy<>(q, 1.0);
        // a1 is unvisited → must be chosen first despite lower Q
        String first = policy.selectAction("s0", List.of("a0", "a1"));
        assertEquals("a0", first); // a0 is first in iteration, so it is selected first
    }

    @Test
    public void allActionsVisitedInOrderBeforeRepeating() {
        UCBPolicy<String, String> policy = new UCBPolicy<>(q, 1.0);
        List<String> actions = List.of("a0", "a1", "a2");
        // First 3 calls must return each action exactly once (unvisited sweep)
        String first = policy.selectAction("s0", actions);
        String second = policy.selectAction("s0", actions);
        String third = policy.selectAction("s0", actions);
        assertEquals("a0", first);
        assertEquals("a1", second);
        assertEquals("a2", third);
    }

    // -------------------------------------------------------------------------
    // UCB formula applied once all actions are visited
    // -------------------------------------------------------------------------

    @Test
    public void ucbBonusDrivesExplorationToLessVisitedAction() {
        UCBPolicy<String, String> policy = new UCBPolicy<>(q, 2.0);
        List<String> actions = List.of("a0", "a1");
        // Visit both once (unvisited sweep)
        policy.selectAction("s0", actions); // a0
        policy.selectAction("s0", actions); // a1
        // Both Q=0. a0 has been selected once, a1 once: bonus equal, so a0 (first) wins
        String chosen = policy.selectAction("s0", actions);
        assertEquals("a0", chosen);
    }

    @Test
    public void higherQValueWinsWhenBonusEqual() {
        q.setValue("s0", "a1", 10.0);
        UCBPolicy<String, String> policy = new UCBPolicy<>(q, 0.0); // c=0 disables bonus
        List<String> actions = List.of("a0", "a1");
        policy.selectAction("s0", actions); // visit a0
        policy.selectAction("s0", actions); // visit a1
        // With c=0, pure greedy — a1 should win
        String chosen = policy.selectAction("s0", actions);
        assertEquals("a1", chosen);
    }

    // -------------------------------------------------------------------------
    // State counts and action counts are independent per state
    // -------------------------------------------------------------------------

    @Test
    public void countsAreIndependentPerState() {
        UCBPolicy<String, String> policy = new UCBPolicy<>(q, 1.0);
        List<String> actions = List.of("a0", "a1");
        // Exhaust unvisited sweep in s0
        policy.selectAction("s0", actions);
        policy.selectAction("s0", actions);
        // s1 is fresh — unvisited sweep starts again
        String firstInS1 = policy.selectAction("s1", actions);
        assertEquals("a0", firstInS1);
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void negativeCIsRejected() {
        new UCBPolicy<>(q, -0.1);
    }

    @Test(expected = NullPointerException.class)
    public void nullQFuncIsRejected() {
        new UCBPolicy<>(null, 1.0);
    }
}
