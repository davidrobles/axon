package net.davidrobles.axon.policies;

import static org.junit.Assert.*;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.Test;

public class RandomPolicyTest {

    private static final double EPS = 1e-9;

    @Test
    public void singleActionIsAlwaysSelected() {
        RandomPolicy<String, String> policy = new RandomPolicy<>(new Random(0));
        for (int i = 0; i < 10; i++) {
            assertEquals("only", policy.selectAction("s", List.of("only")));
        }
    }

    @Test
    public void returnsOnlyActionsFromGivenList() {
        RandomPolicy<String, String> policy = new RandomPolicy<>(new Random(42));
        List<String> actions = List.of("a0", "a1", "a2");
        Set<String> observed = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            String chosen = policy.selectAction("s", actions);
            observed.add(chosen);
            assertTrue(actions.contains(chosen));
        }
        // With 100 draws from 3 actions the probability of missing any is negligible
        assertEquals(3, observed.size());
    }

    @Test
    public void probabilityForSingleAction() {
        RandomPolicy<String, String> policy = new RandomPolicy<>(new Random(0));
        assertEquals(1.0, policy.probability("s", "a0", List.of("a0")), EPS);
    }

    @Test
    public void probabilityForTwoActions() {
        RandomPolicy<String, String> policy = new RandomPolicy<>(new Random(0));
        assertEquals(0.5, policy.probability("s", "a0", List.of("a0", "a1")), EPS);
        assertEquals(0.5, policy.probability("s", "a1", List.of("a0", "a1")), EPS);
    }

    @Test
    public void probabilityForFourActions() {
        RandomPolicy<String, String> policy = new RandomPolicy<>(new Random(0));
        List<String> actions = List.of("a0", "a1", "a2", "a3");
        for (String a : actions) {
            assertEquals(0.25, policy.probability("s", a, actions), EPS);
        }
    }

    @Test
    public void probabilitiesSumToOne() {
        RandomPolicy<String, String> policy = new RandomPolicy<>(new Random(0));
        List<String> actions = List.of("a0", "a1", "a2");
        double sum = 0.0;
        for (String a : actions) sum += policy.probability("s", a, actions);
        assertEquals(1.0, sum, EPS);
    }
}
