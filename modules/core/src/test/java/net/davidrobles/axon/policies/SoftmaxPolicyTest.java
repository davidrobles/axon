package net.davidrobles.axon.policies;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Random;
import net.davidrobles.axon.values.TabularQFunction;
import org.junit.Before;
import org.junit.Test;

public class SoftmaxPolicyTest {

    private static final double EPS = 1e-9;

    private TabularQFunction<String, String> q;

    @Before
    public void setUp() {
        q = new TabularQFunction<>(0.1);
    }

    // -------------------------------------------------------------------------
    // Probabilities sum to 1
    // -------------------------------------------------------------------------

    @Test
    public void probabilitiesSumToOne() {
        q.setValue("s0", "a0", 1.0);
        q.setValue("s0", "a1", 2.0);
        q.setValue("s0", "a2", 3.0);
        SoftmaxPolicy<String, String> policy = new SoftmaxPolicy<>(q, 1.0, new Random(0));
        List<String> actions = List.of("a0", "a1", "a2");
        double sum = 0.0;
        for (String a : actions) sum += policy.probability("s0", a, actions);
        assertEquals(1.0, sum, 1e-10);
    }

    @Test
    public void uniformQValuesYieldUniformDistribution() {
        // All Q=0 → all probs equal 1/n
        SoftmaxPolicy<String, String> policy = new SoftmaxPolicy<>(q, 1.0, new Random(0));
        List<String> actions = List.of("a0", "a1", "a2");
        for (String a : actions) {
            assertEquals(1.0 / 3.0, policy.probability("s0", a, actions), 1e-10);
        }
    }

    // -------------------------------------------------------------------------
    // Higher Q-values get higher probability
    // -------------------------------------------------------------------------

    @Test
    public void higherQValueYieldsHigherProbability() {
        q.setValue("s0", "a0", 0.0);
        q.setValue("s0", "a1", 5.0);
        SoftmaxPolicy<String, String> policy = new SoftmaxPolicy<>(q, 1.0, new Random(0));
        List<String> actions = List.of("a0", "a1");
        double p0 = policy.probability("s0", "a0", actions);
        double p1 = policy.probability("s0", "a1", actions);
        assertTrue(p1 > p0);
    }

    // -------------------------------------------------------------------------
    // Temperature effect
    // -------------------------------------------------------------------------

    @Test
    public void lowTemperatureConcentratesMassOnBestAction() {
        q.setValue("s0", "a0", 1.0);
        q.setValue("s0", "a1", 10.0);
        List<String> actions = List.of("a0", "a1");
        SoftmaxPolicy<String, String> highTemp = new SoftmaxPolicy<>(q, 100.0, new Random(0));
        SoftmaxPolicy<String, String> lowTemp = new SoftmaxPolicy<>(q, 0.01, new Random(0));
        double highTempP1 = highTemp.probability("s0", "a1", actions);
        double lowTempP1 = lowTemp.probability("s0", "a1", actions);
        assertTrue(lowTempP1 > highTempP1);
    }

    @Test
    public void veryHighTemperatureApproachesUniform() {
        q.setValue("s0", "a0", 0.0);
        q.setValue("s0", "a1", 1000.0); // huge difference, but temperature also huge
        List<String> actions = List.of("a0", "a1");
        SoftmaxPolicy<String, String> policy = new SoftmaxPolicy<>(q, 1e9, new Random(0));
        assertEquals(0.5, policy.probability("s0", "a0", actions), 1e-4);
        assertEquals(0.5, policy.probability("s0", "a1", actions), 1e-4);
    }

    // -------------------------------------------------------------------------
    // selectAction returns a valid action from the list
    // -------------------------------------------------------------------------

    @Test
    public void selectActionReturnsActionFromList() {
        q.setValue("s0", "a0", 1.0);
        q.setValue("s0", "a1", 2.0);
        SoftmaxPolicy<String, String> policy = new SoftmaxPolicy<>(q, 1.0, new Random(42));
        List<String> actions = List.of("a0", "a1");
        for (int i = 0; i < 50; i++) {
            assertTrue(actions.contains(policy.selectAction("s0", actions)));
        }
    }

    @Test
    public void selectActionWithSingleActionAlwaysReturnsThat() {
        SoftmaxPolicy<String, String> policy = new SoftmaxPolicy<>(q, 1.0, new Random(0));
        for (int i = 0; i < 10; i++) {
            assertEquals("a0", policy.selectAction("s0", List.of("a0")));
        }
    }

    // -------------------------------------------------------------------------
    // Unknown action returns 0
    // -------------------------------------------------------------------------

    @Test
    public void unknownActionReturnsZero() {
        SoftmaxPolicy<String, String> policy = new SoftmaxPolicy<>(q, 1.0, new Random(0));
        assertEquals(0.0, policy.probability("s0", "unknown", List.of("a0", "a1")), EPS);
    }

    // -------------------------------------------------------------------------
    // Numerical stability with large Q-values
    // -------------------------------------------------------------------------

    @Test
    public void largeQValuesDoNotProduceNaN() {
        q.setValue("s0", "a0", 1e300);
        q.setValue("s0", "a1", 1e300);
        SoftmaxPolicy<String, String> policy = new SoftmaxPolicy<>(q, 1.0, new Random(0));
        List<String> actions = List.of("a0", "a1");
        double p = policy.probability("s0", "a0", actions);
        assertFalse(Double.isNaN(p));
        assertFalse(Double.isInfinite(p));
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void zeroTemperatureIsRejected() {
        new SoftmaxPolicy<>(q, 0.0, new Random(0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void negativeTemperatureIsRejected() {
        new SoftmaxPolicy<>(q, -1.0, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullQFuncIsRejected() {
        new SoftmaxPolicy<>(null, 1.0, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullRngIsRejected() {
        new SoftmaxPolicy<>(q, 1.0, null);
    }
}
