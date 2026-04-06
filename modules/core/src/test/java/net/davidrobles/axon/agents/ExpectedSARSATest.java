package net.davidrobles.axon.agents;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.EpsilonGreedy;
import net.davidrobles.axon.policies.RandomPolicy;
import net.davidrobles.axon.valuefunctions.QFunction;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import org.junit.Before;
import org.junit.Test;

public class ExpectedSARSATest {

    private static final double EPS = 1e-9;

    private TabularQFunction<String, String> table;
    private ExpectedSARSA<String, String> agent;

    @Before
    public void setUp() {
        table = new TabularQFunction<>(0.5);
        // RandomPolicy over 2 actions → π(a|s) = 0.5 each; makes expected value easy to compute
        agent = new ExpectedSARSA<>(table, new RandomPolicy<>(new Random(0)), 0.9);
    }

    // -------------------------------------------------------------------------
    // Update rule: Q(s,a) ← Q(s,a) + α*(r + γ*Σ π(a'|s')*Q(s',a') − Q(s,a))
    // -------------------------------------------------------------------------

    @Test
    public void updateNonTerminalWithUniformPolicy() {
        table.setValue("s1", "a0", 2.0);
        table.setValue("s1", "a1", 4.0);
        // RandomPolicy: π(a0|s1)=0.5, π(a1|s1)=0.5
        // expected = 0.5*2 + 0.5*4 = 3.0
        // target = 0 + 0.9*3.0 = 2.7; new Q = 0 + 0.5*(2.7-0) = 1.35
        agent.update("s0", "a0", new StepResult<>("s1", 0.0, false), List.of("a0", "a1"));
        assertEquals(1.35, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void updateTerminalIgnoresFutureReward() {
        table.setValue("s1", "a0", 100.0); // should be ignored
        // target = 1.0; new Q = 0 + 0.5*1.0 = 0.5
        agent.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        assertEquals(0.5, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void updateUsesExpectedNotSampledValue() {
        // With ε=0, EpsilonGreedy is purely greedy → π(greedy)=1, π(others)=0
        // expected = 1.0 * Q(s1, a1) = 5.0 (a1 is greedy)
        TabularQFunction<String, String> t = new TabularQFunction<>(0.5);
        t.setValue("s1", "a0", 1.0);
        t.setValue("s1", "a1", 5.0); // greedy
        ExpectedSARSA<String, String> greedyAgent =
                new ExpectedSARSA<>(t, new EpsilonGreedy<>(t, 0.0, new Random(0)), 1.0);
        // expected = 5.0; target = 0 + 1.0*5.0 = 5.0; new Q = 0 + 0.5*5.0 = 2.5
        greedyAgent.update("s0", "a0", new StepResult<>("s1", 0.0, false), List.of("a0", "a1"));
        assertEquals(2.5, t.getValue("s0", "a0"), EPS);
    }

    @Test
    public void updateWithGammaZeroIgnoresFuture() {
        ExpectedSARSA<String, String> noDiscount =
                new ExpectedSARSA<>(table, new RandomPolicy<>(new Random(0)), 0.0);
        table.setValue("s1", "a0", 999.0);
        // target = 2.0 + 0 = 2.0; new Q = 0 + 0.5*2.0 = 1.0
        noDiscount.update("s0", "a0", new StepResult<>("s1", 2.0, false), List.of("a0"));
        assertEquals(1.0, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void updateDoesNotChangeOtherQValues() {
        table.setValue("s0", "a1", 7.0);
        table.setValue("s1", "a0", 2.0);
        agent.update("s0", "a0", new StepResult<>("s1", 0.0, false), List.of("a0"));
        assertEquals(7.0, table.getValue("s0", "a1"), EPS);
    }

    @Test
    public void consecutiveUpdatesAccumulate() {
        // Step 1: target=2.0; Q(s0,a0) = 0 + 0.5*(2-0) = 1.0
        agent.update("s0", "a0", new StepResult<>("s1", 2.0, true), List.of());
        assertEquals(1.0, table.getValue("s0", "a0"), EPS);

        // Step 2: target=4.0; Q(s0,a0) = 1.0 + 0.5*(4-1.0) = 2.5
        agent.update("s0", "a0", new StepResult<>("s1", 4.0, true), List.of());
        assertEquals(2.5, table.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // selectAction delegates to the policy
    // -------------------------------------------------------------------------

    @Test
    public void selectActionDelegatesToPolicy() {
        TabularQFunction<String, String> t = new TabularQFunction<>(0.5);
        t.setValue("s0", "a1", 10.0);
        ExpectedSARSA<String, String> greedyAgent =
                new ExpectedSARSA<>(t, new EpsilonGreedy<>(t, 0.0, new Random(0)), 0.9);
        assertEquals("a1", greedyAgent.selectAction("s0", List.of("a0", "a1")));
    }

    // -------------------------------------------------------------------------
    // Observer
    // -------------------------------------------------------------------------

    @Test
    public void observerNotifiedOnEachUpdate() {
        AtomicInteger count = new AtomicInteger();
        agent.addQFunctionObserver(qf -> count.incrementAndGet());

        agent.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        agent.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        assertEquals(2, count.get());
    }

    @Test
    public void observerReceivesCurrentQFunction() {
        QFunction<String, String>[] captured = new QFunction[1];
        agent.addQFunctionObserver(qf -> captured[0] = qf);

        agent.update("s0", "a0", new StepResult<>("s1", 2.0, true), List.of());

        assertNotNull(captured[0]);
        assertEquals(1.0, captured[0].getValue("s0", "a0"), EPS); // 0 + 0.5*2 = 1.0
    }

    @Test
    public void duplicateObserverIsRegisteredOnce() {
        AtomicInteger count = new AtomicInteger();
        net.davidrobles.axon.valuefunctions.QFunctionObserver<String, String> o =
                qf -> count.incrementAndGet();
        agent.addQFunctionObserver(o);
        agent.addQFunctionObserver(o);

        agent.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        assertEquals(1, count.get());
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void gammaBelowZeroIsRejected() {
        new ExpectedSARSA<>(table, new RandomPolicy<>(new Random(0)), -0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaAboveOneIsRejected() {
        new ExpectedSARSA<>(table, new RandomPolicy<>(new Random(0)), 1.1);
    }

    @Test(expected = NullPointerException.class)
    public void nullTableIsRejected() {
        new ExpectedSARSA<>(null, new RandomPolicy<>(new Random(0)), 0.9);
    }

    @Test(expected = NullPointerException.class)
    public void nullPolicyIsRejected() {
        new ExpectedSARSA<>(table, null, 0.9);
    }
}
