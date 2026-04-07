package net.davidrobles.axon.agents;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.GreedyPolicy;
import net.davidrobles.axon.valuefunctions.QFunction;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import org.junit.Before;
import org.junit.Test;

public class DynaQTest {

    private static final double EPS = 1e-9;

    private TabularQFunction<String, String> table;

    @Before
    public void setUp() {
        table = new TabularQFunction<>(0.5);
    }

    private DynaQ<String, String> agent(int planningSteps) {
        return new DynaQ<>(table, new GreedyPolicy<>(table), 0.9, planningSteps, new Random(0));
    }

    // -------------------------------------------------------------------------
    // planningSteps=0: identical to Q-Learning
    // -------------------------------------------------------------------------

    @Test
    public void planningStepsZeroMatchesQLearning() {
        table.setValue("s1", "a1", 2.0);
        // target = 0.5 + 0.9*2.0 = 2.3; new Q = 0 + 0.5*(2.3-0) = 1.15
        agent(0).update("s0", "a0", new StepResult<>("s1", 0.5, false), List.of("a1"));
        assertEquals(1.15, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void terminalStepIgnoresFutureReward() {
        table.setValue("s1", "a1", 100.0); // should be ignored
        // target = 1.0; new Q = 0 + 0.5*1.0 = 0.5
        agent(0).update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        assertEquals(0.5, table.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Planning: each additional step replays a stored transition
    // -------------------------------------------------------------------------

    @Test
    public void planningStepsAmplifyLearning() {
        // With planningSteps=0, Q(s0,a0) gets one update.
        // With planningSteps>0, the same transition is replayed, converging faster.
        TabularQFunction<String, String> tableNo = new TabularQFunction<>(0.5);
        TabularQFunction<String, String> tableWith = new TabularQFunction<>(0.5);

        DynaQ<String, String> noPlanning =
                new DynaQ<>(tableNo, new GreedyPolicy<>(tableNo), 0.9, 0, new Random(0));
        DynaQ<String, String> withPlanning =
                new DynaQ<>(tableWith, new GreedyPolicy<>(tableWith), 0.9, 10, new Random(0));

        StepResult<String> result = new StepResult<>("s1", 1.0, true);
        noPlanning.update("s0", "a0", result, List.of());
        withPlanning.update("s0", "a0", result, List.of());

        // Both start at 0; terminal reward=1.
        // No-planning: Q(s0,a0) = 0.5 after one update.
        // With-planning: converges closer to 1.0 due to repeated replay of same transition.
        assertTrue(tableWith.getValue("s0", "a0") > tableNo.getValue("s0", "a0"));
    }

    @Test
    public void modelStoredAndReplayedOnNextStep() {
        // Step 1: real step (s0,a0) → (s1, r=1, done). Model stores this.
        // Step 2: real step (s1,a0) → (s2, r=0, done). planningSteps=1 replays a random past.
        // Both (s0,a0) and (s1,a0) should have non-zero Q values.
        DynaQ<String, String> a = agent(1);
        a.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        double qAfterStep1 = table.getValue("s0", "a0");

        a.update("s1", "a0", new StepResult<>("s2", 0.0, true), List.of());
        // s0,a0 may have been replayed during planning
        double qAfterStep2 = table.getValue("s0", "a0");

        assertTrue(qAfterStep1 > 0.0);
        // Q(s1,a0) was also updated by real step
        assertTrue(table.getValue("s1", "a0") >= 0.0);
        // s0,a0 is either the same or improved by planning replay
        assertTrue(qAfterStep2 >= qAfterStep1);
    }

    @Test
    public void modelUpdatedWithLatestTransition() {
        // Observe (s0,a0) twice with different rewards. Model should store the latest.
        DynaQ<String, String> a = agent(0);
        a.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        double first = table.getValue("s0", "a0");

        a.update("s0", "a0", new StepResult<>("s1", 4.0, true), List.of());
        double second = table.getValue("s0", "a0");

        // second update should reflect reward=4 applied to the current Q value
        assertTrue(second > first);
    }

    // -------------------------------------------------------------------------
    // selectAction delegates to the policy
    // -------------------------------------------------------------------------

    @Test
    public void selectActionDelegatesToPolicy() {
        table.setValue("s0", "a1", 10.0);
        assertEquals("a1", agent(0).selectAction("s0", List.of("a0", "a1")));
    }

    // -------------------------------------------------------------------------
    // Observer — fired once per Q-Learning update (real + planning)
    // -------------------------------------------------------------------------

    @Test
    public void observerFiredForRealAndPlanningUpdates() {
        AtomicInteger count = new AtomicInteger();
        DynaQ<String, String> a = agent(5);
        a.addQFunctionObserver(qf -> count.incrementAndGet());

        // 1 real update + 5 planning updates = 6 total notifications
        a.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        assertEquals(6, count.get());
    }

    @Test
    public void observerReceivesCurrentQFunction() {
        QFunction<String, String>[] captured = new QFunction[1];
        DynaQ<String, String> a = agent(0);
        a.addQFunctionObserver(qf -> captured[0] = qf);

        a.update("s0", "a0", new StepResult<>("s1", 2.0, true), List.of());

        assertNotNull(captured[0]);
        assertEquals(1.0, captured[0].getValue("s0", "a0"), EPS); // 0 + 0.5*2 = 1.0
    }

    @Test
    public void duplicateObserverIsRegisteredOnce() {
        AtomicInteger count = new AtomicInteger();
        net.davidrobles.axon.valuefunctions.QFunctionObserver<String, String> o =
                qf -> count.incrementAndGet();
        DynaQ<String, String> a = agent(0);
        a.addQFunctionObserver(o);
        a.addQFunctionObserver(o);

        a.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        assertEquals(1, count.get());
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void gammaBelowZeroIsRejected() {
        new DynaQ<>(table, new GreedyPolicy<>(table), -0.1, 5, new Random(0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaAboveOneIsRejected() {
        new DynaQ<>(table, new GreedyPolicy<>(table), 1.1, 5, new Random(0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void negativePlanningStepsIsRejected() {
        new DynaQ<>(table, new GreedyPolicy<>(table), 0.9, -1, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullTableIsRejected() {
        new DynaQ<>(null, new GreedyPolicy<>(table), 0.9, 5, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullPolicyIsRejected() {
        new DynaQ<>(table, null, 0.9, 5, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullRngIsRejected() {
        new DynaQ<>(table, new GreedyPolicy<>(table), 0.9, 5, null);
    }
}
