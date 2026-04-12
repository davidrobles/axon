package net.davidrobles.axon.agents;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.policies.GreedyPolicy;
import net.davidrobles.axon.values.QFunction;
import net.davidrobles.axon.values.TabularQFunction;
import org.junit.Before;
import org.junit.Test;

public class DoubleQLearningTest {

    private static final double EPS = 1e-9;

    // Seeded randoms whose first nextBoolean() returns a known value:
    //   HEADS_RNG -> first call returns true  (updates qA, evaluates with qB)
    //   TAILS_RNG -> first call returns false (updates qB, evaluates with qA)
    private static final Random HEADS_RNG =
            new Random(0) {
                private boolean first = true;

                @Override
                public boolean nextBoolean() {
                    if (first) {
                        first = false;
                        return true;
                    }
                    return super.nextBoolean();
                }
            };

    private TabularQFunction<String, String> qA;
    private TabularQFunction<String, String> qB;

    @Before
    public void setUp() {
        qA = new TabularQFunction<>(0.5);
        qB = new TabularQFunction<>(0.5);
    }

    // Helper: build an agent whose coin flip is forced to "heads" (update qA) or "tails" (update
    // qB)
    private DoubleQLearning<String, String> headsAgent() {
        // Always returns true from nextBoolean
        Random alwaysHeads =
                new Random() {
                    @Override
                    public boolean nextBoolean() {
                        return true;
                    }
                };
        QFunction<String, String> composite =
                (s, a) -> (qA.getValue(s, a) + qB.getValue(s, a)) / 2.0;
        return new DoubleQLearning<>(qA, qB, new GreedyPolicy<>(composite), 0.9, alwaysHeads);
    }

    private DoubleQLearning<String, String> tailsAgent() {
        // Always returns false from nextBoolean
        Random alwaysTails =
                new Random() {
                    @Override
                    public boolean nextBoolean() {
                        return false;
                    }
                };
        QFunction<String, String> composite =
                (s, a) -> (qA.getValue(s, a) + qB.getValue(s, a)) / 2.0;
        return new DoubleQLearning<>(qA, qB, new GreedyPolicy<>(composite), 0.9, alwaysTails);
    }

    // -------------------------------------------------------------------------
    // Heads path: update qA, select with qA, evaluate with qB
    // -------------------------------------------------------------------------

    @Test
    public void headsUpdatesQANotQB() {
        qB.setValue("s1", "a1", 10.0); // qB evaluates
        headsAgent().update(new Experience<>("s0", "a0", 1.0, "s1", false, List.of("a1")));
        // qA(s0,a0): 0 + 0.5*(1 + 0.9*qB(s1,a1) - 0) = 0.5*(1 + 9) = 5.0
        assertEquals(5.0, qA.getValue("s0", "a0"), EPS);
        // qB must be untouched for (s0,a0)
        assertEquals(0.0, qB.getValue("s0", "a0"), EPS);
    }

    @Test
    public void headsSelectsFromQAEvaluatesWithQB() {
        // qA says a2 is best; qB has different values
        qA.setValue("s1", "a1", 1.0);
        qA.setValue("s1", "a2", 5.0); // greedy in qA
        qB.setValue("s1", "a1", 9.0);
        qB.setValue("s1", "a2", 2.0); // qB evaluation of qA-greedy action
        headsAgent().update(new Experience<>("s0", "a0", 0.0, "s1", false, List.of("a1", "a2")));
        // target = 0 + 0.9 * qB(s1, a2) = 0.9 * 2.0 = 1.8
        // new qA(s0,a0) = 0 + 0.5 * (1.8 - 0) = 0.9
        assertEquals(0.9, qA.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Tails path: update qB, select with qB, evaluate with qA
    // -------------------------------------------------------------------------

    @Test
    public void tailsUpdatesQBNotQA() {
        qA.setValue("s1", "a1", 10.0); // qA evaluates
        tailsAgent().update(new Experience<>("s0", "a0", 1.0, "s1", false, List.of("a1")));
        // qB(s0,a0): 0 + 0.5*(1 + 0.9*10 - 0) = 0.5*10 = 5.0
        assertEquals(5.0, qB.getValue("s0", "a0"), EPS);
        assertEquals(0.0, qA.getValue("s0", "a0"), EPS);
    }

    @Test
    public void tailsSelectsFromQBEvaluatesWithQA() {
        qB.setValue("s1", "a1", 1.0);
        qB.setValue("s1", "a2", 5.0); // greedy in qB
        qA.setValue("s1", "a1", 9.0);
        qA.setValue("s1", "a2", 3.0); // qA evaluation of qB-greedy action
        tailsAgent().update(new Experience<>("s0", "a0", 0.0, "s1", false, List.of("a1", "a2")));
        // target = 0 + 0.9 * qA(s1, a2) = 0.9 * 3.0 = 2.7
        // new qB(s0,a0) = 0 + 0.5 * 2.7 = 1.35
        assertEquals(1.35, qB.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Terminal step: no bootstrap
    // -------------------------------------------------------------------------

    @Test
    public void terminalStepIgnoresFutureReward() {
        qA.setValue("s1", "a1", 100.0);
        qB.setValue("s1", "a1", 100.0);
        headsAgent().update(new Experience<>("s0", "a0", 2.0, "s1", true, List.of()));
        // target = 2.0;  new qA(s0,a0) = 0 + 0.5*2.0 = 1.0
        assertEquals(1.0, qA.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Gamma = 0 ignores future
    // -------------------------------------------------------------------------

    @Test
    public void gammaZeroIgnoresFuture() {
        Random alwaysHeads =
                new Random() {
                    @Override
                    public boolean nextBoolean() {
                        return true;
                    }
                };
        QFunction<String, String> composite = (s, a) -> qA.getValue(s, a);
        DoubleQLearning<String, String> agent =
                new DoubleQLearning<>(qA, qB, new GreedyPolicy<>(composite), 0.0, alwaysHeads);
        qB.setValue("s1", "a1", 999.0);
        agent.update(new Experience<>("s0", "a0", 3.0, "s1", false, List.of("a1")));
        // 0 + 0.5 * (3 + 0*999 - 0) = 1.5
        assertEquals(1.5, qA.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Observer receives average of both tables
    // -------------------------------------------------------------------------

    @Test
    public void observerReceivesAverageOfBothTables() {
        qA.setValue("s0", "a0", 4.0);
        qB.setValue("s0", "a0", 2.0);
        QFunction<String, String>[] captured = new QFunction[1];
        DoubleQLearning<String, String> agent = headsAgent();
        agent.addQFunctionObserver(qf -> captured[0] = qf);
        agent.update(new Experience<>("s1", "a1", 0.0, "s2", true, List.of()));
        assertNotNull(captured[0]);
        // average at (s0,a0) should still be (4+2)/2 = 3.0
        assertEquals(3.0, captured[0].getValue("s0", "a0"), EPS);
    }

    @Test
    public void observerNotifiedOnEveryUpdate() {
        AtomicInteger count = new AtomicInteger();
        headsAgent().addQFunctionObserver(qf -> count.incrementAndGet());
        // Use fresh agent with observer attached
        DoubleQLearning<String, String> agent = headsAgent();
        agent.addQFunctionObserver(qf -> count.incrementAndGet());
        agent.update(new Experience<>("s0", "a0", 1.0, "s1", true, List.of()));
        agent.update(new Experience<>("s0", "a0", 1.0, "s1", true, List.of()));
        assertEquals(2, count.get());
    }

    // -------------------------------------------------------------------------
    // selectAction delegates to the injected policy
    // -------------------------------------------------------------------------

    @Test
    public void selectActionDelegatesToPolicy() {
        // composite sees qA only (qB=0)
        qA.setValue("s0", "a1", 10.0);
        assertEquals("a1", headsAgent().selectAction("s0", List.of("a0", "a1")));
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void gammaBelowZeroIsRejected() {
        QFunction<String, String> c = (s, a) -> 0.0;
        new DoubleQLearning<>(qA, qB, new GreedyPolicy<>(c), -0.1, new Random(0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaAboveOneIsRejected() {
        QFunction<String, String> c = (s, a) -> 0.0;
        new DoubleQLearning<>(qA, qB, new GreedyPolicy<>(c), 1.1, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullQAIsRejected() {
        QFunction<String, String> c = (s, a) -> 0.0;
        new DoubleQLearning<>(null, qB, new GreedyPolicy<>(c), 0.9, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullQBIsRejected() {
        QFunction<String, String> c = (s, a) -> 0.0;
        new DoubleQLearning<>(qA, null, new GreedyPolicy<>(c), 0.9, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullPolicyIsRejected() {
        new DoubleQLearning<>(qA, qB, null, 0.9, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullRngIsRejected() {
        QFunction<String, String> c = (s, a) -> 0.0;
        new DoubleQLearning<>(qA, qB, new GreedyPolicy<>(c), 0.9, null);
    }
}
