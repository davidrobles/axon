package net.davidrobles.axon.agents;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import net.davidrobles.axon.ReplayBuffer;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.GreedyPolicy;
import net.davidrobles.axon.valuefunctions.QFunction;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import org.junit.Before;
import org.junit.Test;

public class QLearningWithReplayTest {

    private static final double EPS = 1e-9;

    private TabularQFunction<String, String> table;

    @Before
    public void setUp() {
        table = new TabularQFunction<>(0.5);
    }

    private QLearningWithReplay<String, String> agent(int bufferCapacity, int batchSize) {
        return new QLearningWithReplay<>(
                table,
                new GreedyPolicy<>(table),
                0.9,
                new ReplayBuffer<>(bufferCapacity),
                batchSize,
                new Random(0));
    }

    // -------------------------------------------------------------------------
    // No update until buffer has enough transitions
    // -------------------------------------------------------------------------

    @Test
    public void noUpdateBeforeBufferReachesBatchSize() {
        QLearningWithReplay<String, String> a = agent(10, 3);
        a.update("s0", "a0", new StepResult<>("s1", 5.0, true), List.of());
        a.update("s1", "a0", new StepResult<>("s2", 5.0, true), List.of());
        // only 2 transitions stored, batchSize=3 → no update yet
        assertEquals(0.0, table.getValue("s0", "a0"), EPS);
        assertEquals(0.0, table.getValue("s1", "a0"), EPS);
    }

    @Test
    public void updatesBeginOnceBatchSizeReached() {
        QLearningWithReplay<String, String> a = agent(10, 2);
        a.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        assertEquals(0.0, table.getValue("s0", "a0"), EPS); // not yet

        a.update("s1", "a0", new StepResult<>("s2", 1.0, true), List.of());
        // now 2 transitions, batchSize=2 → updates fired
        assertTrue(table.getValue("s0", "a0") > 0.0 || table.getValue("s1", "a0") > 0.0);
    }

    // -------------------------------------------------------------------------
    // Q-Learning update rule applied to sampled transitions
    // -------------------------------------------------------------------------

    @Test
    public void terminalTransitionUpdatesWithRewardOnly() {
        // batchSize=1 so each step triggers exactly one update from the buffer
        QLearningWithReplay<String, String> a = agent(10, 1);
        a.update("s0", "a0", new StepResult<>("s1", 2.0, true), List.of());
        // target = 2.0; Q(s0,a0) = 0 + 0.5*(2-0) = 1.0
        assertEquals(1.0, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void nonTerminalTransitionBootstraps() {
        table.setValue("s1", "a0", 4.0);
        QLearningWithReplay<String, String> a = agent(10, 1);
        // target = 0.5 + 0.9*4.0 = 4.1; Q = 0 + 0.5*4.1 = 2.05
        a.update("s0", "a0", new StepResult<>("s1", 0.5, false), List.of("a0"));
        assertEquals(2.05, table.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Circular buffer: old transitions evicted
    // -------------------------------------------------------------------------

    @Test
    public void bufferEvictsOldTransitionsWhenFull() {
        // capacity=1: only the latest transition is ever in the buffer
        QLearningWithReplay<String, String> a = agent(1, 1);
        a.update("s0", "a0", new StepResult<>("s1", 10.0, true), List.of());
        double after1 = table.getValue("s0", "a0");

        a.update("s1", "a0", new StepResult<>("s2", 2.0, true), List.of());
        double s1After = table.getValue("s1", "a0");
        // s1 was the only thing in the buffer → it was sampled
        assertTrue(s1After > 0.0);
    }

    // -------------------------------------------------------------------------
    // selectAction delegates to the policy
    // -------------------------------------------------------------------------

    @Test
    public void selectActionDelegatesToPolicy() {
        table.setValue("s0", "a1", 10.0);
        assertEquals("a1", agent(10, 1).selectAction("s0", List.of("a0", "a1")));
    }

    // -------------------------------------------------------------------------
    // Observer — fired once per Q-Learning update in the batch
    // -------------------------------------------------------------------------

    @Test
    public void observerNotFiredBeforeBatchSizeReached() {
        AtomicInteger count = new AtomicInteger();
        QLearningWithReplay<String, String> a = agent(10, 3);
        a.addQFunctionObserver(qf -> count.incrementAndGet());
        a.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        a.update("s1", "a0", new StepResult<>("s2", 1.0, true), List.of());
        assertEquals(0, count.get());
    }

    @Test
    public void observerFiredBatchSizeTimesPerUpdate() {
        AtomicInteger count = new AtomicInteger();
        QLearningWithReplay<String, String> a = agent(10, 3);
        a.addQFunctionObserver(qf -> count.incrementAndGet());

        // Fill buffer to batchSize=3 → first batch update fires 3 notifications
        a.update("s0", "a0", new StepResult<>("s1", 1.0, true), List.of());
        a.update("s1", "a0", new StepResult<>("s2", 1.0, true), List.of());
        a.update("s2", "a0", new StepResult<>("s3", 1.0, true), List.of());
        assertEquals(3, count.get());

        // Each subsequent step also fires a batch of 3
        a.update("s3", "a0", new StepResult<>("s4", 1.0, true), List.of());
        assertEquals(6, count.get());
    }

    @Test
    public void observerReceivesCurrentQFunction() {
        QFunction<String, String>[] captured = new QFunction[1];
        QLearningWithReplay<String, String> a = agent(10, 1);
        a.addQFunctionObserver(qf -> captured[0] = qf);
        a.update("s0", "a0", new StepResult<>("s1", 2.0, true), List.of());
        assertNotNull(captured[0]);
        assertEquals(1.0, captured[0].getValue("s0", "a0"), EPS);
    }

    @Test
    public void duplicateObserverIsRegisteredOnce() {
        AtomicInteger count = new AtomicInteger();
        net.davidrobles.axon.valuefunctions.QFunctionObserver<String, String> o =
                qf -> count.incrementAndGet();
        QLearningWithReplay<String, String> a = agent(10, 1);
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
        new QLearningWithReplay<>(
                table, new GreedyPolicy<>(table), -0.1, new ReplayBuffer<>(10), 4, new Random(0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaAboveOneIsRejected() {
        new QLearningWithReplay<>(
                table, new GreedyPolicy<>(table), 1.1, new ReplayBuffer<>(10), 4, new Random(0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void batchSizeZeroIsRejected() {
        new QLearningWithReplay<>(
                table, new GreedyPolicy<>(table), 0.9, new ReplayBuffer<>(10), 0, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullTableIsRejected() {
        new QLearningWithReplay<>(
                null, new GreedyPolicy<>(table), 0.9, new ReplayBuffer<>(10), 4, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullPolicyIsRejected() {
        new QLearningWithReplay<>(table, null, 0.9, new ReplayBuffer<>(10), 4, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullBufferIsRejected() {
        new QLearningWithReplay<>(table, new GreedyPolicy<>(table), 0.9, null, 4, new Random(0));
    }

    @Test(expected = NullPointerException.class)
    public void nullRngIsRejected() {
        new QLearningWithReplay<>(
                table, new GreedyPolicy<>(table), 0.9, new ReplayBuffer<>(10), 4, null);
    }
}
