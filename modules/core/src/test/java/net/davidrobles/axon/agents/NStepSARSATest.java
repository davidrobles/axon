package net.davidrobles.axon.agents;

import static org.junit.Assert.*;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.policies.GreedyPolicy;
import net.davidrobles.axon.policies.RandomPolicy;
import net.davidrobles.axon.valuefunctions.QFunction;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import org.junit.Before;
import org.junit.Test;

public class NStepSARSATest {

    private static final double EPS = 1e-9;

    // alpha=1 (direct assignment), gamma=1 for clean arithmetic
    private TabularQFunction<String, String> table;

    @Before
    public void setUp() {
        table = new TabularQFunction<>(1.0);
    }

    private NStepSARSA<String, String> agent(int n) {
        return new NStepSARSA<>(table, new GreedyPolicy<>(table), n, 1.0);
    }

    private NStepSARSA<String, String> agent(int n, double gamma) {
        return new NStepSARSA<>(table, new GreedyPolicy<>(table), n, gamma);
    }

    // -------------------------------------------------------------------------
    // n=1 behaves like SARSA
    // -------------------------------------------------------------------------

    @Test
    public void n1SingleTerminalStep() {
        // G = r + 0 (done) = 2; Q(s0,a0) = 2
        agent(1).update(new Experience<>("s0", "a0", 2.0, "s1", true, List.of()));
        assertEquals(2.0, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void n1NonTerminalBootstraps() {
        table.setValue("s1", "a0", 3.0);
        // greedy policy picks a0 at s1; G = 1 + 1*Q(s1,a0) = 1+3 = 4; Q(s0,a0) = 4
        agent(1).update(new Experience<>("s0", "a0", 1.0, "s1", false, List.of("a0")));
        assertEquals(4.0, table.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // n=2: update delayed by 1 step
    // -------------------------------------------------------------------------

    @Test
    public void n2NoUpdateAfterFirstStep() {
        NStepSARSA<String, String> a = agent(2);
        a.update(new Experience<>("s0", "a0", 5.0, "s1", false, List.of("a0")));
        assertEquals(0.0, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void n2UpdateAfterSecondStep() {
        // gamma=1; episode: (s0,a0) -r=1-> (s1,a0) -r=2-> s2 (non-terminal)
        // At step 2: buffer=[(s0,a0,1),(s1,a0,2)], size==2
        // G(s0,a0) = 1 + 2 + Q(s2,greedy) = 3 + 0 = 3; Q(s0,a0) = 3
        NStepSARSA<String, String> a = agent(2);
        a.update(new Experience<>("s0", "a0", 1.0, "s1", false, List.of("a0")));
        a.update(new Experience<>("s1", "a0", 2.0, "s2", false, List.of("a0")));
        assertEquals(3.0, table.getValue("s0", "a0"), EPS);
        assertEquals(0.0, table.getValue("s1", "a0"), EPS); // not yet updated
    }

    @Test
    public void n2FlushesRemainingAtEpisodeEnd() {
        // gamma=1; episode: (s0,a0) -r=1-> (s1,a0) -r=2-> done
        // G(s0,a0) = 1+2+0 = 3; Q(s0,a0) = 3 → flush: G(s1,a0) = 2; Q(s1,a0) = 2
        NStepSARSA<String, String> a = agent(2);
        a.update(new Experience<>("s0", "a0", 1.0, "s1", false, List.of("a0")));
        a.update(new Experience<>("s1", "a0", 2.0, "s2", true, List.of()));
        assertEquals(3.0, table.getValue("s0", "a0"), EPS);
        assertEquals(2.0, table.getValue("s1", "a0"), EPS);
    }

    @Test
    public void n2ShortEpisodeFlushesAll() {
        // Episode shorter than n: (s0,a0) -r=5-> done
        NStepSARSA<String, String> a = agent(2);
        a.update(new Experience<>("s0", "a0", 5.0, "s1", true, List.of()));
        assertEquals(5.0, table.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Discounting
    // -------------------------------------------------------------------------

    @Test
    public void discountAppliedCorrectly() {
        // gamma=0.5, n=2; (s0,a0) -r=0-> (s1,a0) -r=4-> done
        // G(s0,a0) = 0 + 0.5*4 + 0 = 2; Q(s0,a0) = 2
        // flush: G(s1,a0) = 4; Q(s1,a0) = 4
        NStepSARSA<String, String> a = agent(2, 0.5);
        a.update(new Experience<>("s0", "a0", 0.0, "s1", false, List.of("a0")));
        a.update(new Experience<>("s1", "a0", 4.0, "s2", true, List.of()));
        assertEquals(2.0, table.getValue("s0", "a0"), EPS);
        assertEquals(4.0, table.getValue("s1", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Buffer cleared between episodes
    // -------------------------------------------------------------------------

    @Test
    public void bufferClearedBetweenEpisodes() {
        NStepSARSA<String, String> a = agent(2);
        a.update(new Experience<>("s0", "a0", 2.0, "s1", true, List.of()));
        assertEquals(2.0, table.getValue("s0", "a0"), EPS);

        a.update(new Experience<>("s0", "a0", 4.0, "s1", true, List.of()));
        assertEquals(4.0, table.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // selectAction delegates to the policy
    // -------------------------------------------------------------------------

    @Test
    public void selectActionDelegatesToPolicy() {
        table.setValue("s0", "a1", 10.0);
        assertEquals("a1", agent(2).selectAction("s0", List.of("a0", "a1")));
    }

    // -------------------------------------------------------------------------
    // Observer
    // -------------------------------------------------------------------------

    @Test
    public void observerNotNotifiedBeforeBufferFills() {
        AtomicInteger count = new AtomicInteger();
        NStepSARSA<String, String> a = agent(3);
        a.addQFunctionObserver(qf -> count.incrementAndGet());
        a.update(new Experience<>("s0", "a0", 1.0, "s1", false, List.of("a0")));
        a.update(new Experience<>("s1", "a0", 1.0, "s2", false, List.of("a0")));
        assertEquals(0, count.get());
    }

    @Test
    public void observerNotifiedOnEachQUpdate() {
        AtomicInteger count = new AtomicInteger();
        NStepSARSA<String, String> a = agent(2);
        a.addQFunctionObserver(qf -> count.incrementAndGet());
        // 3-step episode with n=2: (s0,a0) updated at step 2, (s1,a0) and (s2,a0) flushed → 3
        a.update(new Experience<>("s0", "a0", 1.0, "s1", false, List.of("a0")));
        a.update(new Experience<>("s1", "a0", 1.0, "s2", false, List.of("a0")));
        a.update(new Experience<>("s2", "a0", 1.0, "s3", true, List.of()));
        assertEquals(3, count.get());
    }

    @Test
    public void observerReceivesUpdatedQFunction() {
        QFunction<String, String>[] captured = new QFunction[1];
        NStepSARSA<String, String> a = agent(1);
        a.addQFunctionObserver(qf -> captured[0] = qf);
        a.update(new Experience<>("s0", "a0", 3.0, "s1", true, List.of()));
        assertNotNull(captured[0]);
        assertEquals(3.0, captured[0].getValue("s0", "a0"), EPS);
    }

    @Test
    public void duplicateObserverIsRegisteredOnce() {
        AtomicInteger count = new AtomicInteger();
        net.davidrobles.axon.valuefunctions.QFunctionObserver<String, String> o =
                qf -> count.incrementAndGet();
        NStepSARSA<String, String> a = agent(1);
        a.addQFunctionObserver(o);
        a.addQFunctionObserver(o);
        a.update(new Experience<>("s0", "a0", 1.0, "s1", true, List.of()));
        assertEquals(1, count.get());
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void nBelowOneIsRejected() {
        new NStepSARSA<>(table, new GreedyPolicy<>(table), 0, 0.9);
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaBelowZeroIsRejected() {
        new NStepSARSA<>(table, new RandomPolicy<>(new java.util.Random(0)), 2, -0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaAboveOneIsRejected() {
        new NStepSARSA<>(table, new RandomPolicy<>(new java.util.Random(0)), 2, 1.1);
    }

    @Test(expected = NullPointerException.class)
    public void nullTableIsRejected() {
        new NStepSARSA<>(null, new GreedyPolicy<>(table), 2, 0.9);
    }

    @Test(expected = NullPointerException.class)
    public void nullPolicyIsRejected() {
        new NStepSARSA<>(table, null, 2, 0.9);
    }
}
