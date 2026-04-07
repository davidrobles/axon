package net.davidrobles.axon.prediction;

import static org.junit.Assert.*;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.RandomPolicy;
import net.davidrobles.axon.valuefunctions.TabularVFunction;
import net.davidrobles.axon.valuefunctions.VFunction;
import org.junit.Before;
import org.junit.Test;

public class NStepTDTest {

    private static final double EPS = 1e-9;

    // alpha=1 (direct assignment), gamma=1 for clean arithmetic
    private TabularVFunction<String> table;

    @Before
    public void setUp() {
        table = new TabularVFunction<>(1.0);
    }

    private NStepTD<String, String> agent(int n) {
        return new NStepTD<>(table, new RandomPolicy<>(new java.util.Random(0)), n, 1.0);
    }

    private NStepTD<String, String> agent(int n, double gamma) {
        return new NStepTD<>(table, new RandomPolicy<>(new java.util.Random(0)), n, gamma);
    }

    // -------------------------------------------------------------------------
    // n=1 behaves like TD(0)
    // -------------------------------------------------------------------------

    @Test
    public void n1SingleTerminalStep() {
        // G = r + 0 (done) = 2; V(s0) = 2
        agent(1).observe("s0", new StepResult<>("s1", 2.0, true));
        assertEquals(2.0, table.getValue("s0"), EPS);
    }

    @Test
    public void n1NonTerminalBootstraps() {
        table.setValue("s1", 3.0);
        // G = 1 + 1*V(s1) = 1+3 = 4; V(s0) = 4
        agent(1).observe("s0", new StepResult<>("s1", 1.0, false));
        assertEquals(4.0, table.getValue("s0"), EPS);
    }

    @Test
    public void n1NoUpdateBeforeStep() {
        // n=1 updates immediately on each step — no buffering needed
        NStepTD<String, String> a = agent(1);
        a.observe("s0", new StepResult<>("s1", 0.0, false));
        // V(s1)=0 → G=0 → V(s0)=0 still (unchanged from default 0)
        assertEquals(0.0, table.getValue("s0"), EPS);
    }

    // -------------------------------------------------------------------------
    // n=2: update delayed by 1 step
    // -------------------------------------------------------------------------

    @Test
    public void n2NoUpdateAfterFirstStep() {
        NStepTD<String, String> a = agent(2);
        a.observe("s0", new StepResult<>("s1", 5.0, false));
        // buffer has 1 entry, not yet n=2 → no update
        assertEquals(0.0, table.getValue("s0"), EPS);
    }

    @Test
    public void n2UpdateAfterSecondStep() {
        // gamma=1; episode: s0 -r=1-> s1 -r=2-> s2 (non-terminal)
        // At step 2: buffer=[(s0,1),(s1,2)], size==2
        // G(s0) = 1 + 2 + V(s2) = 3 + 0 = 3; V(s0) = 3
        NStepTD<String, String> a = agent(2);
        a.observe("s0", new StepResult<>("s1", 1.0, false));
        a.observe("s1", new StepResult<>("s2", 2.0, false));
        assertEquals(3.0, table.getValue("s0"), EPS);
        assertEquals(0.0, table.getValue("s1"), EPS); // s1 not yet updated
    }

    @Test
    public void n2FlushesRemainingAtEpisodeEnd() {
        // gamma=1; episode: s0 -r=1-> s1 -r=2-> done
        // Step 2: buffer full (n=2), done=true
        //   G(s0) = 1 + 2 + 0 (done) = 3; V(s0) = 3  → remove s0
        //   flush: G(s1) = 2; V(s1) = 2
        NStepTD<String, String> a = agent(2);
        a.observe("s0", new StepResult<>("s1", 1.0, false));
        a.observe("s1", new StepResult<>("s2", 2.0, true));
        assertEquals(3.0, table.getValue("s0"), EPS);
        assertEquals(2.0, table.getValue("s1"), EPS);
    }

    @Test
    public void n2ShortEpisodeFlushesAll() {
        // Episode shorter than n: s0 -r=5-> done
        // buffer never fills to n=2; episode ends with flush:
        //   G(s0) = 5; V(s0) = 5
        NStepTD<String, String> a = agent(2);
        a.observe("s0", new StepResult<>("s1", 5.0, true));
        assertEquals(5.0, table.getValue("s0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Discounting
    // -------------------------------------------------------------------------

    @Test
    public void discountAppliedCorrectly() {
        // gamma=0.5, n=2; episode: s0 -r=0-> s1 -r=4-> done
        // G(s0) = 0 + 0.5*4 + 0 (done) = 2; V(s0) = 2
        // flush: G(s1) = 4; V(s1) = 4
        NStepTD<String, String> a = agent(2, 0.5);
        a.observe("s0", new StepResult<>("s1", 0.0, false));
        a.observe("s1", new StepResult<>("s2", 4.0, true));
        assertEquals(2.0, table.getValue("s0"), EPS);
        assertEquals(4.0, table.getValue("s1"), EPS);
    }

    // -------------------------------------------------------------------------
    // Buffer is cleared between episodes
    // -------------------------------------------------------------------------

    @Test
    public void bufferClearedBetweenEpisodes() {
        // Episode 1: s0 -r=2-> done → V(s0) = 2
        NStepTD<String, String> a = agent(2);
        a.observe("s0", new StepResult<>("s1", 2.0, true));
        assertEquals(2.0, table.getValue("s0"), EPS);

        // Episode 2: s0 -r=4-> done → V(s0) updated again: 2 + 1*(4-2) = 4
        a.observe("s0", new StepResult<>("s1", 4.0, true));
        assertEquals(4.0, table.getValue("s0"), EPS);
    }

    // -------------------------------------------------------------------------
    // update() delegates to observe()
    // -------------------------------------------------------------------------

    @Test
    public void updateDelegatesToObserve() {
        agent(1).update("s0", "a0", new StepResult<>("s1", 3.0, true), List.of());
        assertEquals(3.0, table.getValue("s0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Observer
    // -------------------------------------------------------------------------

    @Test
    public void observerNotNotifiedBeforeBufferFills() {
        AtomicInteger count = new AtomicInteger();
        NStepTD<String, String> a = agent(3);
        a.addVFunctionObserver(vf -> count.incrementAndGet());
        a.observe("s0", new StepResult<>("s1", 1.0, false));
        a.observe("s1", new StepResult<>("s2", 1.0, false));
        assertEquals(0, count.get());
    }

    @Test
    public void observerNotifiedOnEachValueUpdate() {
        AtomicInteger count = new AtomicInteger();
        NStepTD<String, String> a = agent(2);
        a.addVFunctionObserver(vf -> count.incrementAndGet());
        // 3-step episode with n=2: s0 updated at step 2, s1 and s2 flushed at end → 3 updates
        a.observe("s0", new StepResult<>("s1", 1.0, false));
        a.observe("s1", new StepResult<>("s2", 1.0, false));
        a.observe("s2", new StepResult<>("s3", 1.0, true));
        assertEquals(3, count.get());
    }

    @Test
    public void observerReceivesUpdatedVFunction() {
        VFunction<String>[] captured = new VFunction[1];
        NStepTD<String, String> a = agent(1);
        a.addVFunctionObserver(vf -> captured[0] = vf);
        a.observe("s0", new StepResult<>("s1", 3.0, true));
        assertNotNull(captured[0]);
        assertEquals(3.0, captured[0].getValue("s0"), EPS);
    }

    @Test
    public void duplicateObserverIsRegisteredOnce() {
        AtomicInteger count = new AtomicInteger();
        net.davidrobles.axon.valuefunctions.VFunctionObserver<String> o =
                vf -> count.incrementAndGet();
        NStepTD<String, String> a = agent(1);
        a.addVFunctionObserver(o);
        a.addVFunctionObserver(o);
        a.observe("s0", new StepResult<>("s1", 1.0, true));
        assertEquals(1, count.get());
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void nBelowOneIsRejected() {
        new NStepTD<>(table, new RandomPolicy<>(new java.util.Random(0)), 0, 0.9);
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaBelowZeroIsRejected() {
        new NStepTD<>(table, new RandomPolicy<>(new java.util.Random(0)), 2, -0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaAboveOneIsRejected() {
        new NStepTD<>(table, new RandomPolicy<>(new java.util.Random(0)), 2, 1.1);
    }

    @Test(expected = NullPointerException.class)
    public void nullTableIsRejected() {
        new NStepTD<>(null, new RandomPolicy<>(new java.util.Random(0)), 2, 0.9);
    }

    @Test(expected = NullPointerException.class)
    public void nullPolicyIsRejected() {
        new NStepTD<>(table, null, 2, 0.9);
    }
}
