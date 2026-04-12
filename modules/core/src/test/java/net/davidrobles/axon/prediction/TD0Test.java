package net.davidrobles.axon.prediction;

import static org.junit.Assert.*;

import java.util.concurrent.atomic.AtomicInteger;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.values.TabularVFunction;
import net.davidrobles.axon.values.VFunction;
import org.junit.Before;
import org.junit.Test;

public class TD0Test {

    private static final double EPS = 1e-9;

    private TabularVFunction<String> table;
    private TD0<String> td0;

    @Before
    public void setUp() {
        table = new TabularVFunction<>(0.5);
        td0 = new TD0<>(table, 0.9);
    }

    // -------------------------------------------------------------------------
    // observe(): V(s) ← V(s) + α*(r + γ*V(s') − V(s))
    // -------------------------------------------------------------------------

    @Test
    public void observeNonTerminal() {
        table.setValue("s1", 2.0);
        // target = 1.0 + 0.9*2.0 = 2.8;  new V(s0) = 0 + 0.5*2.8 = 1.4
        td0.observe("s0", new StepResult<>("s1", 1.0, false));
        assertEquals(1.4, table.getValue("s0"), EPS);
    }

    @Test
    public void observeTerminalIgnoresFutureValue() {
        table.setValue("s1", 100.0); // should be ignored
        // target = 2.0 + 0 = 2.0;  new V(s0) = 0 + 0.5*2.0 = 1.0
        td0.observe("s0", new StepResult<>("s1", 2.0, true));
        assertEquals(1.0, table.getValue("s0"), EPS);
    }

    @Test
    public void observeWithGammaZeroIgnoresFuture() {
        TD0<String> noDiscount = new TD0<>(table, 0.0);
        table.setValue("s1", 999.0);
        // target = 3.0 + 0 = 3.0;  new V = 0 + 0.5*3.0 = 1.5
        noDiscount.observe("s0", new StepResult<>("s1", 3.0, false));
        assertEquals(1.5, table.getValue("s0"), EPS);
    }

    @Test
    public void observeDoesNotAffectOtherStates() {
        table.setValue("s1", 7.0);
        td0.observe("s0", new StepResult<>("s1", 0.0, false));
        assertEquals(7.0, table.getValue("s1"), EPS);
    }

    // -------------------------------------------------------------------------
    // Observer
    // -------------------------------------------------------------------------

    @Test
    public void observerNotifiedOnEachObserve() {
        AtomicInteger count = new AtomicInteger();
        td0.addVFunctionObserver(vf -> count.incrementAndGet());

        td0.observe("s0", new StepResult<>("s1", 1.0, true));
        td0.observe("s0", new StepResult<>("s1", 1.0, true));
        assertEquals(2, count.get());
    }

    @Test
    public void observerReceivesUpdatedVFunction() {
        VFunction<String>[] captured = new VFunction[1];
        td0.addVFunctionObserver(vf -> captured[0] = vf);

        td0.observe("s0", new StepResult<>("s1", 2.0, true));

        assertNotNull(captured[0]);
        assertEquals(1.0, captured[0].getValue("s0"), EPS); // 0 + 0.5*2 = 1.0
    }

    @Test
    public void duplicateObserverIsRegisteredOnce() {
        AtomicInteger count = new AtomicInteger();
        net.davidrobles.axon.values.VFunctionObserver<String> o =
                vf -> count.incrementAndGet();
        td0.addVFunctionObserver(o);
        td0.addVFunctionObserver(o);

        td0.observe("s0", new StepResult<>("s1", 1.0, true));
        assertEquals(1, count.get());
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void gammaBelowZeroIsRejected() {
        new TD0<>(table, -0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaAboveOneIsRejected() {
        new TD0<>(table, 1.1);
    }

    @Test(expected = NullPointerException.class)
    public void nullTableIsRejected() {
        new TD0<>(null, 0.9);
    }
}
