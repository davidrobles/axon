package net.davidrobles.axon.policies;

import static org.junit.Assert.*;

import java.util.List;
import org.junit.Before;
import org.junit.Test;

public class DeterministicPolicyTest {

    private DeterministicPolicy<String, String> policy;

    @Before
    public void setUp() {
        policy = new DeterministicPolicy<>();
    }

    // -------------------------------------------------------------------------
    // getAction / setAction / selectAction
    // -------------------------------------------------------------------------

    @Test
    public void getActionReturnsSetAction() {
        policy.setAction("s0", "a1");
        assertEquals("a1", policy.getAction("s0"));
    }

    @Test
    public void selectActionReturnsSetAction() {
        policy.setAction("s0", "a2");
        assertEquals("a2", policy.selectAction("s0", List.of("a0", "a1", "a2")));
    }

    @Test
    public void setActionOverwrites() {
        policy.setAction("s0", "a0");
        policy.setAction("s0", "a1");
        assertEquals("a1", policy.getAction("s0"));
        assertEquals("a1", policy.selectAction("s0", List.of("a0", "a1")));
    }

    @Test
    public void differentStatesAreIndependent() {
        policy.setAction("s0", "up");
        policy.setAction("s1", "down");
        assertEquals("up", policy.selectAction("s0", List.of("up", "down")));
        assertEquals("down", policy.selectAction("s1", List.of("up", "down")));
    }

    @Test
    public void unmappedStateReturnsNull() {
        assertNull(policy.getAction("unknown"));
        assertNull(policy.selectAction("unknown", List.of("a0")));
    }
}
